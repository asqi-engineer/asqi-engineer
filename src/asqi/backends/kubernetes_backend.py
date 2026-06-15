"""Kubernetes Job-based implementation of ContainerBackend.

Dispatches ASQI containers as Kubernetes Jobs using the K8s Jobs API.
Images are assumed to be available in the cluster -- no pull logic is performed.

Each Job runs the workload container alongside a native sidecar container
(K8s >= 1.28 ``initContainers`` entry with ``restartPolicy: Always``) that
mediates S3 I/O via the shared ``emptyDir`` volume and a per-Job ConfigMap
carrying the workload's ``InputRef`` / ``OutputDestination`` (AIP-2207).

Required RBAC (see ``asqi/k8s/rbac.yaml`` in the installed package):
    A ServiceAccount with ``create / get / watch / delete`` permissions on
    ``batch/v1 Jobs``, ``get / list`` on ``core/v1 Pods``, and
    ``create / delete`` on ``core/v1 ConfigMaps`` in the target namespace.

Optional dependency:
    Install the Kubernetes Python client with::

        pip install 'asqi-engineer[k8s]'

Unsupported Docker-semantics fields (fail-closed):
    ``__volumes``   — the legacy Docker volume-passing key is rejected by
                       ``_extract_io_refs`` (use ``__inputs`` / ``__output``
                       with AIP-2207 ``InputRef`` / ``OutputDestination``).
    ``host_access`` — Docker socket / Docker-in-Docker access is not supported in K8s Jobs.
"""

import json
import logging
import os
import re
import time
import uuid
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Any, cast

import yaml
from asqi.config import ContainerConfig
from asqi.errors import ManifestExtractionError
from asqi.schemas import InputRef, Manifest, OutputDestination
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_K8S_SERVICE_LABEL = "asqi_engineer"
_JOB_NAME_PREFIX = "asqi-job-"
_DEFAULT_NAMESPACE = "default"
_POLL_INTERVAL_SECONDS = 2
_MANIFEST_EXTRACT_TIMEOUT = 60

# ── Sidecar / shared-volume constants ──────────────────────────────────────────
# Default sidecar image is a greppable placeholder so misconfiguration is
# obvious in logs. Helm (AIP-2475) sets AIP_SIDECAR_IMAGE; tests inject the
# image explicitly via the KubernetesBackend constructor.
_DEFAULT_SIDECAR_IMAGE = "aip-runtime-sidecar:placeholder"
_SIDECAR_IMAGE_ENV = "AIP_SIDECAR_IMAGE"
# Helm (AIP-2475) publishes these so the backend can schedule Job Pods under
# the IRSA-bound ServiceAccount and `envFrom` the sidecar's connection/S3 env
# off a ConfigMap + Secret. All optional: unset (local/dev) leaves Pods on the
# namespace default SA with only the per-Job artifact env vars.
_SIDECAR_SA_NAME_ENV = "AIP_SIDECAR_SA_NAME"
_SIDECAR_CONFIGMAP_ENV = "AIP_SIDECAR_CONFIGMAP"
_SIDECAR_SECRET_ENV = "AIP_SIDECAR_SECRET"  # noqa: S105 — env-var name, not a secret value
_SIDECAR_CONTAINER_NAME = "aip-runtime-sidecar"
_WORKLOAD_CONTAINER_NAME = "asqi-container"
_SHARED_VOLUME_NAME = "shared"
_IO_REFS_VOLUME_NAME = "io-refs"
_IO_REFS_CONFIGMAP_KEY = "io.json"
_IO_REFS_CONFIGMAP_MOUNT = "/etc/aip"
_SHARED_MOUNT_SIDECAR = "/shared"
_WORKLOAD_INPUT_MOUNT = "/input"
_WORKLOAD_OUTPUT_MOUNT = "/output"
_SHARED_INPUT_SUBPATH = "input"
_SHARED_OUTPUT_SUBPATH = "output"
# Sidecar window to finish uploading /output to S3 after the workload
# container exits and before the kubelet sends SIGKILL.
_TERMINATION_GRACE_PERIOD_SECONDS = 60
# Starter resource defaults for the sidecar container. Without these the
# sidecar runs at BestEffort QoS and is the first thing the kubelet evicts
# under memory pressure — bad for the post-workload upload window. Values
# are intentionally conservative placeholders; AIP-2475 (Helm) will tune
# them once the real sidecar binary (AIP-2208) has been profiled.
_SIDECAR_CPU_REQUEST = "100m"
_SIDECAR_CPU_LIMIT = "500m"
_SIDECAR_MEMORY_REQUEST = "256Mi"
_SIDECAR_MEMORY_LIMIT = "512Mi"
# Reserved JSON keys carried inside --test-params / --generation-params.
_PARAM_INPUTS_KEY = "__inputs"
_PARAM_OUTPUT_KEY = "__output"
_PARAM_LEGACY_VOLUMES_KEY = "__volumes"
_PARAM_FLAGS = ("--test-params", "--generation-params")


# ── K8s client helpers ─────────────────────────────────────────────────────────


def _load_k8s_clients() -> tuple[Any, Any]:
    """Load K8s config and return ``(BatchV1Api, CoreV1Api)``.

    Tries in-cluster config first (running inside a pod), then falls back to
    the local kubeconfig for development / CI environments.

    Raises:
        ImportError: If the ``kubernetes`` package is not installed.
    """
    try:
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config
    except ImportError as e:
        raise ImportError(
            "kubernetes package is required for KubernetesBackend. Install it with: pip install 'asqi-engineer[k8s]'"
        ) from e

    try:
        k8s_config.load_incluster_config()  # type: ignore[reportUnknownMemberType]
        logger.debug("Loaded in-cluster K8s config")
        return k8s_client.BatchV1Api(), k8s_client.CoreV1Api()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()  # type: ignore[reportUnknownMemberType]
        logger.debug("Loaded local kubeconfig")

    return k8s_client.BatchV1Api(), k8s_client.CoreV1Api()


# ── Name / resource helpers ────────────────────────────────────────────────────


def _make_job_name(hint: str | None = None) -> str:
    """Generate a DNS-safe K8s Job name (<= 63 chars)."""
    suffix = uuid.uuid4().hex[:8]
    if hint:
        safe = re.sub(r"[^a-z0-9]", "-", hint.lower())
        safe = re.sub(r"-+", "-", safe).strip("-")
        safe = safe[:40]
        return f"{_JOB_NAME_PREFIX}{safe}-{suffix}"
    return f"{_JOB_NAME_PREFIX}{suffix}"


def _mem_limit_to_k8s(mem_limit: str) -> str:
    """Convert Docker-style memory limit (``2g``, ``512m``) to K8s format (``2Gi``, ``512Mi``)."""
    if mem_limit.endswith("g"):
        return mem_limit[:-1] + "Gi"
    if mem_limit.endswith("m"):
        return mem_limit[:-1] + "Mi"
    return mem_limit


def _cpu_quota_to_k8s(cpu_period: int, cpu_quota: int) -> str:
    """Convert Docker CPU quota / period to a K8s CPU string (e.g. ``200000 / 100000`` → ``'2'``)."""
    if cpu_period and cpu_quota:
        cores = cpu_quota / cpu_period
        return str(int(cores)) if cores == int(cores) else str(max(0.001, round(cores, 3)))
    return "2"


# ── Args validation / IO-ref extraction ────────────────────────────────────────


@dataclass(frozen=True)
class IORefs:
    """Result of extracting AIP-2207 InputRef / OutputDestination from container args.

    Attributes:
        args: ``args`` with ``__inputs`` / ``__output`` keys stripped from any
            ``--test-params`` / ``--generation-params`` JSON payload. This is
            what the workload container actually receives — the workload sees
            local mount paths, never raw S3 refs.
        inputs: Validated ``InputRef`` list (possibly empty) destined for the
            sidecar's ``io.json`` ConfigMap.
        output: Validated ``OutputDestination`` (single, optional) destined
            for the same ConfigMap.
        error: Non-``None`` when extraction fails (malformed refs, legacy
            ``__volumes`` key, etc.). Callers MUST fail-closed and skip Job
            creation when ``error`` is set — no silent strip.
    """

    args: list[str] = field(default_factory=list[str])
    inputs: list[InputRef] = field(default_factory=list[InputRef])
    output: OutputDestination | None = None
    error: str | None = None


def _extract_io_refs(args: list[str]) -> IORefs:
    """Extract and validate ``__inputs`` / ``__output`` from container args.

    Walks ``args`` looking for ``--test-params`` / ``--generation-params``
    JSON payloads. For each payload:

    1. If it carries the legacy ``__volumes`` key, fail-closed with a message
       pointing at the AIP-2207 replacement.
    2. If it carries ``__inputs`` or ``__output``, validate via the AIP-2207
       Pydantic models and accumulate. Malformed refs fail-closed.
    3. Strip the reserved keys from the JSON payload so the workload never
       sees S3 refs — it gets local mount paths instead.

    Returns:
        :class:`IORefs` with ``error`` set iff validation failed. On success,
        ``args`` is a new list (the original is not mutated).
    """
    if not args:
        return IORefs(args=list(args))

    inputs: list[InputRef] = []
    output: OutputDestination | None = None
    new_args: list[str] = list(args)

    for i, arg in enumerate(new_args):
        if arg not in _PARAM_FLAGS or i + 1 >= len(new_args):
            continue
        raw = new_args[i + 1]
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        # json.loads gives Any; the isinstance check above narrows to dict but
        # pyright still types pop()/items as Unknown. cast keeps subsequent
        # access typed as Any (validated at runtime by the Pydantic models).
        payload_dict = cast(dict[str, Any], payload)

        if _PARAM_LEGACY_VOLUMES_KEY in payload_dict:
            return IORefs(
                args=list(args),
                error=(
                    "KubernetesBackend no longer supports volume-based file passing "
                    f"({_PARAM_LEGACY_VOLUMES_KEY}). Use {_PARAM_INPUTS_KEY} / "
                    f"{_PARAM_OUTPUT_KEY} with AIP-2207 InputRef / OutputDestination, "
                    "or RUN_BACKEND=docker for tests that still require __volumes."
                ),
            )

        raw_inputs: Any = payload_dict.pop(_PARAM_INPUTS_KEY, None)
        raw_output: Any = payload_dict.pop(_PARAM_OUTPUT_KEY, None)
        # Track whether we actually removed a reserved key so we only
        # rewrite the arg when we changed something — leaves the original
        # JSON string (whitespace, key order) untouched in the common
        # workload-args-without-IO-refs case.
        stripped = raw_inputs is not None or raw_output is not None

        if raw_inputs is not None:
            if not isinstance(raw_inputs, list):
                return IORefs(
                    args=list(args),
                    error=(f"{_PARAM_INPUTS_KEY} must be a list of InputRef objects; got {type(raw_inputs).__name__}."),
                )
            raw_inputs_list = cast(list[Any], raw_inputs)
            for j, item in enumerate(raw_inputs_list):
                try:
                    inputs.append(InputRef.model_validate(item))
                except ValidationError as e:
                    return IORefs(
                        args=list(args),
                        error=f"{_PARAM_INPUTS_KEY}[{j}] failed validation: {e}",
                    )

        if raw_output is not None:
            if output is not None:
                return IORefs(
                    args=list(args),
                    error=(
                        f"{_PARAM_OUTPUT_KEY} specified more than once across "
                        "--test-params / --generation-params; only one OutputDestination "
                        "is allowed per workload."
                    ),
                )
            try:
                output = OutputDestination.model_validate(raw_output)
            except ValidationError as e:
                return IORefs(
                    args=list(args),
                    error=f"{_PARAM_OUTPUT_KEY} failed validation: {e}",
                )

        # Re-serialise the stripped payload so the workload never sees S3 refs.
        if stripped:
            new_args[i + 1] = json.dumps(payload_dict)

    return IORefs(args=new_args, inputs=inputs, output=output)


def _build_io_refs_configmap_body(
    configmap_name: str,
    namespace: str,
    io_refs: IORefs,
    workflow_id: str,
) -> dict[str, Any]:
    """Build the per-Job ConfigMap that carries InputRef / OutputDestination.

    Single key ``io.json`` containing the validated refs serialised via the
    AIP-2207 Pydantic models. The sidecar reads this at startup to know what
    to download into ``/shared/input`` and where to publish ``/shared/output``.
    """
    payload: dict[str, Any] = {
        "inputs": [ref.model_dump(mode="json") for ref in io_refs.inputs],
        "output": io_refs.output.model_dump(mode="json") if io_refs.output is not None else None,
    }
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": namespace,
            "labels": {"workflow_id": workflow_id, "service": _K8S_SERVICE_LABEL},
        },
        "data": {_IO_REFS_CONFIGMAP_KEY: json.dumps(payload)},
    }


def _check_host_access(manifest: Manifest | None) -> str | None:
    """Return an error message if ``manifest.host_access`` is ``True``, else ``None``.

    Docker socket / Docker-in-Docker access (``host_access``) is not supported by the K8s backend.
    Callers should fail-closed when a non-``None`` value is returned.
    """
    if manifest is not None and manifest.host_access:
        return (
            "KubernetesBackend does not support host_access=True. "
            "Use RUN_BACKEND=docker for tests that require Docker socket or Docker-in-Docker access."
        )
    return None


# ── Job manifest builder ───────────────────────────────────────────────────────


def _build_job_body(
    job_name: str,
    image: str,
    args: list[str],
    environment: dict[str, str] | None,
    container_config: ContainerConfig,
    workflow_id: str,
    namespace: str,
    sidecar_image: str,
    io_refs_configmap_name: str,
    sidecar_sa_name: str | None = None,
    sidecar_configmap_name: str | None = None,
    sidecar_secret_name: str | None = None,
) -> dict[str, Any]:
    """Return a K8s Job manifest as a plain dict ready for the API client.

    The Pod spec includes:

    - A workload container (``asqi-container``) running ``image`` with the
      caller's ``args`` (already stripped of ``__inputs`` / ``__output`` by
      :func:`_extract_io_refs`) and mounts of the shared ``emptyDir`` at
      ``/input`` and ``/output`` via subPath.
    - A native sidecar container (``aip-runtime-sidecar``, K8s >= 1.28
      ``initContainers`` entry with ``restartPolicy: Always``) running
      ``sidecar_image`` with mounts of the shared ``emptyDir`` at ``/shared``
      and the per-Job ConfigMap at ``/etc/aip``.

    Helm-provided wiring (AIP-2475), all optional:

    - ``sidecar_sa_name``: when set, the Pod runs under this ServiceAccount so
      IRSA can grant the sidecar S3 access. When ``None`` the field is omitted
      and the Pod uses the namespace default SA.
    - ``sidecar_configmap_name`` / ``sidecar_secret_name``: when set, the
      sidecar container gets an ``envFrom`` referencing them, layering the
      runner-connection and S3 env underneath the per-Job artifact env vars.
      ``env`` wins over ``envFrom`` on key collision, so the artifact contract
      frozen by AIP-2473 is preserved.
    """
    env_list = [{"name": k, "value": v} for k, v in (environment or {}).items()]

    run_params = container_config.run_params
    k8s_mem = _mem_limit_to_k8s(str(run_params.get("mem_limit", "2g")))
    k8s_cpu = _cpu_quota_to_k8s(
        int(run_params.get("cpu_period", 100000)),
        int(run_params.get("cpu_quota", 200000)),
    )

    labels: dict[str, str] = {"workflow_id": workflow_id, "service": _K8S_SERVICE_LABEL}

    workload_container: dict[str, Any] = {
        "name": _WORKLOAD_CONTAINER_NAME,
        "image": image,
        "args": args,
        "env": env_list,
        "resources": {
            "limits": {"memory": k8s_mem, "cpu": k8s_cpu},
            "requests": {"memory": k8s_mem, "cpu": k8s_cpu},
        },
        "volumeMounts": [
            {
                "name": _SHARED_VOLUME_NAME,
                "mountPath": _WORKLOAD_INPUT_MOUNT,
                "subPath": _SHARED_INPUT_SUBPATH,
            },
            {
                "name": _SHARED_VOLUME_NAME,
                "mountPath": _WORKLOAD_OUTPUT_MOUNT,
                "subPath": _SHARED_OUTPUT_SUBPATH,
            },
        ],
    }

    # Native sidecar pattern: an initContainers entry with restartPolicy=Always
    # starts before the workload, runs alongside it, and is terminated after
    # the workload exits (subject to terminationGracePeriodSeconds).
    sidecar_container: dict[str, Any] = {
        "name": _SIDECAR_CONTAINER_NAME,
        "image": sidecar_image,
        "restartPolicy": "Always",
        "resources": {
            "limits": {"memory": _SIDECAR_MEMORY_LIMIT, "cpu": _SIDECAR_CPU_LIMIT},
            "requests": {"memory": _SIDECAR_MEMORY_REQUEST, "cpu": _SIDECAR_CPU_REQUEST},
        },
        "env": [
            {"name": "AIP_JOB_HANDLE", "value": workflow_id},
            {"name": "AIP_IO_REFS_PATH", "value": f"{_IO_REFS_CONFIGMAP_MOUNT}/{_IO_REFS_CONFIGMAP_KEY}"},
            {"name": "AIP_SHARED_DIR", "value": _SHARED_MOUNT_SIDECAR},
        ],
        "volumeMounts": [
            {"name": _SHARED_VOLUME_NAME, "mountPath": _SHARED_MOUNT_SIDECAR},
            {
                "name": _IO_REFS_VOLUME_NAME,
                "mountPath": _IO_REFS_CONFIGMAP_MOUNT,
                "readOnly": True,
            },
        ],
    }

    # Helm-provided runner-connection + S3 env (AIP-2475). envFrom is evaluated
    # before env, so the per-Job artifact vars above always win on collision —
    # the AIP-2473 contract stays frozen. Only emit envFrom for the refs that
    # are actually wired (unset → local/dev with no connection env).
    env_from: list[dict[str, Any]] = []
    if sidecar_configmap_name:
        env_from.append({"configMapRef": {"name": sidecar_configmap_name}})
    if sidecar_secret_name:
        env_from.append({"secretRef": {"name": sidecar_secret_name}})
    if env_from:
        sidecar_container["envFrom"] = env_from

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": _TERMINATION_GRACE_PERIOD_SECONDS,
        "initContainers": [sidecar_container],
        "containers": [workload_container],
        "volumes": [
            {"name": _SHARED_VOLUME_NAME, "emptyDir": {}},
            {
                "name": _IO_REFS_VOLUME_NAME,
                "configMap": {"name": io_refs_configmap_name},
            },
        ],
    }
    # Schedule the Pod under the IRSA-bound ServiceAccount when Helm supplies
    # one; otherwise leave the field unset so the namespace default SA is used.
    if sidecar_sa_name:
        pod_spec["serviceAccountName"] = sidecar_sa_name

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "ttlSecondsAfterFinished": 300,
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }


# ── Pod / log helpers ──────────────────────────────────────────────────────────


def _collect_pod_logs(core_api: Any, job_name: str, namespace: str) -> str:
    """Collect stdout/stderr logs from the pod created by a K8s Job."""
    try:
        from kubernetes.client.rest import ApiException
    except ImportError:
        return ""

    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )
        if not pods.items:
            logger.warning("No pods found for job '%s'", job_name)
            return ""
        pod_name = pods.items[0].metadata.name
        response = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            _preload_content=False,
        )
        logs = getattr(response, "data", response) or ""
        if isinstance(logs, bytes):
            return logs.decode("utf-8", errors="replace")
        if isinstance(logs, str) and logs.startswith(("b'", 'b"')):
            try:
                raw_logs = literal_eval(logs)
            except (SyntaxError, ValueError):
                return logs
            if isinstance(raw_logs, bytes):
                return raw_logs.decode("utf-8", errors="replace")
        return logs
    except ApiException as e:
        logger.warning("Failed to collect logs for job '%s': %s", job_name, e)
        return ""


def _get_pod_exit_code(core_api: Any, job_name: str, namespace: str) -> int:
    """Return the container exit code from the completed pod of a K8s Job."""
    try:
        from kubernetes.client.rest import ApiException
    except ImportError:
        return 1

    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )
        if not pods.items:
            return 1
        pod = pods.items[0]
        if pod.status and pod.status.container_statuses:
            cs = pod.status.container_statuses[0]
            if cs.state and cs.state.terminated:
                return cs.state.terminated.exit_code or 1
    except ApiException as e:
        logger.warning("Failed to read exit code for job '%s': %s", job_name, e)
    return 1


# ── Job lifecycle helpers ──────────────────────────────────────────────────────


def _wait_for_job(
    batch_api: Any,
    core_api: Any,
    job_name: str,
    namespace: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Poll a K8s Job until it succeeds, fails, or times out.

    Returns a result dict with ``success``, ``exit_code``, ``output``, ``error``,
    and ``container_id`` keys (matching the DockerBackend result schema).
    """
    try:
        from kubernetes.client.rest import ApiException
    except ImportError as e:
        raise ImportError("kubernetes package is required for KubernetesBackend") from e

    result: dict[str, Any] = {
        "success": False,
        "exit_code": -1,
        "output": "",
        "error": "",
        "container_id": job_name,
    }

    elapsed = 0
    while elapsed < timeout_seconds:
        try:
            job = batch_api.read_namespaced_job(name=job_name, namespace=namespace)
        except ApiException as e:
            result["error"] = f"Failed to read status for job '{job_name}': {e}"
            return result

        status = job.status

        if status.succeeded and status.succeeded > 0:
            result["success"] = True
            result["exit_code"] = 0
            result["output"] = _collect_pod_logs(core_api, job_name, namespace)
            return result

        if status.failed and status.failed > 0:
            exit_code = _get_pod_exit_code(core_api, job_name, namespace)
            failure_msg = ""
            if status.conditions:
                for cond in status.conditions:
                    if cond.type == "Failed" and cond.status == "True":
                        failure_msg = cond.message or ""
                        break
            result["exit_code"] = exit_code
            result["error"] = failure_msg or f"Job '{job_name}' failed with exit code {exit_code}"
            result["output"] = _collect_pod_logs(core_api, job_name, namespace)
            return result

        time.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS

    result["exit_code"] = 137
    result["error"] = f"Job '{job_name}' timed out after {timeout_seconds}s"
    return result


def _delete_job(batch_api: Any, job_name: str, namespace: str) -> None:
    """Delete a K8s Job and its pods (``propagationPolicy=Foreground``)."""
    try:
        from kubernetes import client as k8s_client
        from kubernetes.client.rest import ApiException
    except ImportError:
        return

    try:
        batch_api.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            body=k8s_client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        logger.debug("Deleted K8s Job '%s'", job_name)
    except ApiException as e:
        if e.status != 404:  # type: ignore[reportUnknownMemberType]
            logger.warning("Failed to delete job '%s': %s", job_name, e)


def _delete_configmap(core_api: Any, configmap_name: str, namespace: str) -> None:
    """Delete a per-Job io-refs ConfigMap. 404s are swallowed (idempotent cleanup)."""
    try:
        from kubernetes.client.rest import ApiException
    except ImportError:
        return

    try:
        core_api.delete_namespaced_config_map(name=configmap_name, namespace=namespace)
        logger.debug("Deleted io-refs ConfigMap '%s'", configmap_name)
    except ApiException as e:
        if e.status != 404:  # type: ignore[reportUnknownMemberType]
            logger.warning("Failed to delete ConfigMap '%s': %s", configmap_name, e)


# ── KubernetesBackend ──────────────────────────────────────────────────────────


class KubernetesBackend:
    """ContainerBackend implementation that runs containers as Kubernetes Jobs.

    Each call to :meth:`run` creates a K8s Job, waits for it to complete,
    collects its logs, then deletes the Job.  Images are assumed to be
    available in the cluster — no pull logic is performed.

    Args:
        namespace: Kubernetes namespace in which Jobs are created.
            Defaults to ``"default"``.

    RBAC requirement:
        The ServiceAccount must have ``create / get / watch / delete``
        permissions on ``batch/v1 Jobs`` and ``get / list`` on ``core/v1 Pods``
        in *namespace*.  See ``asqi/k8s/rbac.yaml`` in the installed
        package for a ready-to-apply manifest.
    """

    def __init__(
        self,
        namespace: str = _DEFAULT_NAMESPACE,
        *,
        sidecar_image: str | None = None,
        sidecar_sa_name: str | None = None,
        sidecar_configmap_name: str | None = None,
        sidecar_secret_name: str | None = None,
    ) -> None:
        """Initialise the backend.

        Args:
            namespace: Kubernetes namespace in which Jobs and per-Job
                ConfigMaps are created. Defaults to ``"default"``.
            sidecar_image: Image reference for the AIP Runtime sidecar
                container. When ``None``, falls back to the
                ``AIP_SIDECAR_IMAGE`` environment variable, then to a
                greppable placeholder (``aip-runtime-sidecar:placeholder``).
                Helm wires the production value via the env var; tests
                inject an explicit string here.
            sidecar_sa_name: ServiceAccount the per-Job Pods run under so IRSA
                can grant the sidecar S3 access (AIP-2475). ``None`` falls back
                to ``AIP_SIDECAR_SA_NAME``; absent → namespace default SA.
            sidecar_configmap_name: ConfigMap of runner-connection + S3 env
                ``envFrom``-ed onto each sidecar. ``None`` falls back to
                ``AIP_SIDECAR_CONFIGMAP``; absent → no ConfigMap envFrom.
            sidecar_secret_name: Secret of the runner admin token (and MinIO
                S3 keys) ``envFrom``-ed onto each sidecar. ``None`` falls back
                to ``AIP_SIDECAR_SECRET``; absent → no Secret envFrom.
        """
        self._namespace = namespace
        self._sidecar_image = sidecar_image or os.environ.get(_SIDECAR_IMAGE_ENV, _DEFAULT_SIDECAR_IMAGE)
        # Empty strings (e.g. an unset Helm value) are treated as "not wired"
        # so we never emit an empty serviceAccountName / envFrom ref.
        self._sidecar_sa_name = sidecar_sa_name or os.environ.get(_SIDECAR_SA_NAME_ENV) or None
        self._sidecar_configmap_name = sidecar_configmap_name or os.environ.get(_SIDECAR_CONFIGMAP_ENV) or None
        self._sidecar_secret_name = sidecar_secret_name or os.environ.get(_SIDECAR_SECRET_ENV) or None

    def run(
        self,
        image: str,
        args: list[str],
        container_config: ContainerConfig,
        environment: dict[str, str] | None = None,
        name: str | None = None,
        workflow_id: str = "",
        manifest: Manifest | None = None,
    ) -> dict[str, Any]:
        """Create a K8s Job for *image*, wait for completion, and return results.

        Args:
            image: Container image reference.
            args: Command-line arguments passed to the container. May contain
                ``__inputs`` / ``__output`` keys inside ``--test-params`` /
                ``--generation-params`` JSON payloads; these are extracted
                into the per-Job ConfigMap and stripped before reaching the
                workload (see :func:`_extract_io_refs`).
            container_config: Execution configuration (timeout, resource limits, etc.).
            environment: Optional environment variables injected into the workload container.
            name: Optional human-readable hint used to build the Job name.
            workflow_id: Workflow identifier attached as a Job label and as
                the sidecar's ``AIP_JOB_HANDLE`` env var.
            manifest: Optional manifest for the container image. ``host_access=True``
                causes an immediate fail-closed return without creating a Job.

        Returns:
            Dict with ``success``, ``exit_code``, ``output``, ``error``, ``container_id``.
        """
        io_refs = _extract_io_refs(args)
        if io_refs.error:
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "error": io_refs.error,
                "container_id": "",
            }

        host_access_error = _check_host_access(manifest)
        if host_access_error:
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "error": host_access_error,
                "container_id": "",
            }

        batch_api, core_api = _load_k8s_clients()
        job_name = _make_job_name(name)
        configmap_name = f"{job_name}-io-refs"

        configmap_body = _build_io_refs_configmap_body(
            configmap_name=configmap_name,
            namespace=self._namespace,
            io_refs=io_refs,
            workflow_id=workflow_id,
        )
        job_body = _build_job_body(
            job_name=job_name,
            image=image,
            args=io_refs.args,
            environment=environment,
            container_config=container_config,
            workflow_id=workflow_id,
            namespace=self._namespace,
            sidecar_image=self._sidecar_image,
            io_refs_configmap_name=configmap_name,
            sidecar_sa_name=self._sidecar_sa_name,
            sidecar_configmap_name=self._sidecar_configmap_name,
            sidecar_secret_name=self._sidecar_secret_name,
        )

        try:
            from kubernetes.client.rest import ApiException
        except ImportError as e:
            raise ImportError("kubernetes package is required for KubernetesBackend") from e

        try:
            core_api.create_namespaced_config_map(namespace=self._namespace, body=configmap_body)
            logger.debug("Created io-refs ConfigMap '%s'", configmap_name)
        except ApiException as e:
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "error": f"Failed to create io-refs ConfigMap for image '{image}': {e}",
                "container_id": "",
            }

        try:
            batch_api.create_namespaced_job(namespace=self._namespace, body=job_body)
            logger.info("Created K8s Job '%s' for image '%s'", job_name, image)
        except ApiException as e:
            _delete_configmap(core_api, configmap_name, self._namespace)
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "error": f"Failed to create K8s Job for image '{image}': {e}",
                "container_id": "",
            }

        result = _wait_for_job(
            batch_api=batch_api,
            core_api=core_api,
            job_name=job_name,
            namespace=self._namespace,
            timeout_seconds=container_config.timeout_seconds,
        )
        _delete_job(batch_api, job_name, self._namespace)
        _delete_configmap(core_api, configmap_name, self._namespace)
        return result

    def shutdown(self, workflow_ids: list[str] | None = None) -> None:
        """Delete all ASQI K8s Jobs, optionally scoped to specific workflow IDs.

        Args:
            workflow_ids: If ``None``, deletes all Jobs labelled
                ``service=asqi_engineer`` in the namespace.  Otherwise only
                Jobs matching the given workflow IDs are deleted.
        """
        batch_api, _ = _load_k8s_clients()

        if workflow_ids is None:
            try:
                jobs = batch_api.list_namespaced_job(
                    namespace=self._namespace,
                    label_selector=f"service={_K8S_SERVICE_LABEL}",
                )
                for job in jobs.items:
                    _delete_job(batch_api, job.metadata.name, self._namespace)
            except Exception as e:
                logger.error("Failed to list/delete ASQI K8s jobs: %s", e)
            return

        for wf_id in workflow_ids:
            try:
                jobs = batch_api.list_namespaced_job(
                    namespace=self._namespace,
                    label_selector=f"service={_K8S_SERVICE_LABEL},workflow_id={wf_id}",
                )
                for job in jobs.items:
                    _delete_job(batch_api, job.metadata.name, self._namespace)
            except Exception as e:
                logger.error("Failed to list/delete jobs for workflow '%s': %s", wf_id, e)

    def check_images(self, images: list[str]) -> dict[str, bool]:
        """Return ``True`` for every image -- K8s does not support pre-flight image checks.

        Image availability is determined at pod scheduling time by the kubelet.
        """
        return {image: True for image in images}

    def pull_images(self, images: list[str]) -> None:
        """No-op: image pulling is handled by the kubelet (``imagePullPolicy``).

        Images are assumed to be available in the cluster registry.
        """
        logger.info(
            "KubernetesBackend.pull_images: no-op -- image pulling is managed by the kubelet. "
            "%d image(s) assumed available in the cluster.",
            len(images),
        )

    def extract_manifest(self, image: str, manifest_path: str = ContainerConfig.MANIFEST_PATH) -> Manifest | None:
        """Extract ``manifest.yaml`` from an image by running a one-shot K8s Job.

        The Job runs ``cat <manifest_path>``; the pod stdout is parsed as YAML.

        Args:
            image: Container image reference.
            manifest_path: Path to the manifest file inside the container.

        Returns:
            Parsed :class:`~asqi.schemas.Manifest` or ``None``.

        Raises:
            ManifestExtractionError: If the Job fails or the YAML cannot be parsed.
        """
        batch_api, core_api = _load_k8s_clients()
        job_name = _make_job_name("manifest-extract")

        job_body: dict[str, Any] = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self._namespace,
                "labels": {"service": _K8S_SERVICE_LABEL},
            },
            "spec": {
                "ttlSecondsAfterFinished": 60,
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "manifest-extractor",
                                "image": image,
                                "command": ["cat", manifest_path],
                            }
                        ],
                    }
                },
            },
        }

        try:
            from kubernetes.client.rest import ApiException
        except ImportError as e:
            raise ImportError("kubernetes package is required for KubernetesBackend") from e

        try:
            batch_api.create_namespaced_job(namespace=self._namespace, body=job_body)
        except ApiException as e:
            raise ManifestExtractionError(
                f"Failed to create manifest extraction job for image '{image}': {e}",
                "K8S_JOB_CREATE_ERROR",
                e,
            ) from e

        try:
            result = _wait_for_job(
                batch_api=batch_api,
                core_api=core_api,
                job_name=job_name,
                namespace=self._namespace,
                timeout_seconds=_MANIFEST_EXTRACT_TIMEOUT,
            )

            if not result["success"]:
                raise ManifestExtractionError(
                    f"Manifest extraction job failed for image '{image}': {result['error']}",
                    "K8S_JOB_FAILED",
                )

            raw_yaml = result["output"]
            if not raw_yaml:
                raise ManifestExtractionError(
                    f"Manifest file '{manifest_path}' produced no output in image '{image}'",
                    "EMPTY_MANIFEST_FILE",
                )

            try:
                manifest_data = yaml.safe_load(raw_yaml)
            except yaml.YAMLError as e:
                raise ManifestExtractionError(
                    f"Failed to parse YAML manifest from image '{image}': {e}",
                    "YAML_PARSING_ERROR",
                    e,
                ) from e

            if manifest_data is None:
                raise ManifestExtractionError(
                    f"Manifest file from image '{image}' is empty or null",
                    "EMPTY_MANIFEST_FILE",
                )

            try:
                return Manifest(**manifest_data)
            except (TypeError, ValueError) as e:
                raise ManifestExtractionError(
                    f"Failed to validate manifest schema from image '{image}': {e}",
                    "SCHEMA_VALIDATION_ERROR",
                    e,
                ) from e

        finally:
            _delete_job(batch_api, job_name, self._namespace)
