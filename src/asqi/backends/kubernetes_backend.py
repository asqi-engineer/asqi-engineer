"""Kubernetes Job-based implementation of ContainerBackend.

Dispatches ASQI containers as Kubernetes Jobs using the K8s Jobs API.
Images are assumed to be available in the cluster -- no pull logic is performed.

Required RBAC (see ``k8s/rbac.yaml`` in the asqi-engineer package):
    A ServiceAccount with ``create / get / watch / delete`` permissions on
    ``batch/v1 Jobs`` (and ``get / list`` on ``core/v1 Pods``) in the target namespace.

Optional dependency:
    Install the Kubernetes Python client with::

        pip install 'asqi-engineer[k8s]'
"""

import json
import logging
import re
import time
import uuid
from typing import Any

import yaml
from asqi.config import ContainerConfig
from asqi.errors import ManifestExtractionError
from asqi.schemas import Manifest

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_K8S_SERVICE_LABEL = "asqi_engineer"
_JOB_NAME_PREFIX = "asqi-job-"
_DEFAULT_NAMESPACE = "default"
_POLL_INTERVAL_SECONDS = 2
_MANIFEST_EXTRACT_TIMEOUT = 60


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


# ── Args validation ────────────────────────────────────────────────────────────


def _check_no_volumes(args: list[str]) -> str | None:
    """Return an error message if ``__volumes`` is present in any param flag, else ``None``.

    Volume-based file passing (``__volumes``) is not yet supported by the K8s backend.
    Callers should fail-closed when a non-``None`` value is returned rather than
    silently stripping the key and producing a green run with no output artifacts.
    """
    if not args:
        return None

    param_flags = ("--test-params", "--generation-params")
    for i, arg in enumerate(args):
        if arg in param_flags and i + 1 < len(args):
            try:
                tp = json.loads(args[i + 1])
                if isinstance(tp, dict) and "__volumes" in tp:
                    return (
                        "KubernetesBackend does not yet support volume-based file passing "
                        "(__volumes). Use RUN_BACKEND=docker for tests that require file I/O "
                        "via __volumes, or wait for the shared-PVC implementation in a "
                        "follow-up sprint."
                    )
            except (json.JSONDecodeError, TypeError):
                continue
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
) -> dict[str, Any]:
    """Return a K8s Job manifest as a plain dict ready for the API client."""
    env_list = [{"name": k, "value": v} for k, v in (environment or {}).items()]

    run_params = container_config.run_params
    k8s_mem = _mem_limit_to_k8s(str(run_params.get("mem_limit", "2g")))
    k8s_cpu = _cpu_quota_to_k8s(
        int(run_params.get("cpu_period", 100000)),
        int(run_params.get("cpu_quota", 200000)),
    )

    labels: dict[str, str] = {"workflow_id": workflow_id, "service": _K8S_SERVICE_LABEL}

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
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "asqi-container",
                            "image": image,
                            "args": args,
                            "env": env_list,
                            "resources": {
                                "limits": {"memory": k8s_mem, "cpu": k8s_cpu},
                                "requests": {"memory": k8s_mem, "cpu": k8s_cpu},
                            },
                        }
                    ],
                },
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
        return core_api.read_namespaced_pod_log(name=pod_name, namespace=namespace) or ""
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
        in *namespace*.  See ``k8s/rbac.yaml`` for a ready-to-apply manifest.
    """

    def __init__(self, namespace: str = _DEFAULT_NAMESPACE) -> None:
        self._namespace = namespace

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
            args: Command-line arguments passed to the container.
            container_config: Execution configuration (timeout, resource limits, etc.).
            environment: Optional environment variables injected into the container.
            name: Optional human-readable hint used to build the Job name.
            workflow_id: Workflow identifier attached as a Job label.
            manifest: Unused by the K8s backend (kept for interface compatibility).

        Returns:
            Dict with ``success``, ``exit_code``, ``output``, ``error``, ``container_id``.
        """
        volumes_error = _check_no_volumes(args)
        if volumes_error:
            return {
                "success": False,
                "exit_code": -1,
                "output": "",
                "error": volumes_error,
                "container_id": "",
            }

        batch_api, core_api = _load_k8s_clients()
        job_name = _make_job_name(name)

        job_body = _build_job_body(
            job_name=job_name,
            image=image,
            args=args,
            environment=environment,
            container_config=container_config,
            workflow_id=workflow_id,
            namespace=self._namespace,
        )

        try:
            from kubernetes.client.rest import ApiException
        except ImportError as e:
            raise ImportError("kubernetes package is required for KubernetesBackend") from e

        try:
            batch_api.create_namespaced_job(namespace=self._namespace, body=job_body)
            logger.info("Created K8s Job '%s' for image '%s'", job_name, image)
        except ApiException as e:
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
