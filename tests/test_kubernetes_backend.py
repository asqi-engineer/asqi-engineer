"""Unit tests for KubernetesBackend.

All Kubernetes API calls are mocked -- no cluster is required.
The ``kubernetes`` package must be installed (added to the ``k8s`` optional dependency group).
"""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("kubernetes", reason="kubernetes package not installed — skipping K8s backend tests")

import yaml
from asqi.backends.base import ContainerBackend
from asqi.backends.kubernetes_backend import (
    _DEFAULT_SIDECAR_IMAGE,
    _IO_REFS_CONFIGMAP_KEY,
    _IO_REFS_CONFIGMAP_MOUNT,
    _IO_REFS_VOLUME_NAME,
    _SHARED_INPUT_SUBPATH,
    _SHARED_MOUNT_SIDECAR,
    _SHARED_OUTPUT_SUBPATH,
    _SHARED_VOLUME_NAME,
    _SIDECAR_CONFIGMAP_ENV,
    _SIDECAR_CONTAINER_NAME,
    _SIDECAR_IMAGE_ENV,
    _SIDECAR_SA_NAME_ENV,
    _SIDECAR_SECRET_ENV,
    _TERMINATION_GRACE_PERIOD_SECONDS,
    _WORKLOAD_CONTAINER_NAME,
    _WORKLOAD_INPUT_MOUNT,
    _WORKLOAD_OUTPUT_MOUNT,
    IORefs,
    KubernetesBackend,
    _build_io_refs_configmap_body,
    _build_job_body,
    _collect_pod_logs,
    _cpu_quota_to_k8s,
    _extract_io_refs,
    _load_k8s_clients,
    _make_job_name,
    _mem_limit_to_k8s,
)
from asqi.config import ContainerConfig
from asqi.errors import ManifestExtractionError
from asqi.schemas import InputRef, Manifest, OutputDestination
from pytest import LogCaptureFixture

# ── Protocol compliance ────────────────────────────────────────────────────────


def test_kubernetes_backend_implements_container_backend_protocol() -> None:
    assert isinstance(KubernetesBackend(), ContainerBackend)


# ── Name / resource helpers ────────────────────────────────────────────────────


class TestMakeJobName:
    def test_no_hint_starts_with_prefix(self) -> None:
        name = _make_job_name()
        assert name.startswith("asqi-job-")

    def test_no_hint_max_length(self) -> None:
        name = _make_job_name()
        assert len(name) <= 63

    def test_hint_included_in_name(self) -> None:
        name = _make_job_name("my_test")
        assert "my-test" in name

    def test_hint_sanitised_uppercase(self) -> None:
        name = _make_job_name("MyTest")
        assert name == name.lower()

    def test_hint_sanitised_dots(self) -> None:
        name = _make_job_name("some.image.tag")
        assert "." not in name

    def test_hint_sanitised_slashes(self) -> None:
        name = _make_job_name("registry/image")
        assert "/" not in name

    def test_hint_sanitised_colons(self) -> None:
        name = _make_job_name("image:v1.2.3")
        assert ":" not in name

    def test_hint_sanitised_at_sign(self) -> None:
        name = _make_job_name("image@sha256:abc")
        assert "@" not in name

    def test_uniqueness(self) -> None:
        names = {_make_job_name() for _ in range(20)}
        assert len(names) == 20


class TestMemLimitToK8s:
    def test_gigabytes(self) -> None:
        assert _mem_limit_to_k8s("2g") == "2Gi"

    def test_megabytes(self) -> None:
        assert _mem_limit_to_k8s("512m") == "512Mi"

    def test_already_k8s_format(self) -> None:
        assert _mem_limit_to_k8s("4Gi") == "4Gi"

    def test_integer(self) -> None:
        assert _mem_limit_to_k8s("1073741824") == "1073741824"


class TestCpuQuotaToK8s:
    def test_two_cores(self) -> None:
        assert _cpu_quota_to_k8s(100000, 200000) == "2"

    def test_half_core(self) -> None:
        assert _cpu_quota_to_k8s(100000, 50000) == "0.5"

    def test_one_core(self) -> None:
        assert _cpu_quota_to_k8s(100000, 100000) == "1"

    def test_zero_period_falls_back(self) -> None:
        assert _cpu_quota_to_k8s(0, 200000) == "2"

    def test_zero_quota_falls_back(self) -> None:
        assert _cpu_quota_to_k8s(100000, 0) == "2"

    def test_tiny_quota_enforces_minimum(self) -> None:
        result = _cpu_quota_to_k8s(100000, 1)
        assert float(result) >= 0.001


class TestLoadK8sClients:
    @patch("kubernetes.client.CoreV1Api")
    @patch("kubernetes.client.BatchV1Api")
    @patch("kubernetes.config.load_incluster_config")
    def test_incluster_config_uses_default_refreshing_client(
        self,
        load_incluster_config: MagicMock,
        batch_api_cls: MagicMock,
        core_api_cls: MagicMock,
    ) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        batch_api_cls.return_value = batch_api
        core_api_cls.return_value = core_api

        assert _load_k8s_clients() == (batch_api, core_api)

        load_incluster_config.assert_called_once_with()
        batch_api_cls.assert_called_once_with()
        core_api_cls.assert_called_once_with()

    def test_incluster_config_maps_token_to_bearer_auth_and_keeps_refresh_hook(self, tmp_path: Path) -> None:
        from kubernetes.client import Configuration
        from kubernetes.config.incluster_config import (
            SERVICE_HOST_ENV_NAME,
            SERVICE_PORT_ENV_NAME,
            InClusterConfigLoader,
        )

        token_file = tmp_path / "token"
        token_file.write_text("token-value", encoding="utf-8")
        cert_file = tmp_path / "ca.crt"
        cert_file.write_text("ca", encoding="utf-8")
        env = {
            SERVICE_HOST_ENV_NAME: "10.0.0.1",
            SERVICE_PORT_ENV_NAME: "443",
        }
        config = Configuration()

        loader = InClusterConfigLoader(
            token_filename=str(token_file),
            cert_filename=str(cert_file),
            environ=env,
        )
        cast(Any, loader).load_and_set(config)

        auth = cast(dict[str, dict[str, str]], config.auth_settings())

        assert "BearerToken" in auth
        assert auth["BearerToken"]["key"] == "authorization"
        assert auth["BearerToken"]["value"] == "bearer token-value"
        assert config.refresh_api_key_hook is not None


# ── Args validation ────────────────────────────────────────────────────────────


def _input_ref(key: str = "in/data.json") -> InputRef:
    return InputRef(bucket="my-bucket", key=key)


def _output_dest(key_prefix: str = "out/run-1") -> OutputDestination:
    return OutputDestination(bucket="my-bucket", key_prefix=key_prefix)


class TestIORefs:
    def test_defaults_are_empty(self) -> None:
        refs = IORefs()
        assert refs.args == []
        assert refs.inputs == []
        assert refs.output is None
        assert refs.error is None

    def test_is_frozen(self) -> None:
        import dataclasses

        refs = IORefs()
        with pytest.raises(dataclasses.FrozenInstanceError):
            refs.error = "nope"  # type: ignore[misc]


class TestExtractIORefs:
    def test_no_args(self) -> None:
        refs = _extract_io_refs([])
        assert refs.error is None
        assert refs.args == []
        assert refs.inputs == []
        assert refs.output is None

    def test_no_param_flags(self) -> None:
        refs = _extract_io_refs(["run", "--foo", "bar"])
        assert refs.error is None
        assert refs.args == ["run", "--foo", "bar"]

    def test_malformed_json_passes_through(self) -> None:
        refs = _extract_io_refs(["run", "--test-params", "not-json"])
        assert refs.error is None
        assert refs.args == ["run", "--test-params", "not-json"]

    def test_legacy_volumes_in_test_params_fails_closed(self) -> None:
        import json

        params = json.dumps({"k": "v", "__volumes": {"input": "/in"}})
        refs = _extract_io_refs(["run", "--test-params", params])
        assert refs.error is not None
        assert "__volumes" in refs.error
        assert "RUN_BACKEND=docker" in refs.error

    def test_legacy_volumes_in_generation_params_fails_closed(self) -> None:
        import json

        params = json.dumps({"__volumes": {"input": "/in"}})
        refs = _extract_io_refs(["run", "--generation-params", params])
        assert refs.error is not None
        assert "__volumes" in refs.error

    def test_inputs_extracted_and_stripped(self) -> None:
        import json

        params = json.dumps(
            {
                "model": "gpt-x",
                "__inputs": [
                    {"bucket": "my-bucket", "key": "in/a.json"},
                    {"bucket": "my-bucket", "key": "in/b.json", "checksum": "deadbeef"},
                ],
            }
        )
        refs = _extract_io_refs(["run", "--test-params", params])
        assert refs.error is None
        assert len(refs.inputs) == 2
        assert refs.inputs[0].key == "in/a.json"
        assert refs.inputs[1].checksum == "deadbeef"
        # Workload sees the stripped payload — no __inputs key.
        stripped = json.loads(refs.args[2])
        assert "__inputs" not in stripped
        assert stripped == {"model": "gpt-x"}

    def test_output_extracted_and_stripped(self) -> None:
        import json

        params = json.dumps(
            {
                "model": "gpt-x",
                "__output": {"bucket": "my-bucket", "key_prefix": "out/run-1"},
            }
        )
        refs = _extract_io_refs(["run", "--test-params", params])
        assert refs.error is None
        assert refs.output is not None
        assert refs.output.key_prefix == "out/run-1"
        stripped = json.loads(refs.args[2])
        assert "__output" not in stripped

    def test_inputs_must_be_list(self) -> None:
        import json

        params = json.dumps({"__inputs": {"bucket": "b", "key": "k"}})
        refs = _extract_io_refs(["run", "--test-params", params])
        assert refs.error is not None
        assert "__inputs" in refs.error

    def test_malformed_input_ref_fails_closed(self) -> None:
        import json

        params = json.dumps({"__inputs": [{"bucket": "b"}]})  # missing key
        refs = _extract_io_refs(["run", "--test-params", params])
        assert refs.error is not None
        assert "__inputs[0]" in refs.error

    def test_malformed_output_ref_fails_closed(self) -> None:
        import json

        params = json.dumps({"__output": {"bucket": "b"}})  # missing key_prefix
        refs = _extract_io_refs(["run", "--test-params", params])
        assert refs.error is not None
        assert "__output" in refs.error

    def test_duplicate_output_across_flags_fails_closed(self) -> None:
        import json

        tp = json.dumps({"__output": {"bucket": "my-bucket", "key_prefix": "out/a"}})
        gp = json.dumps({"__output": {"bucket": "my-bucket", "key_prefix": "out/b"}})
        refs = _extract_io_refs(["run", "--test-params", tp, "--generation-params", gp])
        assert refs.error is not None
        assert "more than once" in refs.error

    def test_original_args_not_mutated(self) -> None:
        import json

        params = json.dumps({"k": "v", "__inputs": [{"bucket": "my-bucket", "key": "in/a"}]})
        original = ["run", "--test-params", params]
        original_copy = list(original)
        _extract_io_refs(original)
        assert original == original_copy

    def test_payload_string_preserved_when_no_io_refs(self) -> None:
        """When no `__inputs` / `__output` keys are present, the JSON string
        is passed through byte-for-byte (no whitespace/key-order churn)."""
        # Whitespace + non-alphabetical key order — would be lost on a
        # naive json.dumps round-trip.
        raw = '{ "z": 1,    "a": 2 }'
        refs = _extract_io_refs(["run", "--test-params", raw])
        assert refs.error is None
        assert refs.args[2] == raw  # byte-for-byte identical


class TestBuildIORefsConfigmapBody:
    def test_metadata_fields(self) -> None:
        body = _build_io_refs_configmap_body(
            configmap_name="job-x-io-refs",
            namespace="prod",
            io_refs=IORefs(inputs=[_input_ref()], output=_output_dest()),
            workflow_id="wf-abc",
        )
        assert body["kind"] == "ConfigMap"
        assert body["apiVersion"] == "v1"
        assert body["metadata"]["name"] == "job-x-io-refs"
        assert body["metadata"]["namespace"] == "prod"
        assert body["metadata"]["labels"]["workflow_id"] == "wf-abc"

    def test_data_payload_round_trips(self) -> None:
        import json

        body = _build_io_refs_configmap_body(
            configmap_name="cm-1",
            namespace="default",
            io_refs=IORefs(
                inputs=[_input_ref("in/a.json"), _input_ref("in/b.json")],
                output=_output_dest("out/run-1"),
            ),
            workflow_id="wf-1",
        )
        payload = json.loads(body["data"][_IO_REFS_CONFIGMAP_KEY])
        assert [i["key"] for i in payload["inputs"]] == ["in/a.json", "in/b.json"]
        assert payload["output"]["key_prefix"] == "out/run-1"

    def test_empty_refs_serialise_to_empty_inputs_and_null_output(self) -> None:
        import json

        body = _build_io_refs_configmap_body(
            configmap_name="cm-1",
            namespace="default",
            io_refs=IORefs(),
            workflow_id="wf-1",
        )
        payload = json.loads(body["data"][_IO_REFS_CONFIGMAP_KEY])
        assert payload["inputs"] == []
        assert payload["output"] is None


# ── Job manifest builder ───────────────────────────────────────────────────────


def _build_default_job_body(
    *,
    job_name: str = "job-1",
    image: str = "img:1",
    args: list[str] | None = None,
    environment: dict[str, str] | None = None,
    config: ContainerConfig | None = None,
    workflow_id: str = "wf1",
    namespace: str = "default",
    sidecar_image: str = "sidecar:test",
    io_refs_configmap_name: str = "job-1-io-refs",
) -> dict[str, Any]:
    return _build_job_body(
        job_name=job_name,
        image=image,
        args=list(args) if args is not None else [],
        environment=environment,
        container_config=config or ContainerConfig(),
        workflow_id=workflow_id,
        namespace=namespace,
        sidecar_image=sidecar_image,
        io_refs_configmap_name=io_refs_configmap_name,
    )


def _workload(body: dict[str, Any]) -> dict[str, Any]:
    containers = body["spec"]["template"]["spec"]["containers"]
    return next(c for c in containers if c["name"] == _WORKLOAD_CONTAINER_NAME)


def _sidecar(body: dict[str, Any]) -> dict[str, Any]:
    init_containers = body["spec"]["template"]["spec"].get("initContainers", [])
    return next(c for c in init_containers if c["name"] == _SIDECAR_CONTAINER_NAME)


class TestBuildJobBody:
    def test_api_version(self) -> None:
        assert _build_default_job_body()["apiVersion"] == "batch/v1"

    def test_job_name_in_metadata(self) -> None:
        body = _build_default_job_body(job_name="my-job")
        assert body["metadata"]["name"] == "my-job"

    def test_namespace_in_metadata(self) -> None:
        body = _build_default_job_body(namespace="prod")
        assert body["metadata"]["namespace"] == "prod"

    def test_workflow_id_label(self) -> None:
        body = _build_default_job_body(workflow_id="wf-abc")
        assert body["metadata"]["labels"]["workflow_id"] == "wf-abc"

    def test_restart_policy_never(self) -> None:
        body = _build_default_job_body()
        assert body["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_backoff_limit_zero(self) -> None:
        assert _build_default_job_body()["spec"]["backoffLimit"] == 0

    def test_image_in_workload_container(self) -> None:
        body = _build_default_job_body(image="my-image:latest")
        assert _workload(body)["image"] == "my-image:latest"

    def test_args_in_workload_container(self) -> None:
        body = _build_default_job_body(args=["--foo", "bar"])
        assert _workload(body)["args"] == ["--foo", "bar"]

    def test_environment_injected_into_workload(self) -> None:
        body = _build_default_job_body(environment={"KEY": "val"})
        assert {"name": "KEY", "value": "val"} in _workload(body)["env"]

    def test_memory_limit_converted(self) -> None:
        body = _build_default_job_body()
        assert _workload(body)["resources"]["limits"]["memory"] == "2Gi"

    def test_cpu_limit_converted(self) -> None:
        body = _build_default_job_body()
        assert _workload(body)["resources"]["limits"]["cpu"] == "2"

    # ── Sidecar / shared-volume coverage (AIP-2473) ────────────────────────────

    def test_sidecar_uses_native_pattern(self) -> None:
        """Sidecar is an initContainer with restartPolicy=Always (K8s >= 1.28)."""
        body = _build_default_job_body(sidecar_image="my-sidecar:1.2.3")
        sidecar = _sidecar(body)
        assert sidecar["image"] == "my-sidecar:1.2.3"
        assert sidecar["restartPolicy"] == "Always"

    def test_sidecar_env_carries_job_handle_and_paths(self) -> None:
        body = _build_default_job_body(workflow_id="wf-abc")
        env = {e["name"]: e["value"] for e in _sidecar(body)["env"]}
        assert env["AIP_JOB_HANDLE"] == "wf-abc"
        assert env["AIP_IO_REFS_PATH"] == f"{_IO_REFS_CONFIGMAP_MOUNT}/{_IO_REFS_CONFIGMAP_KEY}"
        assert env["AIP_SHARED_DIR"] == _SHARED_MOUNT_SIDECAR

    def test_sidecar_mounts_shared_volume_and_configmap(self) -> None:
        body = _build_default_job_body()
        mounts = {m["name"]: m for m in _sidecar(body)["volumeMounts"]}
        assert mounts[_SHARED_VOLUME_NAME]["mountPath"] == _SHARED_MOUNT_SIDECAR
        cm_mount = mounts[_IO_REFS_VOLUME_NAME]
        assert cm_mount["mountPath"] == _IO_REFS_CONFIGMAP_MOUNT
        assert cm_mount.get("readOnly") is True

    def test_workload_mounts_shared_volume_via_subpaths(self) -> None:
        body = _build_default_job_body()
        mounts = _workload(body)["volumeMounts"]
        by_subpath = {m["subPath"]: m for m in mounts}
        assert by_subpath[_SHARED_INPUT_SUBPATH]["mountPath"] == _WORKLOAD_INPUT_MOUNT
        assert by_subpath[_SHARED_OUTPUT_SUBPATH]["mountPath"] == _WORKLOAD_OUTPUT_MOUNT
        assert all(m["name"] == _SHARED_VOLUME_NAME for m in mounts)

    def test_workload_does_not_mount_io_refs_configmap(self) -> None:
        body = _build_default_job_body()
        for mount in _workload(body)["volumeMounts"]:
            assert mount["name"] != _IO_REFS_VOLUME_NAME

    def test_pod_volumes_include_emptydir_and_configmap(self) -> None:
        body = _build_default_job_body(io_refs_configmap_name="job-x-io-refs")
        vols = {v["name"]: v for v in body["spec"]["template"]["spec"]["volumes"]}
        assert "emptyDir" in vols[_SHARED_VOLUME_NAME]
        assert vols[_IO_REFS_VOLUME_NAME]["configMap"]["name"] == "job-x-io-refs"

    def test_termination_grace_period_set(self) -> None:
        body = _build_default_job_body()
        assert body["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == _TERMINATION_GRACE_PERIOD_SECONDS

    def test_only_one_workload_container(self) -> None:
        body = _build_default_job_body()
        containers = body["spec"]["template"]["spec"]["containers"]
        assert len(containers) == 1
        assert containers[0]["name"] == _WORKLOAD_CONTAINER_NAME

    def test_sidecar_has_resource_requests_and_limits(self) -> None:
        """Sidecar must declare resources so the kubelet does not evict it
        first under memory pressure — that would break the upload window."""
        body = _build_default_job_body()
        resources = _sidecar(body)["resources"]
        assert "requests" in resources and "limits" in resources
        for kind in ("requests", "limits"):
            assert resources[kind].get("cpu")
            assert resources[kind].get("memory")

    # ── Helm sidecar wiring: ServiceAccount + envFrom (AIP-2475) ───────────────

    def test_no_service_account_by_default(self) -> None:
        """Unset sidecar_sa_name → field omitted so the Pod uses the namespace
        default SA (local / MinIO, no IRSA)."""
        spec = _build_default_job_body()["spec"]["template"]["spec"]
        assert "serviceAccountName" not in spec

    def test_service_account_set_when_provided(self) -> None:
        body = _build_job_body(
            job_name="job-1",
            image="img:1",
            args=[],
            environment=None,
            container_config=ContainerConfig(),
            workflow_id="wf1",
            namespace="default",
            sidecar_image="sidecar:test",
            io_refs_configmap_name="job-1-io-refs",
            sidecar_sa_name="asqi-runner-sidecar",
        )
        assert body["spec"]["template"]["spec"]["serviceAccountName"] == "asqi-runner-sidecar"

    def test_no_envfrom_on_sidecar_by_default(self) -> None:
        """Unset ConfigMap/Secret refs → sidecar has no envFrom (only the
        per-Job artifact env vars are present)."""
        assert "envFrom" not in _sidecar(_build_default_job_body())

    def test_sidecar_envfrom_references_configmap_and_secret(self) -> None:
        body = _build_job_body(
            job_name="job-1",
            image="img:1",
            args=[],
            environment=None,
            container_config=ContainerConfig(),
            workflow_id="wf1",
            namespace="default",
            sidecar_image="sidecar:test",
            io_refs_configmap_name="job-1-io-refs",
            sidecar_configmap_name="asqi-runner-sidecar",
            sidecar_secret_name="asqi-runner-sidecar",
        )
        env_from = _sidecar(body)["envFrom"]
        assert {"configMapRef": {"name": "asqi-runner-sidecar"}} in env_from
        assert {"secretRef": {"name": "asqi-runner-sidecar"}} in env_from

    def test_sidecar_envfrom_only_configmap_when_secret_unset(self) -> None:
        body = _build_job_body(
            job_name="job-1",
            image="img:1",
            args=[],
            environment=None,
            container_config=ContainerConfig(),
            workflow_id="wf1",
            namespace="default",
            sidecar_image="sidecar:test",
            io_refs_configmap_name="job-1-io-refs",
            sidecar_configmap_name="asqi-runner-sidecar",
        )
        env_from = _sidecar(body)["envFrom"]
        assert env_from == [{"configMapRef": {"name": "asqi-runner-sidecar"}}]

    def test_artifact_env_preserved_alongside_envfrom(self) -> None:
        """The AIP-2473 per-Job artifact contract (env) must survive even when
        Helm wires envFrom. env is evaluated after envFrom, so it wins on
        collision — but more importantly the three keys must still be present."""
        body = _build_job_body(
            job_name="job-1",
            image="img:1",
            args=[],
            environment=None,
            container_config=ContainerConfig(),
            workflow_id="wf-abc",
            namespace="default",
            sidecar_image="sidecar:test",
            io_refs_configmap_name="job-1-io-refs",
            sidecar_configmap_name="cm",
            sidecar_secret_name="sec",
        )
        sidecar = _sidecar(body)
        env = {e["name"]: e["value"] for e in sidecar["env"]}
        assert env["AIP_JOB_HANDLE"] == "wf-abc"
        assert env["AIP_IO_REFS_PATH"] == f"{_IO_REFS_CONFIGMAP_MOUNT}/{_IO_REFS_CONFIGMAP_KEY}"
        assert env["AIP_SHARED_DIR"] == _SHARED_MOUNT_SIDECAR
        # envFrom present too — both wiring mechanisms coexist.
        assert sidecar["envFrom"]


# ── KubernetesBackend.check_images / pull_images ───────────────────────────────


class TestCheckImages:
    def test_all_true(self) -> None:
        backend = KubernetesBackend()
        result = backend.check_images(["img:1", "img:2"])
        assert result == {"img:1": True, "img:2": True}

    def test_empty(self) -> None:
        assert KubernetesBackend().check_images([]) == {}


class TestPullImages:
    def test_no_op(self, caplog: LogCaptureFixture) -> None:
        """pull_images should not raise and should log a message."""
        import logging

        backend = KubernetesBackend()
        with caplog.at_level(logging.INFO, logger="asqi.backends.kubernetes_backend"):
            backend.pull_images(["img:1"])
        assert "no-op" in caplog.text.lower() or "kubelet" in caplog.text.lower()


# ── KubernetesBackend.run ──────────────────────────────────────────────────────


def _make_mock_clients(
    succeeded: int = 1,
    failed: int = 0,
    conditions: list[Any] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_batch_api, mock_core_api) pre-configured for a job result."""
    batch_api = MagicMock()
    core_api = MagicMock()

    job_status = MagicMock()
    job_status.succeeded = succeeded
    job_status.failed = failed
    job_status.conditions = conditions or []

    job = MagicMock()
    job.status = job_status
    batch_api.read_namespaced_job.return_value = job

    core_api.list_namespaced_pod.return_value = MagicMock(items=[])
    core_api.read_namespaced_pod_log.return_value = "test output"

    return batch_api, core_api


class TestKubernetesBackendRun:
    def test_collect_pod_logs_reads_raw_response_without_deserializing_json(self) -> None:
        core_api = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        core_api.list_namespaced_pod.return_value = MagicMock(items=[pod])
        response = MagicMock()
        response.data = b'{"test_results": {"success": true}}\n'
        core_api.read_namespaced_pod_log.return_value = response

        assert _collect_pod_logs(core_api, "job-1", "test-ns") == '{"test_results": {"success": true}}\n'
        core_api.read_namespaced_pod_log.assert_called_once_with(
            name="pod-1",
            namespace="test-ns",
            _preload_content=False,
        )

    def test_collect_pod_logs_decodes_bytes(self) -> None:
        core_api = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        core_api.list_namespaced_pod.return_value = MagicMock(items=[pod])
        core_api.read_namespaced_pod_log.return_value = b'{"success": true}\n'

        assert _collect_pod_logs(core_api, "job-1", "test-ns") == '{"success": true}\n'

    def test_collect_pod_logs_decodes_stringified_bytes(self) -> None:
        core_api = MagicMock()
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        core_api.list_namespaced_pod.return_value = MagicMock(items=[pod])
        core_api.read_namespaced_pod_log.return_value = "b'{\"success\": true}\\n'"

        assert _collect_pod_logs(core_api, "job-1", "test-ns") == '{"success": true}\n'

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_successful_run_returns_success(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        backend = KubernetesBackend(namespace="test-ns")
        result = backend.run(image="img:1", args=["--foo"], container_config=ContainerConfig())

        assert result["success"] is True
        assert result["exit_code"] == 0

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_successful_run_creates_job(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        backend = KubernetesBackend(namespace="test-ns")
        backend.run(image="img:1", args=[], container_config=ContainerConfig())

        batch_api.create_namespaced_job.assert_called_once()
        call_kwargs = batch_api.create_namespaced_job.call_args
        assert call_kwargs.kwargs["namespace"] == "test-ns"

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_successful_run_deletes_job_after(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        backend = KubernetesBackend()
        backend.run(image="img:1", args=[], container_config=ContainerConfig())

        batch_api.delete_namespaced_job.assert_called_once()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_failed_job_returns_failure(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=0, failed=1)
        pod = MagicMock()
        pod.metadata.name = "pod-1"
        pod.status.container_statuses = [MagicMock(state=MagicMock(terminated=MagicMock(exit_code=1)))]
        core_api.list_namespaced_pod.return_value = MagicMock(items=[pod])
        mock_load.return_value = (batch_api, core_api)

        backend = KubernetesBackend()
        result = backend.run(image="img:1", args=[], container_config=ContainerConfig())

        assert result["success"] is False
        assert result["exit_code"] == 1

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_create_job_api_error_returns_error_dict(self, mock_load: MagicMock) -> None:
        from kubernetes.client.rest import ApiException  # requires kubernetes test dep

        batch_api = MagicMock()
        batch_api.create_namespaced_job.side_effect = ApiException(status=403, reason="Forbidden")
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        backend = KubernetesBackend()
        result = backend.run(image="img:1", args=[], container_config=ContainerConfig())

        assert result["success"] is False
        assert result["container_id"] == ""
        assert "403" in result["error"] or "Forbidden" in result["error"]

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_timeout_returns_exit_137(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job_status = MagicMock()
        job_status.succeeded = 0
        job_status.failed = 0
        job_status.conditions = []
        job = MagicMock()
        job.status = job_status
        batch_api.read_namespaced_job.return_value = job

        config = ContainerConfig(timeout_seconds=1)

        backend = KubernetesBackend()
        result = backend.run(image="img:1", args=[], container_config=config)

        assert result["exit_code"] == 137
        assert "timed out" in result["error"].lower()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_volumes_in_args_returns_failure_without_creating_job(self, mock_load: MagicMock) -> None:
        import json

        params = json.dumps({"key": "val", "__volumes": {"input": "/in"}})
        backend = KubernetesBackend()
        result = backend.run(
            image="img:1",
            args=["--test-params", params],
            container_config=ContainerConfig(),
        )

        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "__volumes" in result["error"]
        assert "RUN_BACKEND=docker" in result["error"]
        mock_load.assert_not_called()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_host_access_true_returns_failure_without_creating_job(self, mock_load: MagicMock) -> None:
        manifest = Manifest(
            name="host-test",
            version="1.0.0",
            input_systems=[],
            input_schema=[],
            input_datasets=[],
            output_metrics=[],
            environment_variables=[],
            host_access=True,
        )
        backend = KubernetesBackend()
        result = backend.run(
            image="img:1",
            args=[],
            container_config=ContainerConfig(),
            manifest=manifest,
        )

        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "host_access" in result["error"]
        assert "RUN_BACKEND=docker" in result["error"]
        mock_load.assert_not_called()

    # ── ConfigMap lifecycle (AIP-2473) ─────────────────────────────────────────

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_configmap_created_before_job_and_named_after_it(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        KubernetesBackend(namespace="test-ns").run(image="img:1", args=[], container_config=ContainerConfig())

        core_api.create_namespaced_config_map.assert_called_once()
        cm_call = core_api.create_namespaced_config_map.call_args
        cm_body = cm_call.kwargs["body"]
        assert cm_call.kwargs["namespace"] == "test-ns"
        # Job manifest must reference the same ConfigMap name.
        job_body = batch_api.create_namespaced_job.call_args.kwargs["body"]
        vols = {v["name"]: v for v in job_body["spec"]["template"]["spec"]["volumes"]}
        assert vols[_IO_REFS_VOLUME_NAME]["configMap"]["name"] == cm_body["metadata"]["name"]
        assert cm_body["metadata"]["name"].endswith("-io-refs")

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_configmap_deleted_on_normal_completion(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        KubernetesBackend().run(image="img:1", args=[], container_config=ContainerConfig())

        core_api.delete_namespaced_config_map.assert_called_once()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_configmap_deleted_when_job_create_fails(self, mock_load: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        batch_api = MagicMock()
        core_api = MagicMock()
        batch_api.create_namespaced_job.side_effect = ApiException(status=403, reason="Forbidden")
        mock_load.return_value = (batch_api, core_api)

        result = KubernetesBackend().run(image="img:1", args=[], container_config=ContainerConfig())

        assert result["success"] is False
        # ConfigMap was created up-front, so it must be deleted on the failure path.
        core_api.delete_namespaced_config_map.assert_called_once()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_configmap_create_failure_short_circuits_before_job(self, mock_load: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        batch_api = MagicMock()
        core_api = MagicMock()
        core_api.create_namespaced_config_map.side_effect = ApiException(status=403, reason="Forbidden")
        mock_load.return_value = (batch_api, core_api)

        result = KubernetesBackend().run(image="img:1", args=[], container_config=ContainerConfig())

        assert result["success"] is False
        assert "ConfigMap" in result["error"]
        batch_api.create_namespaced_job.assert_not_called()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_sidecar_image_propagated_into_job_body(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        KubernetesBackend(sidecar_image="custom-sidecar:9.9").run(
            image="img:1", args=[], container_config=ContainerConfig()
        )

        job_body = batch_api.create_namespaced_job.call_args.kwargs["body"]
        sidecar = next(
            c for c in job_body["spec"]["template"]["spec"]["initContainers"] if c["name"] == _SIDECAR_CONTAINER_NAME
        )
        assert sidecar["image"] == "custom-sidecar:9.9"


class TestKubernetesBackendSidecarImageResolution:
    def test_explicit_constructor_arg_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_SIDECAR_IMAGE_ENV, "from-env:1")
        backend = KubernetesBackend(sidecar_image="explicit:1")
        assert backend._sidecar_image == "explicit:1"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_SIDECAR_IMAGE_ENV, "from-env:2")
        backend = KubernetesBackend()
        assert backend._sidecar_image == "from-env:2"

    def test_placeholder_default_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_SIDECAR_IMAGE_ENV, raising=False)
        backend = KubernetesBackend()
        assert backend._sidecar_image == _DEFAULT_SIDECAR_IMAGE


class TestKubernetesBackendSidecarWiringResolution:
    """SA / ConfigMap / Secret refs resolve constructor-arg > env var > None
    (AIP-2475), mirroring the sidecar-image precedence."""

    def test_explicit_constructor_args_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_SIDECAR_SA_NAME_ENV, "env-sa")
        monkeypatch.setenv(_SIDECAR_CONFIGMAP_ENV, "env-cm")
        monkeypatch.setenv(_SIDECAR_SECRET_ENV, "env-sec")
        backend = KubernetesBackend(
            sidecar_sa_name="arg-sa",
            sidecar_configmap_name="arg-cm",
            sidecar_secret_name="arg-sec",
        )
        assert backend._sidecar_sa_name == "arg-sa"
        assert backend._sidecar_configmap_name == "arg-cm"
        assert backend._sidecar_secret_name == "arg-sec"

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_SIDECAR_SA_NAME_ENV, "env-sa")
        monkeypatch.setenv(_SIDECAR_CONFIGMAP_ENV, "env-cm")
        monkeypatch.setenv(_SIDECAR_SECRET_ENV, "env-sec")
        backend = KubernetesBackend()
        assert backend._sidecar_sa_name == "env-sa"
        assert backend._sidecar_configmap_name == "env-cm"
        assert backend._sidecar_secret_name == "env-sec"

    def test_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (_SIDECAR_SA_NAME_ENV, _SIDECAR_CONFIGMAP_ENV, _SIDECAR_SECRET_ENV):
            monkeypatch.delenv(var, raising=False)
        backend = KubernetesBackend()
        assert backend._sidecar_sa_name is None
        assert backend._sidecar_configmap_name is None
        assert backend._sidecar_secret_name is None

    def test_empty_env_var_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty Helm value (e.g. serviceAccount left as default) must not
        emit an empty serviceAccountName / envFrom ref."""
        monkeypatch.setenv(_SIDECAR_SA_NAME_ENV, "")
        monkeypatch.setenv(_SIDECAR_CONFIGMAP_ENV, "")
        monkeypatch.setenv(_SIDECAR_SECRET_ENV, "")
        backend = KubernetesBackend()
        assert backend._sidecar_sa_name is None
        assert backend._sidecar_configmap_name is None
        assert backend._sidecar_secret_name is None

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_wiring_propagated_into_job_body(self, mock_load: MagicMock) -> None:
        batch_api, core_api = _make_mock_clients(succeeded=1)
        mock_load.return_value = (batch_api, core_api)

        KubernetesBackend(
            sidecar_sa_name="asqi-runner",
            sidecar_configmap_name="asqi-runner-sidecar",
            sidecar_secret_name="asqi-runner-sidecar",
        ).run(image="img:1", args=[], container_config=ContainerConfig())

        job_body = batch_api.create_namespaced_job.call_args.kwargs["body"]
        pod_spec = job_body["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "asqi-runner"
        sidecar = next(c for c in pod_spec["initContainers"] if c["name"] == _SIDECAR_CONTAINER_NAME)
        assert {"configMapRef": {"name": "asqi-runner-sidecar"}} in sidecar["envFrom"]
        assert {"secretRef": {"name": "asqi-runner-sidecar"}} in sidecar["envFrom"]


# ── KubernetesBackend.shutdown ─────────────────────────────────────────────────


class TestKubernetesBackendShutdown:
    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_shutdown_all_deletes_all_service_jobs(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job1 = MagicMock()
        job1.metadata.name = "asqi-job-aaa"
        job2 = MagicMock()
        job2.metadata.name = "asqi-job-bbb"
        batch_api.list_namespaced_job.return_value = MagicMock(items=[job1, job2])

        KubernetesBackend().shutdown()

        assert batch_api.delete_namespaced_job.call_count == 2

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_shutdown_by_workflow_id(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job1 = MagicMock()
        job1.metadata.name = "asqi-job-wf1"
        batch_api.list_namespaced_job.return_value = MagicMock(items=[job1])

        KubernetesBackend().shutdown(workflow_ids=["wf-123"])

        batch_api.list_namespaced_job.assert_called_once()
        call_kwargs = batch_api.list_namespaced_job.call_args.kwargs
        assert "wf-123" in call_kwargs["label_selector"]
        batch_api.delete_namespaced_job.assert_called_once()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_shutdown_empty_workflow_ids(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        KubernetesBackend().shutdown(workflow_ids=[])

        batch_api.delete_namespaced_job.assert_not_called()


# ── KubernetesBackend.extract_manifest ────────────────────────────────────────


class TestKubernetesBackendExtractManifest:
    def _valid_manifest_yaml(self) -> str:
        return yaml.dump(
            {
                "name": "test-package",
                "version": "1.0.0",
                "type": "test",
                "entry_point": "run_tests",
            }
        )

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_success_returns_manifest(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job_status = MagicMock()
        job_status.succeeded = 1
        job_status.failed = 0
        job_status.conditions = []
        job = MagicMock()
        job.status = job_status
        batch_api.read_namespaced_job.return_value = job

        pod = MagicMock()
        pod.metadata.name = "pod-1"
        core_api.list_namespaced_pod.return_value = MagicMock(items=[pod])
        core_api.read_namespaced_pod_log.return_value = self._valid_manifest_yaml()

        backend = KubernetesBackend()
        result = backend.extract_manifest("img:1")

        assert isinstance(result, Manifest)
        assert result.name == "test-package"

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_job_failure_raises_manifest_error(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job_status = MagicMock()
        job_status.succeeded = 0
        job_status.failed = 1
        job_status.conditions = []
        job = MagicMock()
        job.status = job_status
        batch_api.read_namespaced_job.return_value = job
        core_api.list_namespaced_pod.return_value = MagicMock(items=[])

        backend = KubernetesBackend()
        with pytest.raises(ManifestExtractionError) as exc_info:
            backend.extract_manifest("img:1")

        assert exc_info.value.error_type in ("K8S_JOB_FAILED",)

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_job_always_cleaned_up_on_success(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job_status = MagicMock()
        job_status.succeeded = 1
        job_status.failed = 0
        job_status.conditions = []
        job = MagicMock()
        job.status = job_status
        batch_api.read_namespaced_job.return_value = job

        pod = MagicMock()
        pod.metadata.name = "pod-1"
        core_api.list_namespaced_pod.return_value = MagicMock(items=[pod])
        core_api.read_namespaced_pod_log.return_value = self._valid_manifest_yaml()

        backend = KubernetesBackend()
        backend.extract_manifest("img:1")

        batch_api.delete_namespaced_job.assert_called_once()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_job_always_cleaned_up_on_failure(self, mock_load: MagicMock) -> None:
        batch_api = MagicMock()
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        job_status = MagicMock()
        job_status.succeeded = 0
        job_status.failed = 1
        job_status.conditions = []
        job = MagicMock()
        job.status = job_status
        batch_api.read_namespaced_job.return_value = job
        core_api.list_namespaced_pod.return_value = MagicMock(items=[])

        backend = KubernetesBackend()
        with pytest.raises(ManifestExtractionError):
            backend.extract_manifest("img:1")

        batch_api.delete_namespaced_job.assert_called_once()

    @patch("asqi.backends.kubernetes_backend._load_k8s_clients")
    def test_create_job_error_raises_manifest_error(self, mock_load: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        batch_api = MagicMock()
        batch_api.create_namespaced_job.side_effect = ApiException(status=403, reason="Forbidden")
        core_api = MagicMock()
        mock_load.return_value = (batch_api, core_api)

        backend = KubernetesBackend()
        with pytest.raises(ManifestExtractionError) as exc_info:
            backend.extract_manifest("img:1")

        assert exc_info.value.error_type == "K8S_JOB_CREATE_ERROR"
