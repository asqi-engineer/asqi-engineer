"""Unit tests for KubernetesBackend.

All Kubernetes API calls are mocked -- no cluster is required.
The ``kubernetes`` package must be installed (added to the ``k8s`` optional dependency group).
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("kubernetes", reason="kubernetes package not installed — skipping K8s backend tests")

import yaml
from asqi.backends.base import ContainerBackend
from asqi.backends.kubernetes_backend import (
    KubernetesBackend,
    _build_job_body,
    _check_no_volumes,
    _cpu_quota_to_k8s,
    _make_job_name,
    _mem_limit_to_k8s,
)
from asqi.config import ContainerConfig
from asqi.errors import ManifestExtractionError
from asqi.schemas import Manifest
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


# ── Args validation ────────────────────────────────────────────────────────────


class TestCheckNoVolumes:
    def test_no_args(self) -> None:
        assert _check_no_volumes([]) is None

    def test_no_param_flags(self) -> None:
        assert _check_no_volumes(["run", "--foo", "bar"]) is None

    def test_detects_volumes_in_test_params(self) -> None:
        import json

        params = json.dumps({"key": "val", "__volumes": {"input": "/in", "output": "/out"}})
        result = _check_no_volumes(["run", "--test-params", params])
        assert result is not None
        assert "__volumes" in result

    def test_detects_volumes_in_generation_params(self) -> None:
        import json

        params = json.dumps({"__volumes": {"input": "/in"}})
        result = _check_no_volumes(["run", "--generation-params", params])
        assert result is not None
        assert "__volumes" in result

    def test_no_volumes_key_returns_none(self) -> None:
        import json

        params = json.dumps({"key": "val"})
        assert _check_no_volumes(["run", "--test-params", params]) is None

    def test_malformed_json_returns_none(self) -> None:
        assert _check_no_volumes(["run", "--test-params", "not-json"]) is None

    def test_detects_volumes_in_first_matching_flag(self) -> None:
        import json

        tp = json.dumps({"key": "val", "__volumes": {"a": "/a"}})
        gp = json.dumps({"gen": "ok", "__volumes": {"b": "/b"}})
        result = _check_no_volumes(["run", "--test-params", tp, "--generation-params", gp])
        assert result is not None


# ── Job manifest builder ───────────────────────────────────────────────────────


class TestBuildJobBody:
    def _default_config(self) -> ContainerConfig:
        return ContainerConfig()

    def test_api_version(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf1", "default")
        assert body["apiVersion"] == "batch/v1"

    def test_job_name_in_metadata(self) -> None:
        body = _build_job_body("my-job", "img:1", [], None, self._default_config(), "wf1", "default")
        assert body["metadata"]["name"] == "my-job"

    def test_namespace_in_metadata(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf1", "prod")
        assert body["metadata"]["namespace"] == "prod"

    def test_workflow_id_label(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf-abc", "default")
        assert body["metadata"]["labels"]["workflow_id"] == "wf-abc"

    def test_restart_policy_never(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf1", "default")
        assert body["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_backoff_limit_zero(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf1", "default")
        assert body["spec"]["backoffLimit"] == 0

    def test_image_in_container(self) -> None:
        body = _build_job_body("job-1", "my-image:latest", [], None, self._default_config(), "wf1", "default")
        container = body["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "my-image:latest"

    def test_args_in_container(self) -> None:
        body = _build_job_body("job-1", "img:1", ["--foo", "bar"], None, self._default_config(), "wf1", "default")
        container = body["spec"]["template"]["spec"]["containers"][0]
        assert container["args"] == ["--foo", "bar"]

    def test_environment_injected(self) -> None:
        body = _build_job_body("job-1", "img:1", [], {"KEY": "val"}, self._default_config(), "wf1", "default")
        container = body["spec"]["template"]["spec"]["containers"][0]
        assert {"name": "KEY", "value": "val"} in container["env"]

    def test_memory_limit_converted(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf1", "default")
        resources = body["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert resources["limits"]["memory"] == "2Gi"

    def test_cpu_limit_converted(self) -> None:
        body = _build_job_body("job-1", "img:1", [], None, self._default_config(), "wf1", "default")
        resources = body["spec"]["template"]["spec"]["containers"][0]["resources"]
        assert resources["limits"]["cpu"] == "2"


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
        result = backend.run(image="img:1", args=["--test-params", params], container_config=ContainerConfig())

        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "__volumes" in result["error"]
        assert "RUN_BACKEND=docker" in result["error"]
        mock_load.assert_not_called()


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
