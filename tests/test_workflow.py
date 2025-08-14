from unittest.mock import patch

import pytest

from asqi.schemas import Manifest, SUTSupport
from asqi.workflow import (
    TestExecutionResult,
    execute_single_test,
    save_results_to_file_step,
)
from asqi.workflow import (
    run_test_suite_workflow as _workflow,
)


def _call_inner_workflow(suite_config, suts_config):
    """Call the inner (undecorated) workflow function if available."""
    workflow_fn = getattr(_workflow, "__wrapped__", _workflow)
    return workflow_fn(suite_config, suts_config)


class DummyHandle:
    def __init__(self, result):
        self._result = result

    def get_result(self):
        return self._result


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Ensure DB-related env vars don't interfere with tests."""
    monkeypatch.delenv("DBOS_DATABASE_URL", raising=False)
    monkeypatch.setenv("TESTING_DATABASE_URL", "sqlite://:memory:")
    yield


def test_run_test_suite_workflow_success():
    # Arrange minimal suite and SUTs configs
    suite_config = {
        "suite_name": "demo",
        "test_suite": [
            {
                "name": "t1",
                "image": "test/image:latest",
                "target_suts": ["sutA"],
                "params": {"p": "v"},
            }
        ],
    }

    suts_config = {
        "systems_under_test": {
            "sutA": {"type": "llm_api", "config": {"endpoint": "http://x"}}
        }
    }

    # Build a minimal manifest that supports the SUT type
    manifest = Manifest(
        name="mock",
        version="1",
        description="",
        image_name="test/image:latest",
        supported_suts=[SUTSupport(type="llm_api", required_config=None)],
        input_schema=[],
        output_metrics=[],
        output_artifacts=None,
    )

    success_result = TestExecutionResult("t1_sutA", "sutA", "test/image:latest")
    success_result.start_time = 1.0
    success_result.end_time = 2.0
    success_result.exit_code = 0
    success_result.success = True
    success_result.test_results = {"success": True}

    with (
        patch("asqi.workflow.check_image_availability") as mock_avail,
        patch("asqi.workflow.extract_manifest_from_image_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
        patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        patch("asqi.workflow.test_queue") as mock_queue,
    ):
        mock_avail.return_value = {"test/image:latest": True}
        mock_extract.return_value = manifest
        mock_validate.return_value = []
        mock_plan.return_value = [
            {
                "test_name": "t1_sutA",
                "image": "test/image:latest",
                "sut_name": "sutA",
                "sut_config": {"type": "llm_api", "endpoint": "http://x"},
                "test_params": {"p": "v"},
            }
        ]

        # Enqueue returns a handle with get_result -> success_result
        mock_queue.enqueue.side_effect = lambda *args, **kwargs: DummyHandle(
            success_result
        )

        out = _call_inner_workflow(suite_config, suts_config)

    assert out["summary"]["status"] == "COMPLETED"
    assert out["summary"]["total_tests"] == 1
    assert out["summary"]["successful_tests"] == 1
    assert out["summary"]["failed_tests"] == 0
    assert len(out["results"]) == 1
    assert out["results"][0]["metadata"]["success"] is True


def test_run_test_suite_workflow_validation_failure():
    suite_config = {
        "suite_name": "demo",
        "test_suite": [
            {
                "name": "bad_test",
                "image": "missing/image:latest",
                "target_suts": ["sutA"],
                "params": {},
            }
        ],
    }

    suts_config = {"systems_under_test": {"sutA": {"type": "llm_api", "config": {}}}}

    with (
        patch("asqi.workflow.check_image_availability") as mock_avail,
        patch("asqi.workflow.extract_manifest_from_image_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
    ):
        mock_avail.return_value = {"missing/image:latest": True}
        mock_extract.return_value = None  # no manifest extracted
        mock_validate.return_value = [
            "Test 'bad_test': No manifest available for image 'missing/image:latest'"
        ]

        out = _call_inner_workflow(suite_config, suts_config)

    assert out["summary"]["status"] == "VALIDATION_FAILED"
    assert out["summary"]["total_tests"] == 0
    assert out["summary"]["successful_tests"] == 0
    assert out["summary"]["failed_tests"] == 0
    assert out["results"] == []


def test_execute_single_test_success():
    fake_container_output = '{"success": true, "metric": 1}'
    with patch("asqi.workflow.run_container_with_args") as run_mock:
        run_mock.return_value = {
            "success": True,
            "exit_code": 0,
            "output": fake_container_output,
            "error": "",
            "container_id": "abc123",
        }

        inner_step = getattr(execute_single_test, "__wrapped__", execute_single_test)
        result = inner_step(
            test_name="t1_sutA",
            image="test/image:latest",
            sut_name="sutA",
            sut_config={"type": "llm_api"},
            test_params={"p": "v"},
        )

    assert result.success is True
    assert result.exit_code == 0
    assert result.container_id == "abc123"
    assert result.test_results.get("success") is True


def test_save_results_to_file_step_calls_impl(tmp_path):
    data = {"summary": {"status": "COMPLETED"}}
    out = tmp_path / "res.json"
    with patch("asqi.workflow.save_results_to_file") as save_mock:
        inner_step = getattr(
            save_results_to_file_step, "__wrapped__", save_results_to_file_step
        )
        inner_step(data, str(out))
        save_mock.assert_called_once_with(data, str(out))


def test_execute_single_test_container_failure():
    """Test handling of container execution failures."""
    with patch("asqi.workflow.run_container_with_args") as run_mock:
        run_mock.return_value = {
            "success": False,
            "exit_code": 1,
            "output": "",
            "error": "Container failed to start",
            "container_id": "abc123",
        }

        inner_step = getattr(execute_single_test, "__wrapped__", execute_single_test)
        result = inner_step(
            test_name="failing_test",
            image="test/image:latest",
            sut_name="sutA",
            sut_config={"type": "llm_api"},
            test_params={},
        )

    assert result.success is False
    assert result.exit_code == 1
    assert "Container failed to start" in result.error_message


def test_execute_single_test_invalid_json():
    """Test handling of invalid JSON output from container."""
    with patch("asqi.workflow.run_container_with_args") as run_mock:
        run_mock.return_value = {
            "success": True,
            "exit_code": 0,
            "output": "invalid json output",
            "error": "",
            "container_id": "abc123",
        }

        inner_step = getattr(execute_single_test, "__wrapped__", execute_single_test)
        result = inner_step(
            test_name="json_test",
            image="test/image:latest",
            sut_name="sutA",
            sut_config={"type": "llm_api"},
            test_params={},
        )

    assert result.success is False
    assert "Failed to parse JSON output" in result.error_message
