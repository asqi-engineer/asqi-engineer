import json
from unittest.mock import Mock, patch

import pytest

from asqi.config import ContainerConfig, ExecutionMode, ExecutorConfig
from asqi.schemas import Manifest, OutputReports, ScoreCard, SystemInput
from asqi.workflow import (
    TestExecutionResult,
    add_score_cards_to_results,
    convert_test_results_to_objects,
    evaluate_score_cards_workflow,
    execute_single_test,
    run_end_to_end_workflow,
    save_container_results_to_file_step,
    save_results_to_file_step,
    start_score_card_evaluation,
    start_test_execution,
    validate_test_container_reports,
)
from asqi.workflow import (
    run_test_suite_workflow as _workflow,
)
from test_data import MOCK_AUDIT_RESPONSES, MOCK_SCORE_CARD_CONFIG


def _call_inner_workflow(
    suite_config, systems_config, executor_config, container_config
):
    """Call the inner (undecorated) workflow function if available."""
    workflow_fn = getattr(_workflow, "__wrapped__", _workflow)
    return workflow_fn(suite_config, systems_config, executor_config, container_config)


class DummyHandle:
    def __init__(self, result, workflow_id="test-workflow-123", return_tuple=False):
        self._result = result
        self._workflow_id = workflow_id
        self._return_tuple = return_tuple

    def get_result(self):
        if self._return_tuple:
            return self._result, []
        return self._result

    def get_workflow_id(self):
        return self._workflow_id


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Ensure DB-related env vars don't interfere with tests."""
    monkeypatch.delenv("DBOS_DATABASE_URL", raising=False)
    monkeypatch.setenv("TESTING_DATABASE_URL", "sqlite://:memory:")
    yield


def test_run_test_suite_workflow_success():
    # Arrange minimal suite and systems configs
    suite_config = {
        "suite_name": "demo",
        "test_suite": [
            {
                "name": "t1",
                "id": "t1",
                "image": "test/image:latest",
                "systems_under_test": ["systemA"],
                "params": {"p": "v"},
            }
        ],
    }

    systems_config = {
        "systems": {
            "systemA": {
                "type": "llm_api",
                "params": {"base_url": "http://x", "model": "x-model"},
            }
        }
    }

    container_config: ContainerConfig = ContainerConfig()

    # Build a minimal manifest that supports the system type
    manifest = Manifest(
        name="mock",
        version="1",
        description="",
        input_systems=[
            SystemInput(name="system_under_test", type="llm_api", required=True)
        ],
        input_schema=[],
        output_metrics=[],
        output_artifacts=None,
    )

    success_result = TestExecutionResult(
        "t1_systemA", "t1_systemA", "systemA", "test/image:latest"
    )
    success_result.start_time = 1.0
    success_result.end_time = 2.0
    success_result.exit_code = 0
    success_result.success = True
    success_result.test_results = {"success": True}

    with (
        patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
        patch("asqi.workflow.extract_manifests_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
        patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        patch("asqi.workflow.Queue") as mock_queue_class,
    ):
        mock_avail.return_value = {"test/image:latest": True}
        mock_extract.return_value = {"test/image:latest": manifest}
        mock_validate.return_value = []
        mock_plan.return_value = [
            {
                "test_id": "t1 systemA",
                "test_name": "t1_systemA",
                "image": "test/image:latest",
                "sut_name": "sutA",
                "systems_params": {
                    "system_under_test": {"type": "llm_api", "endpoint": "http://x"}
                },
                "test_params": {"p": "v"},
            }
        ]

        # Enqueue returns a handle with get_result -> success_result
        mock_queue = mock_queue_class.return_value
        mock_queue.enqueue.side_effect = lambda *args, **kwargs: DummyHandle(
            success_result
        )
        results, container_results = _call_inner_workflow(
            suite_config,
            systems_config,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
        )

    assert results["summary"]["status"] == "COMPLETED"
    assert results["summary"]["total_tests"] == 1
    assert results["summary"]["successful_tests"] == 1
    assert results["summary"]["failed_tests"] == 0
    assert len(results["results"]) == 1
    assert results["results"][0]["metadata"]["success"] is True

    assert len(container_results) == 1


def test_run_test_suite_workflow_validation_failure():
    suite_config = {
        "suite_name": "demo",
        "test_suite": [
            {
                "name": "bad_test",
                "id": "bad_test",
                "image": "missing/image:latest",
                "systems_under_test": ["systemA"],
                "params": {},
            }
        ],
    }

    systems_config = {
        "systems": {
            "systemA": {
                "type": "llm_api",
                "params": {"base_url": "http://x", "model": "x-model"},
            }
        }
    }

    container_config: ContainerConfig = ContainerConfig()

    with (
        patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
        patch("asqi.workflow.extract_manifests_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
    ):
        mock_avail.return_value = {"missing/image:latest": True}
        mock_extract.return_value = None  # no manifest extracted
        mock_validate.return_value = [
            "Test 'bad_test': No manifest available for image 'missing/image:latest'"
        ]

        results, container_results = _call_inner_workflow(
            suite_config,
            systems_config,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
        )

    assert results["summary"]["status"] == "VALIDATION_FAILED"
    assert results["summary"]["total_tests"] == 0
    assert results["summary"]["successful_tests"] == 0
    assert results["summary"]["failed_tests"] == 0
    assert results["results"] == []

    assert len(container_results) == 0


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
            test_id="t1 systemA",
            test_name="t1_systemA",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={"system_under_test": {"type": "llm_api"}},
            test_params={"p": "v"},
            container_config=ContainerConfig(),
        )

    assert result.success is True
    assert result.exit_code == 0
    assert result.container_id == "abc123"
    assert result.results.get("success") is True


def test_save_results_to_file_step_calls_impl(tmp_path):
    data = {"summary": {"status": "COMPLETED"}}
    out = tmp_path / "res.json"
    with patch("asqi.workflow.save_results_to_file") as save_mock:
        inner_step = getattr(
            save_results_to_file_step, "__wrapped__", save_results_to_file_step
        )
        inner_step(data, str(out))
        save_mock.assert_called_once_with(data, str(out))


def test_save_container_results_to_file(tmp_path):
    data = [{"test_results": {"success": "true"}}]
    logsFile, logsFolder = "container_res.json", "logs"
    out = tmp_path / logsFile
    with patch("asqi.workflow.save_container_results_to_file") as save_mock:
        inner_step = getattr(
            save_container_results_to_file_step,
            "__wrapped__",
            save_container_results_to_file_step,
        )
        inner_step(data, str(out))
        save_mock.assert_called_once_with(data, logsFolder, logsFile)


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
            test_id="failing_test",
            test_name="failing test",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={"system_under_test": {"type": "llm_api"}},
            test_params={},
            container_config=ContainerConfig(),
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
            test_id="json_test",
            test_name="json test",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={"system_under_test": {"type": "llm_api"}},
            test_params={},
            container_config=ContainerConfig(),
        )

    assert result.success is False
    assert "Failed to parse JSON output" in result.error_message


def test_execute_single_test_container_error_in_json():
    """Test that errors in JSON output are extracted to error_message field."""
    # Simulate container that exits with code 1 and includes error in JSON
    container_json_output = json.dumps(
        {
            "results": {
                "success": False,
                "error": "Dataset validation failed: missing required column 'label'",
            },
            "test_results": None,
            "generated_reports": [],
            "generated_datasets": [],
        }
    )

    with patch("asqi.workflow.run_container_with_args") as run_mock:
        run_mock.return_value = {
            "success": False,  # Exit code != 0
            "exit_code": 1,
            "output": container_json_output,
            "error": "",  # No Docker-level error
            "container_id": "abc123",
        }

        inner_step = getattr(execute_single_test, "__wrapped__", execute_single_test)
        result = inner_step(
            test_id="validation_error_test",
            test_name="validation error test",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={"system_under_test": {"type": "llm_api"}},
            test_params={},
            container_config=ContainerConfig(),
        )

    # Verify that error from JSON is extracted
    assert result.success is False
    assert result.exit_code == 1
    assert "Dataset validation failed" in result.error_message
    assert "missing required column 'label'" in result.error_message


def test_execute_single_test_combined_docker_and_json_errors():
    """Test that Docker errors and JSON errors are combined."""
    container_json_output = json.dumps(
        {
            "results": {"success": False, "error": "Internal validation error"},
            "test_results": None,
            "generated_reports": [],
            "generated_datasets": [],
        }
    )

    with patch("asqi.workflow.run_container_with_args") as run_mock:
        run_mock.return_value = {
            "success": False,
            "exit_code": 137,  # OOM killed
            "output": container_json_output,
            "error": "Container exceeded memory limit",
            "container_id": "abc123",
        }

        inner_step = getattr(execute_single_test, "__wrapped__", execute_single_test)
        result = inner_step(
            test_id="combined_error_test",
            test_name="combined error test",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={"system_under_test": {"type": "llm_api"}},
            test_params={},
            container_config=ContainerConfig(),
        )

    # Verify both errors are present
    assert result.success is False
    assert "Container exceeded memory limit" in result.error_message
    assert "Internal validation error" in result.error_message


def test_execute_single_test_env_file_falsy_values():
    """Test that env_file processing is skipped when env_file has falsy values."""
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

        # Test with empty string env_file - should skip env_file processing
        result = inner_step(
            test_id="test_empty_env_file",
            test_name="test empty env file",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={
                "system_under_test": {
                    "type": "llm_api",
                    "env_file": "",  # Empty string - should be treated as falsy
                }
            },
            test_params={},
            container_config=ContainerConfig(),
        )

        assert result.success is True
        # Should not have tried to load env file

        # Test with None env_file - should skip env_file processing
        result = inner_step(
            test_id="test_none_env_file",
            test_name="test none env file",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={
                "system_under_test": {
                    "type": "llm_api",
                    "env_file": None,  # None value - should be treated as falsy
                }
            },
            test_params={},
            container_config=ContainerConfig(),
        )

        assert result.success is True
        # Should not have tried to load env file

        # Test with missing env_file key - should skip env_file processing
        result = inner_step(
            test_id="test_missing_env_file",
            test_name="test missing env file",
            image="test/image:latest",
            sut_name="systemA",
            systems_params={
                "system_under_test": {
                    "type": "llm_api"
                    # No env_file key - should skip env_file processing
                }
            },
            test_params={},
            container_config=ContainerConfig(),
        )

        assert result.success is True
        # Should not have tried to load env file


def test_convert_test_results_to_objects():
    """Test converting test results data back to TestExecutionResult objects."""

    test_results_data = {
        "results": [
            {
                "metadata": {
                    "test_id": "test1",
                    "test_name": "test1",
                    "sut_name": "system1",
                    "image": "test/image:latest",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "success": True,
                    "container_id": "abc123",
                    "exit_code": 0,
                },
                "test_results": {"success": True, "score": 0.9},
                "generated_reports": [],
            }
        ]
    }

    test_container_data = [
        {
            "error_message": "",
            "container_output": '{"success": true}',
        }
    ]

    inner_step = getattr(
        convert_test_results_to_objects, "__wrapped__", convert_test_results_to_objects
    )
    results = inner_step(test_results_data, test_container_data)

    assert len(results) == 1
    result = results[0]
    assert result.test_name == "test1"
    assert result.test_id == "test1"
    assert result.sut_name == "system1"
    assert result.image == "test/image:latest"
    assert result.start_time == 1.0
    assert result.end_time == 2.0
    assert result.success is True
    assert result.container_id == "abc123"
    assert result.exit_code == 0
    assert result.results == {"success": True, "score": 0.9}
    assert result.generated_reports == []


def test_add_score_cards_to_results():
    """Test adding score card evaluation results to test results data."""

    test_results_data = {"summary": {"status": "COMPLETED"}, "results": []}

    score_card_evaluation = [
        {
            "indicator_id": "test_success",
            "indicator_name": "Test success",
            "test_name": "test1",
            "test_id": "test1",
            "sut_name": "system1",
            "outcome": "PASS",
            "score_card_name": "Test scorecard",
        }
    ]

    inner_step = getattr(
        add_score_cards_to_results, "__wrapped__", add_score_cards_to_results
    )
    result = inner_step(test_results_data, score_card_evaluation)

    assert "score_card" in result
    assert result["score_card"]["score_card_name"] == "Test scorecard"
    assert result["score_card"]["total_evaluations"] == 1
    assert len(result["score_card"]["assessments"]) == 1
    assert result["score_card"]["assessments"][0]["outcome"] == "PASS"
    assert "score_card_name" not in result["score_card"]["assessments"][0]


def test_add_score_cards_to_results_multiple_score_cards():
    """Test adding multiple score cards creates array structure."""

    test_results_data = {"summary": {"status": "COMPLETED"}, "results": []}

    score_card_evaluation = [
        {
            "indicator_id": "test_1",
            "indicator_name": "Test 1",
            "outcome": "PASS",
            "score_card_name": "Scorecard A",
        },
        {
            "indicator_id": "test_2",
            "indicator_name": "Test 2",
            "outcome": "FAIL",
            "score_card_name": "Scorecard B",
        },
    ]

    inner_step = getattr(
        add_score_cards_to_results, "__wrapped__", add_score_cards_to_results
    )
    result = inner_step(test_results_data, score_card_evaluation)

    assert isinstance(result["score_card"], list)
    assert len(result["score_card"]) == 2
    assert result["score_card"][0]["score_card_name"] == "Scorecard A"
    assert result["score_card"][1]["score_card_name"] == "Scorecard B"


def test_evaluate_score_cards_workflow():
    """Test the evaluate_score_cards_workflow function."""

    test_results_data = {
        "summary": {"status": "COMPLETED"},
        "results": [
            {
                "metadata": {
                    "test_id": "test1",
                    "test_name": "test1",
                    "sut_name": "system1",
                    "image": "test/image:latest",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "success": True,
                    "container_id": "abc123",
                    "exit_code": 0,
                },
                "test_results": {"success": True},
            }
        ],
    }

    test_container_data = [
        {
            "test_id": "test1",
            "error_message": "",
            "container_output": "",
        }
    ]

    score_card_configs = [{"score_card_name": "Test scorecard", "indicators": []}]

    with (
        patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
        patch("asqi.workflow.evaluate_score_card") as mock_evaluate,
        patch("asqi.workflow.add_score_cards_to_results") as mock_add,
        patch("asqi.workflow.console") as mock_console,
    ):
        mock_convert.return_value = []
        mock_evaluate.return_value = []
        mock_add.return_value = test_results_data

        inner_workflow = getattr(
            evaluate_score_cards_workflow, "__wrapped__", evaluate_score_cards_workflow
        )
        _result = inner_workflow(
            test_results_data, test_container_data, score_card_configs
        )

        mock_convert.assert_called_once_with(test_results_data, test_container_data)
        mock_evaluate.assert_called_once()
        mock_add.assert_called_once()
        mock_console.print.assert_called_once()


def test_evaluate_score_cards_workflow_with_audit_responses():
    """End-to-end-ish test: audit-only score card + audit responses."""

    # No metric-based indicators needed because scorecard contains only audit indicators
    test_results_data = {
        "summary": {"status": "COMPLETED"},
        "results": [],  # audit indicators don't need test_results
    }

    test_container_data = []

    score_card_configs = [MOCK_SCORE_CARD_CONFIG]
    audit_responses_data = MOCK_AUDIT_RESPONSES

    inner_workflow = getattr(
        evaluate_score_cards_workflow, "__wrapped__", evaluate_score_cards_workflow
    )

    result = inner_workflow(
        test_results_data,
        test_container_data,
        score_card_configs,
        audit_responses_data,
    )

    # We expect a single score_card block with both audit indicators evaluated
    assert "score_card" in result
    score = result["score_card"]
    assert score["score_card_name"] == "Mock Chatbot Scorecard"
    assert score["total_evaluations"] == 2

    # Map indicator_id -> outcome for easy checking
    outcomes = {a["indicator_id"]: a["outcome"] for a in score["assessments"]}

    assert outcomes["config_easy"] == "A"
    assert outcomes["config_v2"] == "C"

    # Optional: check notes/description wiring as well
    notes_by_id = {a["indicator_id"]: a["audit_notes"] for a in score["assessments"]}
    assert notes_by_id["config_easy"] == "ok"
    assert notes_by_id["config_v2"] == "ok"


def test_run_end_to_end_workflow():
    """Test the run_end_to_end_workflow function."""

    suite_config = {"suite_name": "test"}
    systems_config = {"systems_under_test": {}}
    score_card_configs = [{"score_card_name": "test"}]
    container_config: ContainerConfig = ContainerConfig()

    test_results = {"summary": {"status": "COMPLETED"}, "results": []}
    test_container = []
    final_results = {
        "summary": {"status": "COMPLETED"},
        "results": [],
        "score_card": {},
    }

    with (
        patch("asqi.workflow.run_test_suite_workflow") as mock_test_workflow,
        patch("asqi.workflow.evaluate_score_cards_workflow") as mock_score_workflow,
    ):
        mock_test_workflow.return_value = test_results, []
        mock_score_workflow.return_value = final_results

        inner_workflow = getattr(
            run_end_to_end_workflow, "__wrapped__", run_end_to_end_workflow
        )
        result, _ = inner_workflow(
            suite_config,
            systems_config,
            score_card_configs,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
        )

        mock_test_workflow.assert_called_once_with(
            suite_config,
            systems_config,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
            None,  # datasets_config
            score_card_configs,
        )
        mock_score_workflow.assert_called_once_with(
            test_results, test_container, score_card_configs, None
        )
        assert result == final_results


def test_run_end_to_end_workflow_with_audit_responses():
    """Ensure run_end_to_end_workflow forwards audit_responses_data."""

    suite_config = {"suite_name": "test"}
    systems_config = {"systems_under_test": {}}
    score_card_configs = [MOCK_SCORE_CARD_CONFIG]
    container_config: ContainerConfig = ContainerConfig()
    audit_responses_data = MOCK_AUDIT_RESPONSES

    test_results = {"summary": {"status": "COMPLETED"}, "results": []}
    test_container = []
    final_results = {
        "summary": {"status": "COMPLETED"},
        "results": [],
        "score_card": {},
    }

    with (
        patch("asqi.workflow.run_test_suite_workflow") as mock_test_workflow,
        patch("asqi.workflow.evaluate_score_cards_workflow") as mock_score_workflow,
    ):
        mock_test_workflow.return_value = test_results, test_container
        mock_score_workflow.return_value = final_results

        inner_workflow = getattr(
            run_end_to_end_workflow, "__wrapped__", run_end_to_end_workflow
        )
        result, _ = inner_workflow(
            suite_config,
            systems_config,
            score_card_configs,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
            audit_responses_data,
        )

        mock_test_workflow.assert_called_once_with(
            suite_config,
            systems_config,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
            None,  # datasets_config
            score_card_configs,
        )

        mock_score_workflow.assert_called_once_with(
            test_results, test_container, score_card_configs, audit_responses_data
        )
        assert result == final_results


def test_start_test_execution_tests_only_mode():
    """Test start_test_execution with `ExecutionMode.TESTS_ONLY`."""

    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}}, return_tuple=True)

    with (
        patch("asqi.workflow.load_config_file") as mock_load,
        patch("asqi.workflow.DBOS.start_workflow") as mock_start,
    ):
        mock_load.return_value = {"test": "config"}
        mock_start.return_value = mock_handle

        workflow_id = start_test_execution(
            "suite.yaml",
            "systems.yaml",
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            ContainerConfig(),
            None,
            "output.json",
            None,
            ExecutionMode.TESTS_ONLY,
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call run_test_suite_workflow for ExecutionMode.TESTS_ONLY
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "run_test_suite_workflow"


def test_start_test_execution_end_to_end_mode():
    """Test start_test_execution with `ExecutionMode.END_TO_END`."""

    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}}, return_tuple=True)
    score_card_configs = [{"score_card_name": "test"}]

    with (
        patch("asqi.workflow.load_config_file") as mock_load,
        patch("asqi.workflow.DBOS.start_workflow") as mock_start,
    ):
        mock_load.return_value = {"test": "config"}
        mock_start.return_value = mock_handle

        workflow_id = start_test_execution(
            "suite.yaml",
            "systems.yaml",
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            ContainerConfig(),
            None,
            "output.json",
            score_card_configs,
            ExecutionMode.END_TO_END,
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call run_end_to_end_workflow for end_to_end mode with score cards
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "run_end_to_end_workflow"


def test_start_test_execution_end_to_end_mode_with_audit_responses():
    """start_test_execution should pass audit_responses_data down for `ExecutionMode.END_TO_END`."""

    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}}, return_tuple=True)
    score_card_configs = [MOCK_SCORE_CARD_CONFIG]
    audit_responses_data = MOCK_AUDIT_RESPONSES

    with (
        patch("asqi.workflow.load_config_file") as mock_load,
        patch("asqi.workflow.DBOS.start_workflow") as mock_start,
    ):
        mock_load.return_value = {"test": "config"}
        mock_start.return_value = mock_handle

        workflow_id = start_test_execution(
            "suite.yaml",
            "systems.yaml",
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            ContainerConfig(),
            audit_responses_data,
            "output.json",
            score_card_configs,
            ExecutionMode.END_TO_END,
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()

        call_args = mock_start.call_args[0]
        # Should call run_end_to_end_workflow
        assert call_args[0].__name__ == "run_end_to_end_workflow"
        # audit_responses_data is the second-to-last positional arg
        assert call_args[-2] == audit_responses_data


def test_start_score_card_evaluation(tmp_path):
    """Test start_score_card_evaluation function."""

    test_data = {"summary": {"status": "COMPLETED"}, "results": []}
    score_card_configs = [{"score_card_name": "test"}]
    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}})

    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    with open(input_json, "w") as f:
        json.dump(test_data, f)

    with patch("asqi.workflow.DBOS.start_workflow") as mock_start:
        mock_start.return_value = mock_handle

        workflow_id = start_score_card_evaluation(
            str(input_json), score_card_configs, None, str(output_json)
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call evaluate_score_cards_workflow
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "evaluate_score_cards_workflow"


def test_start_score_card_evaluation_with_audit_responses(tmp_path):
    """start_score_card_evaluation should forward audit_responses_data."""

    test_data = {"summary": {"status": "COMPLETED"}, "results": []}
    score_card_configs = [MOCK_SCORE_CARD_CONFIG]
    audit_responses_data = MOCK_AUDIT_RESPONSES
    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}})

    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    with open(input_json, "w") as f:
        json.dump(test_data, f)

    with patch("asqi.workflow.DBOS.start_workflow") as mock_start:
        mock_start.return_value = mock_handle

        workflow_id = start_score_card_evaluation(
            str(input_json), score_card_configs, audit_responses_data, str(output_json)
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()

        call_args = mock_start.call_args[0]
        # Function should be evaluate_score_cards_workflow
        assert call_args[0].__name__ == "evaluate_score_cards_workflow"
        # audit_responses_data is the last positional arg
        assert call_args[4] == audit_responses_data


def test_image_pulled_but_manifest_not_extracted_bug():
    """Test that reproduces issue #150 where validation fails despite image being pulled.

    After pulling missing images, manifests are not extracted from newly pulled images,
    causing validation to fail even though the images are now available.
    """
    suite_config = {
        "suite_name": "test",
        "test_suite": [
            {
                "name": "test1",
                "id": "test1",
                "image": "test/image:latest",
                "systems_under_test": ["sys1"],
            }
        ],
    }

    systems_config = {
        "systems": {
            "sys1": {
                "type": "llm_api",
                "params": {"base_url": "http://x", "model": "x-model"},
            }
        }
    }

    manifest = Manifest(
        name="test",
        version="1",
        description="",
        input_systems=[
            SystemInput(name="system_under_test", type="llm_api", required=True)
        ],
        input_schema=[],
        output_metrics=[],
        output_artifacts=None,
    )

    with (
        patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
        patch("asqi.workflow.dbos_pull_images"),
        patch("asqi.workflow.extract_manifests_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
        patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        patch("asqi.workflow.Queue") as mock_queue,
    ):
        # Image not available initially, then available after pull
        mock_avail.side_effect = [
            {"test/image:latest": False},  # Before pull
            {"test/image:latest": True},  # After pull (our fix enables this)
        ]

        mock_extract.return_value = {"test/image:latest": manifest}
        mock_validate.side_effect = (
            lambda s, sys, manifests: [] if manifests else ["No manifest"]
        )
        mock_plan.return_value = [
            {
                "test_id": "test1",
                "test_name": "test1",
                "image": "test/image:latest",
                "sut_name": "sys1",
                "systems_params": {"system_under_test": {"type": "llm_api"}},
                "test_params": {},
            }
        ]

        success_result = TestExecutionResult(
            "test1", "test1", "sys1", "test/image:latest"
        )
        success_result.success = True
        mock_queue.return_value.enqueue.return_value = DummyHandle(success_result)

        results, _ = _call_inner_workflow(
            suite_config,
            systems_config,
            {"concurrent_tests": 1, "max_failures": 10, "progress_interval": 10},
            ContainerConfig(),
        )

        assert results["summary"]["status"] == "COMPLETED"


def test_run_test_suite_workflow_handle_exception():
    """Test that exceptions from test execution handles are caught and handled gracefully."""
    suite_config = {
        "suite_name": "demo",
        "test_suite": [
            {
                "name": "t1",
                "id": "t1",
                "image": "test/image:latest",
                "systems_under_test": ["systemA"],
                "params": {"p": "v"},
            }
        ],
    }

    systems_config = {
        "systems": {
            "systemA": {
                "type": "llm_api",
                "params": {"base_url": "http://x", "model": "x-model"},
            }
        }
    }

    container_config: ContainerConfig = ContainerConfig()

    # Build a minimal manifest
    manifest = Manifest(
        name="mock",
        version="1",
        description="",
        input_systems=[
            SystemInput(
                name="system_under_test", type="llm_api", required=True, description=""
            )
        ],
        input_schema=[],
        output_metrics=[],
        output_artifacts=None,
    )

    with (
        patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
        patch("asqi.workflow.extract_manifests_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
        patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        patch("asqi.workflow.Queue") as mock_queue_class,
    ):
        mock_avail.return_value = {"test/image:latest": True}
        mock_extract.return_value = {"test/image:latest": manifest}
        mock_validate.return_value = []
        mock_plan.return_value = [
            {
                "test_id": "t1_systemA",
                "test_name": "t1 systemA",
                "image": "test/image:latest",
                "sut_name": "systemA",
                "systems_params": {
                    "system_under_test": {
                        "type": "llm_api",
                        "params": {"base_url": "http://x", "endpoint": "http://x"},
                    }
                },
                "test_params": {"p": "v"},
            }
        ]

        # Create a handle that raises an exception
        failing_handle = DummyHandle(None)  # get_result will raise AttributeError
        failing_handle.get_result = Mock(side_effect=Exception("Network timeout"))

        mock_queue = mock_queue_class.return_value
        mock_queue.enqueue.side_effect = lambda *args, **kwargs: failing_handle

        results, container_results = _call_inner_workflow(
            suite_config,
            systems_config,
            {
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config,
        )

    assert results["summary"]["status"] == "COMPLETED"
    assert results["summary"]["total_tests"] == 1
    assert results["summary"]["successful_tests"] == 0
    assert results["summary"]["failed_tests"] == 1
    assert len(results["results"]) == 1
    assert (
        "Test execution failed: Network timeout"
        in container_results[0]["error_message"]
    )

    result = results["results"][0]
    assert result["metadata"]["success"] is False
    assert result["metadata"]["exit_code"] == 137
    assert result["metadata"]["test_name"] == "t1 systemA"
    assert result["metadata"]["test_id"] == "t1_systemA"
    assert result["metadata"]["image"] == "test/image:latest"


class TestContainerReports:
    def test_invalid_file_error(self, tmp_path):
        """
        Test validation fails when report file does not exist.
        """
        from asqi.response_schemas import GeneratedReport

        manifests = {}

        result = TestExecutionResult(
            "test report file not exist", "test_not_exist", "sut", "report-image:latest"
        )
        result.success = True
        result.generated_reports = [
            GeneratedReport(
                report_name="test_report",
                report_type="html",
                report_path=str(tmp_path / "invalid.html"),
            )
        ]

        errors = validate_test_container_reports([result], manifests)

        assert len(errors) == 1
        assert result.success is False
        assert "Report 'test_report' does not exist at path" in result.error_message

    def test_multiple_report_errors(self, tmp_path):
        """
        Test that errors from multiple reports are all collected.
        """
        from asqi.response_schemas import GeneratedReport

        manifests = {}

        result = TestExecutionResult(
            "test report file not exist", "test_not_exist", "sut", "report-image:latest"
        )
        result.success = True
        result.generated_reports = [
            GeneratedReport(
                report_name="first_report",
                report_type="html",
                report_path=str(tmp_path / "invalid.html"),
            ),
            GeneratedReport(
                report_name="second_report",
                report_type="html",
                report_path=str(tmp_path / "invalid.html"),
            ),
        ]

        errors = validate_test_container_reports([result], manifests)

        assert len(errors) == 1
        assert result.success is False
        assert "Report 'first_report' does not exist at path" in result.error_message
        assert "Report 'second_report' does not exist at path" in result.error_message

    def test_validate_test_container_reports_success(self, tmp_path):
        """
        Test validation passes when the report returned by the test container matches the manifest definitions
        """
        from asqi.response_schemas import GeneratedReport

        report_file_html = tmp_path / "valid_report.html"
        report_file_html.write_text("some content")
        report_file_pdf = tmp_path / "valid_report.pdf"
        report_file_pdf.write_text("some content")
        manifests = {
            "report-image:latest": Manifest(
                name="report-manifest",
                version="1.0",
                input_systems=[],
                output_reports=[
                    OutputReports(name="valid_report", type="html"),
                    OutputReports(name="another_report", type="pdf"),
                ],
            )
        }

        result = TestExecutionResult(
            "test success", "test_success", "sut", "report-image:latest"
        )
        result.success = True
        result.generated_reports = [
            GeneratedReport(
                report_name="valid_report",
                report_type="html",
                report_path=str(report_file_html),
            ),
            GeneratedReport(
                report_name="another_report",
                report_type="pdf",
                report_path=str(report_file_pdf),
            ),
        ]
        errors = validate_test_container_reports([result], manifests)
        assert len(errors) == 0
        assert result.success is True

    def test_skips_failed_tests(self):
        """
        Test that validation skips tests that already failed.
        """
        manifests = {}

        result = TestExecutionResult("test", "test_error", "sut", "report-image:latest")
        result.success = False
        result.error_message = "Test failed for other reasons"
        result.generated_reports = [
            {
                "report_name": "summary",
                "report_type": "html",
                "report_path": "report.html",
            }
        ]

        errors = validate_test_container_reports([result], manifests)

        assert len(errors) == 0
        assert result.success is False
        assert result.error_message == "Test failed for other reasons"


# =============================================================================
# Data Generation Workflow Tests
# =============================================================================


class TestDataGenerationWorkflow:
    """Test data generation workflow with optional systems config."""

    def test_start_data_generation_with_systems(self, tmp_path):
        """Test start_data_generation with systems config provided."""
        from asqi.workflow import start_data_generation

        # Create test config files
        generation_config_file = tmp_path / "generation.yaml"
        generation_config_file.write_text("""
job_name: "Test Generation Job"
generation_jobs:
  - id: "job1"
    name: "Test Job"
    systems:
      generation_system: "gpt4o"
    image: "my-registry/sdg:latest"
    params:
      num_samples: 10
""")

        systems_config_file = tmp_path / "systems.yaml"
        systems_config_file.write_text("""
systems:
  gpt4o:
    type: "llm_api"
    params:
      model: "gpt-4o"
      base_url: "https://api.openai.com/v1"
      api_key: "test-key"
""")

        output_file = tmp_path / "output.json"

        executor_config = {
            "concurrent_tests": 1,
            "max_failures": 1,
            "progress_interval": 1,
        }
        container_config = ContainerConfig()

        # Mock DBOS workflow
        mock_result = {"summary": {"status": "COMPLETED"}, "results": []}
        mock_handle = DummyHandle(
            mock_result, workflow_id="test-gen-123", return_tuple=True
        )

        with patch("asqi.workflow.DBOS") as mock_dbos:
            mock_dbos.start_workflow.return_value = mock_handle
            mock_dbos.workflow_id = "test-gen-123"

            workflow_id = start_data_generation(
                generation_config_path=str(generation_config_file),
                systems_path=str(systems_config_file),
                executor_config=executor_config,
                container_config=container_config,
                output_path=str(output_file),
            )

            assert workflow_id == "test-gen-123"
            # Verify workflow was called with non-None systems config
            call_args = mock_dbos.start_workflow.call_args
            assert call_args[0][1] is not None  # systems_config arg

    def test_start_data_generation_without_systems(self, tmp_path):
        """Test start_data_generation with None systems config."""
        from asqi.workflow import start_data_generation

        # Create test config file
        generation_config_file = tmp_path / "generation.yaml"
        generation_config_file.write_text("""
job_name: "Template Generation Job"
generation_jobs:
  - id: "job2"
    name: "Template Job"
    image: "my-registry/template:latest"
    params:
      template: "simple"
""")

        output_file = tmp_path / "output.json"

        executor_config = {
            "concurrent_tests": 1,
            "max_failures": 1,
            "progress_interval": 1,
        }
        container_config = ContainerConfig()

        # Mock DBOS workflow
        mock_result = {"summary": {"status": "COMPLETED"}, "results": []}
        mock_handle = DummyHandle(
            mock_result, workflow_id="test-gen-456", return_tuple=True
        )

        with patch("asqi.workflow.DBOS") as mock_dbos:
            mock_dbos.start_workflow.return_value = mock_handle
            mock_dbos.workflow_id = "test-gen-456"

            workflow_id = start_data_generation(
                generation_config_path=str(generation_config_file),
                systems_path=None,  # No systems config
                executor_config=executor_config,
                container_config=container_config,
                output_path=str(output_file),
            )

            assert workflow_id == "test-gen-456"
            # Verify start_workflow was called
            assert mock_dbos.start_workflow.called

    def test_start_data_generation_validation_error(self):
        """Test start_data_generation with invalid inputs."""
        from asqi.workflow import start_data_generation

        executor_config = {
            "concurrent_tests": 1,
            "max_failures": 1,
            "progress_interval": 1,
        }
        container_config = ContainerConfig()

        # Test with empty generation config path
        with pytest.raises(ValueError, match="Invalid generation_config_path"):
            start_data_generation(
                generation_config_path="",
                systems_path=None,
                executor_config=executor_config,
                container_config=container_config,
            )

        # Test with invalid systems path type
        with pytest.raises(ValueError, match="Invalid systems_path"):
            start_data_generation(
                generation_config_path="valid.yaml",
                systems_path=123,  # type: ignore[arg-type]  # Intentionally invalid type for testing
                executor_config=executor_config,
                container_config=container_config,
            )

    def test_run_data_generation_workflow_parses_systems_config(self):
        """Test that run_data_generation_workflow correctly parses systems config."""
        from asqi.schemas import DataGenerationConfig, SystemsConfig

        generation_config_dict = {
            "job_name": "Test",
            "generation_jobs": [
                {
                    "id": "job1",
                    "name": "Test Job",
                    "systems": {"generation_system": "gpt4o"},
                    "image": "my-registry/sdg:latest",
                    "params": {"num_samples": 10},
                }
            ],
        }

        systems_config_dict = {
            "systems": {
                "gpt4o": {
                    "type": "llm_api",
                    "params": {
                        "model": "gpt-4o",
                        "base_url": "https://api.openai.com/v1",
                    },
                }
            }
        }

        # Test that configs can be parsed successfully
        gen_config = DataGenerationConfig(**generation_config_dict)
        sys_config = SystemsConfig(**systems_config_dict)

        assert gen_config.job_name == "Test"
        assert len(gen_config.generation_jobs) == 1
        assert gen_config.generation_jobs[0].systems == {"generation_system": "gpt4o"}
        assert "gpt4o" in sys_config.systems

    def test_run_data_generation_workflow_parses_no_systems_config(self):
        """Test that run_data_generation_workflow handles None systems config."""
        from asqi.schemas import DataGenerationConfig

        generation_config_dict = {
            "job_name": "Template Test",
            "generation_jobs": [
                {
                    "id": "job2",
                    "name": "Template Job",
                    "image": "my-registry/template:latest",
                    "params": {"template": "simple"},
                }
            ],
        }

        # Test that config can be parsed without systems
        gen_config = DataGenerationConfig(**generation_config_dict)

        assert gen_config.job_name == "Template Test"
        assert len(gen_config.generation_jobs) == 1
        assert gen_config.generation_jobs[0].systems is None


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for utility and helper functions."""

    def test_get_docker_socket_path_with_custom_env(self):
        """Test _get_docker_socket_path with custom DOCKER_HOST environment variable."""
        from asqi.workflow import _get_docker_socket_path

        env_vars = {"DOCKER_HOST": "unix:///custom/docker.sock"}
        socket_path = _get_docker_socket_path(env_vars)
        assert socket_path == "/custom/docker.sock"

    def test_get_docker_socket_path_with_default(self):
        """Test _get_docker_socket_path with default socket path when no env var."""
        from asqi.workflow import _get_docker_socket_path

        socket_path = _get_docker_socket_path({})
        assert socket_path == "/var/run/docker.sock"

    def test_get_docker_socket_path_empty_env_var(self):
        """Test _get_docker_socket_path with empty DOCKER_HOST."""
        from asqi.workflow import _get_docker_socket_path

        env_vars = {"DOCKER_HOST": ""}
        socket_path = _get_docker_socket_path(env_vars)
        assert socket_path == "/var/run/docker.sock"

    def test_get_available_images_all_local(self):
        """Test _get_available_images when all images are available locally."""
        from asqi.workflow import _get_available_images

        images = ["test/image1:latest", "test/image2:latest"]

        with patch("asqi.workflow.dbos_check_images_availability") as mock_check:
            mock_check.return_value = {img: True for img in images}

            available, availability = _get_available_images(images)

            assert len(available) == 2
            assert "test/image1:latest" in available
            assert "test/image2:latest" in available
            assert all(availability.values())
            mock_check.assert_called_once_with(images)

    def test_get_available_images_some_missing(self):
        """Test _get_available_images when some images need to be pulled."""
        from asqi.workflow import _get_available_images

        images = ["local/image:v1", "remote/image:v1"]

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_check,
            patch("asqi.workflow.dbos_pull_images") as mock_pull,
        ):
            # First check: one missing, second check: both available after pull
            mock_check.side_effect = [
                {"local/image:v1": True, "remote/image:v1": False},
                {"local/image:v1": True, "remote/image:v1": True},
            ]

            available, availability = _get_available_images(images)

            assert len(available) == 2
            assert all(availability.values())
            mock_pull.assert_called_once_with(["remote/image:v1"])
            assert mock_check.call_count == 2

    def test_get_available_images_all_missing(self):
        """Test _get_available_images when all images need to be pulled."""
        from asqi.workflow import _get_available_images

        images = ["remote/image1:latest", "remote/image2:latest"]

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_check,
            patch("asqi.workflow.dbos_pull_images") as mock_pull,
        ):
            # First check: all missing, second check: all available after pull
            mock_check.side_effect = [
                {img: False for img in images},
                {img: True for img in images},
            ]

            available, availability = _get_available_images(images)

            assert len(available) == 2
            mock_pull.assert_called_once_with(images)

    def test_get_available_images_empty_list(self):
        """Test _get_available_images with empty image list."""
        from asqi.workflow import _get_available_images

        with patch("asqi.workflow.dbos_check_images_availability") as mock_check:
            mock_check.return_value = {}

            available, availability = _get_available_images([])

            assert len(available) == 0
            assert len(availability) == 0

    def test_get_available_images_pull_fails(self):
        """Test _get_available_images when pull fails (images remain unavailable)."""
        from asqi.workflow import _get_available_images

        images = ["nonexistent/image:latest"]

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_check,
            patch("asqi.workflow.dbos_pull_images") as mock_pull,
        ):
            # Both checks return unavailable (pull failed)
            mock_check.side_effect = [
                {"nonexistent/image:latest": False},
                {"nonexistent/image:latest": False},
            ]

            available, availability = _get_available_images(images)

            assert len(available) == 0
            assert not availability["nonexistent/image:latest"]
            mock_pull.assert_called_once()


# =============================================================================
# Environment Variable and Multi-System Tests
# =============================================================================


class TestEnvironmentVariablesAndSystems:
    """Tests for environment variable handling and multiple systems."""

    def test_execute_single_test_env_var_merging(self, tmp_path):
        """Test that test-level env vars override system-level."""
        # Create test env files
        system_env_file = tmp_path / "system.env"
        system_env_file.write_text("SHARED_VAR=system_value\nSYSTEM_ONLY=sys_val")

        test_env_file = tmp_path / "test.env"
        test_env_file.write_text("SHARED_VAR=test_value\nTEST_ONLY=test_val")

        fake_output = '{"success": true}'
        with patch("asqi.workflow.run_container_with_args") as run_mock:
            run_mock.return_value = {
                "success": True,
                "exit_code": 0,
                "output": fake_output,
                "error": "",
                "container_id": "abc",
            }

            inner_step = getattr(
                execute_single_test, "__wrapped__", execute_single_test
            )
            result = inner_step(
                test_id="env_test",
                test_name="env test",
                image="test/image:latest",
                sut_name="systemA",
                systems_params={
                    "system_under_test": {
                        "type": "llm_api",
                        "env_file": str(system_env_file),
                    }
                },
                test_params={},
                container_config=ContainerConfig(),
                env_file=str(test_env_file),  # Test level (should override)
                environment={"OVERRIDE_VAR": "explicit_value"},
            )

            assert result.success is True
            # Verify container was called with merged env vars
            call_args = run_mock.call_args
            env_vars = call_args[1]["environment"]

            # Test-level should override system-level
            assert env_vars.get("SHARED_VAR") == "test_value"
            # Both should be present
            assert "SYSTEM_ONLY" in env_vars
            assert "TEST_ONLY" in env_vars
            # Explicit environment dict has highest priority
            assert env_vars.get("OVERRIDE_VAR") == "explicit_value"

    def test_execute_single_test_multiple_systems(self):
        """Test execution with multiple system roles (system_under_test + additional systems)."""
        fake_output = '{"success": true, "evaluation_score": 0.95}'
        with patch("asqi.workflow.run_container_with_args") as run_mock:
            run_mock.return_value = {
                "success": True,
                "exit_code": 0,
                "output": fake_output,
                "error": "",
                "container_id": "multi_sys_123",
            }

            systems_params = {
                "system_under_test": {
                    "type": "llm_api",
                    "model": "my-chatbot",
                    "base_url": "http://localhost:8000",
                },
                "simulator_system": {
                    "type": "llm_api",
                    "model": "gpt-4o",
                    "base_url": "https://api.openai.com/v1",
                },
                "evaluator_system": {
                    "type": "llm_api",
                    "model": "claude-3-5-sonnet-20241022",
                    "base_url": "https://api.anthropic.com/v1",
                },
            }

            inner_step = getattr(
                execute_single_test, "__wrapped__", execute_single_test
            )
            result = inner_step(
                test_id="multi_sys_test",
                test_name="multi system test",
                image="test/chatbot_simulator:latest",
                sut_name="my_chatbot",
                systems_params=systems_params,
                test_params={"num_conversations": 5},
                container_config=ContainerConfig(),
            )

            assert result.success is True
            # Verify all systems were passed to container
            call_kwargs = run_mock.call_args[1]
            # Args are passed as a list with --systems-params flag
            args_list = call_kwargs["args"]

            # Find the systems-params JSON in the args
            systems_json_idx = args_list.index("--systems-params") + 1
            systems_json = args_list[systems_json_idx]
            systems_dict = json.loads(systems_json)

            assert "system_under_test" in systems_dict
            assert "simulator_system" in systems_dict
            assert "evaluator_system" in systems_dict
            assert systems_dict["system_under_test"]["model"] == "my-chatbot"
            assert systems_dict["simulator_system"]["model"] == "gpt-4o"
            assert (
                systems_dict["evaluator_system"]["model"]
                == "claude-3-5-sonnet-20241022"
            )

    def test_execute_single_test_env_interpolation(self, monkeypatch):
        """Test environment variable interpolation with ${VAR} syntax."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        fake_output = '{"success": true}'
        with patch("asqi.workflow.run_container_with_args") as run_mock:
            run_mock.return_value = {
                "success": True,
                "exit_code": 0,
                "output": fake_output,
                "error": "",
                "container_id": "interp_123",
            }

            inner_step = getattr(
                execute_single_test, "__wrapped__", execute_single_test
            )
            result = inner_step(
                test_id="interp_test",
                test_name="interpolation test",
                image="test/image:latest",
                sut_name="systemA",
                systems_params={"system_under_test": {"type": "llm_api"}},
                test_params={},
                container_config=ContainerConfig(),
                environment={
                    "API_KEY": "${OPENAI_API_KEY}",
                    "LOG_LEVEL": "${LOG_LEVEL:-INFO}",  # Default value
                    "MISSING_VAR": "${MISSING:-default_value}",
                },
            )

            assert result.success is True
            call_args = run_mock.call_args
            env_vars = call_args[1]["environment"]

            # Check interpolation worked
            assert env_vars.get("API_KEY") == "sk-test-openai-key"
            assert env_vars.get("LOG_LEVEL") == "DEBUG"
            assert env_vars.get("MISSING_VAR") == "default_value"

    def test_execute_single_test_missing_required_env_vars(self, tmp_path):
        """Test handling of missing required environment variables."""
        # This would typically be validated before execution
        # Test that validation catches missing required env vars
        from asqi.schemas import EnvironmentVariable, Manifest, SystemInput

        manifest = Manifest(
            name="test-manifest",
            version="1.0",
            input_systems=[
                SystemInput(name="system_under_test", type="llm_api", required=True)
            ],
            environment_variables=[
                EnvironmentVariable(
                    name="REQUIRED_API_KEY",
                    required=True,
                    description="Required API key for the service",
                )
            ],
        )

        # Test that error message is helpful
        from asqi.validation import build_env_var_error_message

        missing_vars = [manifest.environment_variables[0]]
        error_msg = build_env_var_error_message(
            missing_vars, "my_test", "test/image:latest"
        )

        assert "REQUIRED_API_KEY" in error_msg
        assert "Required API key for the service" in error_msg
        assert "env_file" in error_msg
        assert "environment:" in error_msg


# =============================================================================
# Display Reports Validation Tests
# =============================================================================


class TestDisplayReportsValidation:
    """Tests for display reports validation functions."""

    def test_resolve_display_reports_inputs_end_to_end_mode(self):
        """Test that END_TO_END mode returns empty mappings."""
        from asqi.config import ExecutionMode
        from asqi.workflow import _resolve_display_reports_inputs

        test_result = TestExecutionResult("t1", "t1", "sys1", "test/image:v1")
        test_results = [test_result]

        test_id_to_image, manifests = _resolve_display_reports_inputs(
            test_results, ExecutionMode.END_TO_END
        )

        assert test_id_to_image == {}
        assert manifests == {}

    def test_resolve_display_reports_inputs_evaluate_only(self):
        """Test manifest extraction in EVALUATE_ONLY mode."""
        from asqi.config import ExecutionMode
        from asqi.schemas import SystemInput
        from asqi.workflow import _resolve_display_reports_inputs

        manifest = Manifest(
            name="test-manifest",
            version="1.0",
            input_systems=[
                SystemInput(name="system_under_test", type="llm_api", required=True)
            ],
        )

        test_result = TestExecutionResult("t1", "test_1", "sys1", "test/image:v1")
        test_results = [test_result]

        with (
            patch("asqi.workflow._get_available_images") as mock_get_images,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
        ):
            mock_get_images.return_value = (["test/image:v1"], {})
            mock_extract.return_value = {"test/image:v1": manifest}

            test_id_to_image, manifests = _resolve_display_reports_inputs(
                test_results, ExecutionMode.EVALUATE_ONLY
            )

            assert "test_1" in test_id_to_image
            assert test_id_to_image["test_1"] == "test/image:v1"
            assert "test/image:v1" in manifests
            assert manifests["test/image:v1"] == manifest

    def test_resolve_display_reports_inputs_multiple_tests_same_image(self):
        """Test with multiple tests using the same image."""
        from asqi.config import ExecutionMode
        from asqi.schemas import SystemInput
        from asqi.workflow import _resolve_display_reports_inputs

        manifest = Manifest(
            name="shared-manifest",
            version="1.0",
            input_systems=[
                SystemInput(name="system_under_test", type="llm_api", required=True)
            ],
        )

        test_results = [
            TestExecutionResult("t1", "test_1", "sys1", "shared/image:v1"),
            TestExecutionResult("t2", "test_2", "sys2", "shared/image:v1"),
        ]

        with (
            patch("asqi.workflow._get_available_images") as mock_get_images,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
        ):
            # Only one unique image should be extracted
            mock_get_images.return_value = (["shared/image:v1"], {})
            mock_extract.return_value = {"shared/image:v1": manifest}

            test_id_to_image, manifests = _resolve_display_reports_inputs(
                test_results, ExecutionMode.EVALUATE_ONLY
            )

            assert len(test_id_to_image) == 2
            assert len(manifests) == 1  # Only one unique image
            assert test_id_to_image["test_1"] == "shared/image:v1"
            assert test_id_to_image["test_2"] == "shared/image:v1"

    def test_validate_display_reports_valid(self):
        """Test validate_display_reports with valid display reports."""
        from asqi.schemas import OutputReports, ScoreCardIndicator, SystemInput
        from asqi.workflow import validate_display_reports

        manifest = Manifest(
            name="report-manifest",
            version="1.0",
            input_systems=[
                SystemInput(name="system_under_test", type="llm_api", required=True)
            ],
            output_reports=[
                OutputReports(name="summary_report", type="html"),
                OutputReports(name="detail_report", type="pdf"),
            ],
        )

        manifests = {"test/image:v1": manifest}
        test_id_to_image = {"test_1": "test/image:v1"}

        score_card = ScoreCard(
            score_card_name="Test Score Card",
            indicators=[
                ScoreCardIndicator(
                    id="ind_1",
                    name="Test Indicator",
                    apply_to={"test_id": "test_1"},
                    metric="success",
                    assessment=[],
                    display_reports=["summary_report"],
                )
            ],
        )

        # Should not raise any errors
        validate_display_reports(manifests, score_card, test_id_to_image)

    def test_validate_display_reports_invalid_report_name(self):
        """Test validate_display_reports with invalid report name."""
        from asqi.errors import ReportValidationError
        from asqi.schemas import OutputReports, ScoreCardIndicator, SystemInput
        from asqi.workflow import validate_display_reports

        manifest = Manifest(
            name="report-manifest",
            version="1.0",
            input_systems=[
                SystemInput(name="system_under_test", type="llm_api", required=True)
            ],
            output_reports=[OutputReports(name="valid_report", type="html")],
        )

        manifests = {"test/image:v1": manifest}
        test_id_to_image = {"test_1": "test/image:v1"}

        score_card = ScoreCard(
            score_card_name="Test Score Card",
            indicators=[
                ScoreCardIndicator(
                    id="ind_1",
                    name="Test Indicator",
                    apply_to={"test_id": "test_1"},
                    metric="success",
                    assessment=[],
                    display_reports=["invalid_report"],  # Not in manifest
                )
            ],
        )

        with pytest.raises(ReportValidationError):
            validate_display_reports(manifests, score_card, test_id_to_image)

    def test_validate_display_reports_missing_test_id(self):
        """Test validate_display_reports with missing test_id reference."""
        from asqi.errors import ReportValidationError
        from asqi.schemas import ScoreCardIndicator, SystemInput
        from asqi.workflow import validate_display_reports

        manifest = Manifest(
            name="report-manifest",
            version="1.0",
            input_systems=[
                SystemInput(name="system_under_test", type="llm_api", required=True)
            ],
        )

        manifests = {"test/image:v1": manifest}
        test_id_to_image = {"test_1": "test/image:v1"}

        score_card = ScoreCard(
            score_card_name="Test Score Card",
            indicators=[
                ScoreCardIndicator(
                    id="ind_1",
                    name="Test Indicator",
                    apply_to={"test_id": "nonexistent_test"},  # Test doesn't exist
                    metric="success",
                    assessment=[],
                    display_reports=["some_report"],
                )
            ],
        )

        with pytest.raises(ReportValidationError):
            validate_display_reports(manifests, score_card, test_id_to_image)
