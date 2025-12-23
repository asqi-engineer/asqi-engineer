import json
from unittest.mock import Mock, patch

import pytest

from asqi.config import ContainerConfig, ExecutorConfig
from asqi.schemas import Manifest, OutputReports, SystemInput
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
    validate_test_container_technical_reports,
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
                "technical_reports": [],
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
    assert result.test_results == {"success": True, "score": 0.9}
    assert result.technical_reports == []


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
            score_card_configs,
        )

        mock_score_workflow.assert_called_once_with(
            test_results, test_container, score_card_configs, audit_responses_data
        )
        assert result == final_results


def test_start_test_execution_tests_only_mode():
    """Test start_test_execution with tests_only mode."""

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
            "tests_only",
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call run_test_suite_workflow for tests_only mode
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "run_test_suite_workflow"


def test_start_test_execution_end_to_end_mode():
    """Test start_test_execution with end_to_end mode."""

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
            "end_to_end",
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call run_end_to_end_workflow for end_to_end mode with score cards
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "run_end_to_end_workflow"


def test_start_test_execution_end_to_end_mode_with_audit_responses():
    """start_test_execution should pass audit_responses_data down for end_to_end mode."""

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
            "end_to_end",
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()

        call_args = mock_start.call_args[0]
        # Should call run_end_to_end_workflow
        assert call_args[0].__name__ == "run_end_to_end_workflow"
        # audit_responses_data is the last positional arg passed into the workflow
        assert call_args[-1] == audit_responses_data


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


class TestTechnicalReports:
    def test_report_name_errors(self, tmp_path):
        """
        Test validation fails when report_name is missing, empty or invalid.
        """
        report_file = tmp_path / "report.html"
        report_file.write_text("content")

        manifests = {
            "report-image:latest": Manifest(
                name="test-manifest",
                version="1.0",
                input_systems=[],
                output_reports=[OutputReports(name="valid_report", type="html")],
            )
        }

        result_report_name_missing = TestExecutionResult(
            "test report name missing",
            "test_report_name_missing",
            "sut",
            "report-image:latest",
        )
        result_report_name_missing.success = True
        result_report_name_missing.technical_reports = [
            {"report_type": "html", "report_path": str(report_file)}
        ]

        result_report_name_empty = TestExecutionResult(
            "test report name empty",
            "test_report_name_empty",
            "sut",
            "report-image:latest",
        )
        result_report_name_empty.success = True
        result_report_name_empty.technical_reports = [
            {
                "report_name": "",
                "report_type": "html",
                "report_path": str(report_file),
            }
        ]

        result_report_name_invalid = TestExecutionResult(
            "test report name invalid",
            "test_report_name_invalid",
            "sut",
            "report-image:latest",
        )
        result_report_name_invalid.success = True
        result_report_name_invalid.technical_reports = [
            {
                "report_name": "invalid",
                "report_type": "html",
                "report_path": str(report_file),
            }
        ]

        errors = validate_test_container_technical_reports(
            [
                result_report_name_missing,
                result_report_name_empty,
                result_report_name_invalid,
            ],
            manifests,
        )

        assert len(errors) == 3
        assert "Report missing valid 'report_name' field" in errors[0]
        assert "Report missing valid 'report_name' field" in errors[1]
        assert "Missing expected reports: [('valid_report', 'html')]" in errors[2]

    def test_report_type_errors(self, tmp_path):
        """
        Test validation fails when report_type is missing, empty or invalid.
        """
        report_file = tmp_path / "report.html"
        report_file.write_text("content")

        manifests = {
            "report-image:latest": Manifest(
                name="test-manifest",
                version="1.0",
                input_systems=[],
                output_reports=[OutputReports(name="valid_report", type="html")],
            )
        }

        result_report_type_missing = TestExecutionResult(
            "test report type missing",
            "test_report_type_missing",
            "sut",
            "report-image:latest",
        )
        result_report_type_missing.success = True
        result_report_type_missing.technical_reports = [
            {"report_name": "valid_report", "report_path": str(report_file)}
        ]

        result_report_type_empty = TestExecutionResult(
            "test report type empty",
            "test_report_type_empty",
            "sut",
            "report-image:latest",
        )
        result_report_type_empty.success = True
        result_report_type_empty.technical_reports = [
            {
                "report_name": "valid_report",
                "report_type": "",
                "report_path": str(report_file),
            }
        ]

        result_report_type_invalid = TestExecutionResult(
            "test report type invalid",
            "test_report_type_invalid",
            "sut",
            "report-image:latest",
        )
        result_report_type_invalid.success = True
        result_report_type_invalid.technical_reports = [
            {
                "report_name": "valid_report",
                "report_type": "invalid",
                "report_path": str(report_file),
            }
        ]

        errors = validate_test_container_technical_reports(
            [
                result_report_type_missing,
                result_report_type_empty,
                result_report_type_invalid,
            ],
            manifests,
        )

        assert len(errors) == 3
        assert (
            "Report name 'valid_report' missing valid 'report_type' field" in errors[0]
        )
        assert (
            "Report name 'valid_report' missing valid 'report_type' field" in errors[1]
        )
        assert "Missing expected reports: [('valid_report', 'html')]" in errors[2]

    def test_report_path_errors(self, tmp_path):
        """
        Test validation fails when report_path is missing or empty.
        """
        report_file = tmp_path / "report.html"
        report_file.write_text("content")

        manifests = {
            "report-image:latest": Manifest(
                name="test-manifest",
                version="1.0",
                input_systems=[],
                output_reports=[OutputReports(name="valid_report", type="html")],
            )
        }

        result_report_path_missing = TestExecutionResult(
            "test report path missing",
            "test_report_path_missing",
            "sut",
            "report-image:latest",
        )
        result_report_path_missing.success = True
        result_report_path_missing.technical_reports = [
            {"report_name": "valid_report", "report_type": "html"}
        ]

        result_report_path_empty = TestExecutionResult(
            "test report path empty",
            "test_report_path_empty",
            "sut",
            "report-image:latest",
        )
        result_report_path_empty.success = True
        result_report_path_empty.technical_reports = [
            {
                "report_name": "valid_report",
                "report_type": "html",
                "report_path": "",
            }
        ]

        errors = validate_test_container_technical_reports(
            [result_report_path_missing, result_report_path_empty], manifests
        )

        assert len(errors) == 2
        assert (
            "Report name 'valid_report' missing valid 'report_path' field" in errors[0]
        )
        assert (
            "Report name 'valid_report' missing valid 'report_path' field" in errors[1]
        )

    def test_invalid_file_error(self, tmp_path):
        """
        Test validation fails when report file does not exist.
        """
        manifests = {}

        result = TestExecutionResult(
            "test report file not exist", "test_not_exist", "sut", "report-image:latest"
        )
        result.success = True
        result.technical_reports = [
            {
                "report_name": "test_report",
                "report_type": "html",
                "report_path": str(tmp_path / "invalid.html"),
            }
        ]

        errors = validate_test_container_technical_reports([result], manifests)

        assert len(errors) == 1
        assert result.success is False
        assert (
            "Report name 'test_report' does not exist at path" in result.error_message
        )

    def test_validate_test_container_technical_reports_success(self, tmp_path):
        """
        Test validation passes when the report returned by the test container matches the manifest definitions
        """
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
        result.technical_reports = [
            {
                "report_name": "valid_report",
                "report_type": "html",
                "report_path": str(report_file_html),
            },
            {
                "report_name": "another_report",
                "report_type": "pdf",
                "report_path": str(report_file_pdf),
            },
        ]
        errors = validate_test_container_technical_reports([result], manifests)
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
        result.technical_reports = [
            {
                "report_name": "summary",
                "report_type": "html",
                "report_path": "report.html",
            }
        ]

        errors = validate_test_container_technical_reports([result], manifests)

        assert len(errors) == 0
        assert result.success is False
        assert result.error_message == "Test failed for other reasons"
