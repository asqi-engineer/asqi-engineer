import json
from unittest.mock import patch

import pytest

from asqi.schemas import Manifest, SUTSupport
from asqi.workflow import (
    TestExecutionResult,
    add_score_cards_to_results,
    convert_test_results_to_objects,
    evaluate_score_cards_workflow,
    execute_single_test,
    run_end_to_end_workflow,
    save_results_to_file_step,
    start_score_card_evaluation,
    start_test_execution,
)
from asqi.workflow import (
    run_test_suite_workflow as _workflow,
)


def _call_inner_workflow(suite_config, suts_config):
    """Call the inner (undecorated) workflow function if available."""
    workflow_fn = getattr(_workflow, "__wrapped__", _workflow)
    return workflow_fn(suite_config, suts_config)


class DummyHandle:
    def __init__(self, result, workflow_id="test-workflow-123"):
        self._result = result
        self._workflow_id = workflow_id

    def get_result(self):
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
            "sutA": {"type": "llm_api", "params": {"endpoint": "http://x"}}
        }
    }

    # Build a minimal manifest that supports the SUT type
    manifest = Manifest(
        name="mock",
        version="1",
        description="",
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
        patch("asqi.workflow.dbos_check_images_availabilty") as mock_avail,
        patch("asqi.workflow.extract_manifest_from_image_step") as mock_extract,
        patch("asqi.workflow.validate_test_plan") as mock_validate,
        patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        patch("asqi.workflow.Queue") as mock_queue_class,
    ):
        mock_avail.return_value = {"test/image:latest": True}
        mock_extract.return_value = manifest
        mock_validate.return_value = []
        mock_plan.return_value = [
            {
                "test_name": "t1_sutA",
                "image": "test/image:latest",
                "sut_name": "sutA",
                "sut_params": {"type": "llm_api", "endpoint": "http://x"},
                "test_params": {"p": "v"},
            }
        ]

        # Enqueue returns a handle with get_result -> success_result
        mock_queue = mock_queue_class.return_value
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

    suts_config = {"systems_under_test": {"sutA": {"type": "llm_api", "params": {}}}}

    with (
        patch("asqi.workflow.dbos_check_images_availabilty") as mock_avail,
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
            sut_params={"type": "llm_api"},
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
            sut_params={"type": "llm_api"},
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
            sut_params={"type": "llm_api"},
            test_params={},
        )

    assert result.success is False
    assert "Failed to parse JSON output" in result.error_message


def test_convert_test_results_to_objects():
    """Test converting test results data back to TestExecutionResult objects."""

    test_results_data = {
        "results": [
            {
                "metadata": {
                    "test_name": "test1",
                    "sut_name": "sut1",
                    "image": "test/image:latest",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "success": True,
                    "container_id": "abc123",
                    "exit_code": 0,
                },
                "test_results": {"success": True, "score": 0.9},
                "error_message": "",
                "container_output": '{"success": true}',
            }
        ]
    }

    inner_step = getattr(
        convert_test_results_to_objects, "__wrapped__", convert_test_results_to_objects
    )
    results = inner_step(test_results_data)

    assert len(results) == 1
    result = results[0]
    assert result.test_name == "test1"
    assert result.sut_name == "sut1"
    assert result.image == "test/image:latest"
    assert result.start_time == 1.0
    assert result.end_time == 2.0
    assert result.success is True
    assert result.container_id == "abc123"
    assert result.exit_code == 0
    assert result.test_results == {"success": True, "score": 0.9}


def test_add_score_cards_to_results():
    """Test adding score card evaluation results to test results data."""

    test_results_data = {"summary": {"status": "COMPLETED"}, "results": []}

    score_card_evaluation = [
        {
            "indicator_name": "Test success",
            "test_name": "test1",
            "sut_name": "sut1",
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
            "indicator_name": "Test 1",
            "outcome": "PASS",
            "score_card_name": "Scorecard A",
        },
        {
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
                    "test_name": "test1",
                    "sut_name": "sut1",
                    "image": "test/image:latest",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "success": True,
                    "container_id": "abc123",
                    "exit_code": 0,
                },
                "test_results": {"success": True},
                "error_message": "",
                "container_output": "",
            }
        ],
    }

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
        _result = inner_workflow(test_results_data, score_card_configs)

        mock_convert.assert_called_once_with(test_results_data)
        mock_evaluate.assert_called_once()
        mock_add.assert_called_once()
        mock_console.print.assert_called_once()


def test_run_end_to_end_workflow():
    """Test the run_end_to_end_workflow function."""

    suite_config = {"suite_name": "test"}
    suts_config = {"systems_under_test": {}}
    score_card_configs = [{"score_card_name": "test"}]
    concurrent_tests = 3

    test_results = {"summary": {"status": "COMPLETED"}, "results": []}
    final_results = {
        "summary": {"status": "COMPLETED"},
        "results": [],
        "score_card": {},
    }

    with (
        patch("asqi.workflow.run_test_suite_workflow") as mock_test_workflow,
        patch("asqi.workflow.evaluate_score_cards_workflow") as mock_score_workflow,
    ):
        mock_test_workflow.return_value = test_results
        mock_score_workflow.return_value = final_results

        inner_workflow = getattr(
            run_end_to_end_workflow, "__wrapped__", run_end_to_end_workflow
        )
        result = inner_workflow(suite_config, suts_config, score_card_configs)

        mock_test_workflow.assert_called_once_with(suite_config, suts_config)
        mock_score_workflow.assert_called_once_with(test_results, score_card_configs)
        assert result == final_results


def test_start_test_execution_tests_only_mode():
    """Test start_test_execution with tests_only mode."""

    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}})

    with (
        patch("asqi.workflow.load_config_file") as mock_load,
        patch("asqi.workflow.DBOS.start_workflow") as mock_start,
    ):
        mock_load.return_value = {"test": "config"}
        mock_start.return_value = mock_handle

        workflow_id = start_test_execution(
            "suite.yaml", "suts.yaml", "output.json", None, "tests_only"
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call run_test_suite_workflow for tests_only mode
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "run_test_suite_workflow"


def test_start_test_execution_end_to_end_mode():
    """Test start_test_execution with end_to_end mode."""

    mock_handle = DummyHandle({"summary": {"status": "COMPLETED"}})
    score_card_configs = [{"score_card_name": "test"}]

    with (
        patch("asqi.workflow.load_config_file") as mock_load,
        patch("asqi.workflow.DBOS.start_workflow") as mock_start,
    ):
        mock_load.return_value = {"test": "config"}
        mock_start.return_value = mock_handle

        workflow_id = start_test_execution(
            "suite.yaml", "suts.yaml", "output.json", score_card_configs, "end_to_end"
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call run_end_to_end_workflow for end_to_end mode with score cards
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "run_end_to_end_workflow"


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
            str(input_json), score_card_configs, str(output_json)
        )

        assert workflow_id == mock_handle.get_workflow_id()
        mock_start.assert_called_once()
        # Should call evaluate_score_cards_workflow
        call_args = mock_start.call_args[0]
        assert call_args[0].__name__ == "evaluate_score_cards_workflow"
