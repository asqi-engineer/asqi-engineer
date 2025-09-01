import os
import tempfile
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from asqi.config import ContainerConfig, ExecutorConfig
from asqi.main import app, load_score_card_file, load_yaml_file


class TestMainCLI:
    """Test the main CLI with typer subcommands."""

    def setup_method(self):
        self.runner = CliRunner()

    @pytest.mark.parametrize(
        "command,expected_missing",
        [
            # validate command tests
            (["validate"], "Missing option '--test-suite-config'"),
            (
                ["validate", "--test-suite-config", "suite.yaml"],
                "Missing option '--systems-config'",
            ),
            (
                [
                    "validate",
                    "--test-suite-config",
                    "suite.yaml",
                    "--systems-config",
                    "systems.yaml",
                ],
                "Missing option '--manifests-dir'",
            ),
            # execute command tests
            (["execute"], "Missing option '--test-suite-config'"),
            (
                [
                    "execute",
                    "--test-suite-config",
                    "suite.yaml",
                    "--systems-config",
                    "systems.yaml",
                ],
                "Missing option '--score-card-config'",
            ),
            # execute-tests command tests
            (["execute-tests"], "Missing option '--test-suite-config'"),
            (
                ["execute-tests", "--test-suite-config", "suite.yaml"],
                "Missing option '--systems-config'",
            ),
            # evaluate-score-cards command tests
            (["evaluate-score-cards"], "Missing option '--input-file'"),
            (
                ["evaluate-score-cards", "--input-file", "input.json"],
                "Missing option '--score-card-config'",
            ),
        ],
    )
    @pytest.mark.skipif(os.getenv("CI") is not None, reason="ci display issue")
    def test_missing_required_arguments(self, command, expected_missing):
        """Test that all commands require their respective arguments."""
        result = self.runner.invoke(app, command)
        assert result.exit_code == 2
        assert expected_missing in result.output

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.workflow.DBOS")
    def test_execute_tests_success(self, mock_dbos, mock_start):
        """Test successful execute-tests command."""
        mock_start.return_value = "workflow-123"

        result = self.runner.invoke(
            app,
            [
                "execute-tests",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-o",
                "output.json",
            ],
        )

        assert result.exit_code == 0
        mock_dbos.launch.assert_called_once()
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="output.json",
            score_card_configs=None,
            execution_mode="tests_only",
            test_names=None,
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
        )
        assert "✨ Test execution completed! Workflow ID: workflow-123" in result.stdout

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_execute_with_score_card(self, mock_dbos, mock_load_score, mock_start):
        """Test execute command with score card (end-to-end)."""
        mock_load_score.return_value = {"score_card_name": "Test scorecard"}
        mock_start.return_value = "workflow-456"

        result = self.runner.invoke(
            app,
            [
                "execute",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-r",
                "score_card.yaml",
                "-o",
                "output.json",
            ],
        )

        assert result.exit_code == 0
        mock_load_score.assert_called_once_with("score_card.yaml")
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="output.json",
            score_card_configs=[{"score_card_name": "Test scorecard"}],
            execution_mode="end_to_end",
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
        )
        assert "✅ Loaded grading score card: Test scorecard" in result.stdout
        assert "✨ Execution completed! Workflow ID: workflow-456" in result.stdout

    @patch("asqi.workflow.start_score_card_evaluation")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_evaluate_score_cards_success(
        self, mock_dbos, mock_load_score, mock_start_eval
    ):
        """Test successful evaluate-score-cards command."""
        mock_load_score.return_value = {"score_card_name": "Test scorecard"}
        mock_start_eval.return_value = "workflow-789"

        result = self.runner.invoke(
            app,
            [
                "evaluate-score-cards",
                "--input-file",
                "input.json",
                "-r",
                "score_card.yaml",
                "-o",
                "output.json",
            ],
        )

        assert result.exit_code == 0
        mock_load_score.assert_called_once_with("score_card.yaml")
        mock_start_eval.assert_called_once_with(
            input_path="input.json",
            score_card_configs=[{"score_card_name": "Test scorecard"}],
            output_path="output.json",
        )
        assert "✅ Loaded grading score card: Test scorecard" in result.stdout
        assert (
            "✨ Score card evaluation completed! Workflow ID: workflow-789"
            in result.stdout
        )

    @patch("asqi.main.load_and_validate_plan")
    def test_validate_success(self, mock_validate):
        """Test successful validate command."""
        mock_validate.return_value = {"status": "success", "errors": []}

        result = self.runner.invoke(
            app,
            [
                "validate",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "--manifests-dir",
                "manifests/",
            ],
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            manifests_path="manifests/",
        )
        assert "✨ Success! The test plan is valid." in result.stdout
        assert (
            "Use 'execute' or 'execute-tests' commands to run tests." in result.stdout
        )

    @patch("asqi.main.load_and_validate_plan")
    def test_validate_failure(self, mock_validate):
        """Test validate command with errors."""
        mock_validate.return_value = {
            "status": "failure",
            "errors": ["Error 1", "Error 2"],
        }

        result = self.runner.invoke(
            app,
            [
                "validate",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "--manifests-dir",
                "manifests/",
            ],
        )

        assert result.exit_code == 1
        assert "❌ Test Plan Validation Failed:" in result.stdout
        assert "Error 1" in result.stdout
        assert "Error 2" in result.stdout

    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_score_card_config_error(self, mock_dbos, mock_load_score):
        """Test handling score card configuration errors."""
        mock_load_score.side_effect = ValueError("Invalid score card format")

        result = self.runner.invoke(
            app,
            [
                "execute",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-r",
                "bad_score_card.yaml",
            ],
        )

        assert result.exit_code == 1
        assert (
            "❌ score card configuration error: Invalid score card format"
            in result.stdout
        )

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.workflow.DBOS")
    def test_execute_tests_with_test_names_success(self, mock_dbos, mock_start):
        """Test execute-tests succeeds when valid test-names are passed."""
        mock_start.return_value = "workflow-888"

        result = self.runner.invoke(
            app,
            [
                "execute-tests",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-tn",
                "t1",
                "-o",
                "out.json",
            ],
        )

        assert result.exit_code == 0
        mock_dbos.launch.assert_called_once()
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="out.json",
            score_card_configs=None,
            execution_mode="tests_only",
            test_names=["t1"],
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
        )
        assert "✨ Test execution completed! Workflow ID: workflow-888" in result.stdout

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.workflow.DBOS")
    def test_execute_tests_with_test_names_failure(self, mock_dbos, mock_start):
        """Test execute-tests fails when invalid test-names are passed."""
        mock_start.side_effect = ValueError(
            "❌ Test execution failed: ❌ Test not found: tes1\n   Did you mean: test1"
        )

        result = self.runner.invoke(
            app,
            [
                "execute-tests",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-tn",
                "tes1",
                "-o",
                "out.json",
            ],
        )

        assert result.exit_code != 0
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="out.json",
            score_card_configs=None,
            execution_mode="tests_only",
            test_names=["tes1"],
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
        )

        mock_dbos.start_workflow.assert_not_called()
        assert "❌ Test execution failed: ❌ Test not found: tes1" in result.stdout
        assert "Did you mean: test1" in result.stdout


class TestUtilityFunctions:
    """Test utility functions in main.py."""

    def test_load_yaml_file_success(self):
        """Test successful YAML file loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"test": "data", "number": 42}, f)
            temp_path = f.name

        try:
            result = load_yaml_file(temp_path)
            assert result == {"test": "data", "number": 42}
        finally:
            os.unlink(temp_path)

    def test_load_yaml_file_not_found(self):
        """Test YAML file loading with missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_yaml_file("/nonexistent/file.yaml")

    def test_load_yaml_file_invalid_syntax(self):
        """Test YAML file loading with invalid syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML syntax"):
                load_yaml_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_score_card_file_success(self):
        """Test successful score card file loading."""
        score_card_data = {
            "score_card_name": "Test Score Card",
            "indicators": [
                {
                    "name": "test_indicator",
                    "apply_to": {"test_name": "test1"},
                    "metric": "success",
                    "assessment": [
                        {"outcome": "PASS", "condition": "equal_to", "threshold": True}
                    ],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(score_card_data, f)
            temp_path = f.name

        try:
            result = load_score_card_file(temp_path)
            assert result["score_card_name"] == "Test Score Card"
            assert len(result["indicators"]) == 1
        finally:
            os.unlink(temp_path)

    def test_load_score_card_file_invalid_schema(self):
        """Test score card file loading with invalid schema."""
        invalid_data = {"invalid": "schema"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid score card configuration"):
                load_score_card_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_score_card_file_not_found(self):
        """Test score card file loading with missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_score_card_file("/nonexistent/score_card.yaml")


class TestPermissionErrors:
    """Test permission error handling."""

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_load_yaml_file_permission_error(self, mock_open):
        """Test YAML file loading with permission error."""
        with pytest.raises(
            PermissionError, match="Permission denied accessing configuration file"
        ):
            load_yaml_file("restricted_file.yaml")

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_load_score_card_file_permission_error(self, mock_open):
        """Test score card file loading with permission error."""
        with pytest.raises(
            PermissionError, match="Permission denied accessing configuration file"
        ):
            load_score_card_file("restricted_score_card.yaml")


class TestShutdownHandlers:
    """Test signal handling and cleanup functionality."""

    @patch("asqi.main.shutdown_containers")
    def test_handle_shutdown_with_signal(self, mock_shutdown):
        """Test shutdown handler with signal."""
        import signal

        from asqi.main import _handle_shutdown

        _handle_shutdown(signal.SIGINT, None)
        mock_shutdown.assert_called_once()

    @patch("asqi.main.shutdown_containers")
    def test_handle_shutdown_without_signal(self, mock_shutdown):
        """Test shutdown handler without signal."""
        from asqi.main import _handle_shutdown

        _handle_shutdown(None, None)
        mock_shutdown.assert_not_called()


class TestErrorScenarios:
    """Test additional error scenarios."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch("asqi.workflow.DBOS")
    def test_execute_tests_import_error(self, mock_dbos):
        """Test execute-tests with ImportError for DBOS."""
        # Simulate ImportError by removing the import
        with patch.dict("sys.modules", {"asqi.workflow": None}):
            result = self.runner.invoke(
                app,
                [
                    "execute-tests",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                ],
            )
            assert result.exit_code == 1
            assert "DBOS workflow dependencies not available" in result.stdout

    @patch("asqi.workflow.DBOS")
    def test_execute_import_error(self, mock_dbos):
        """Test execute with ImportError for DBOS."""
        with patch.dict("sys.modules", {"asqi.workflow": None}):
            result = self.runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "score_card.yaml",
                ],
            )
            assert result.exit_code == 1
            assert "DBOS workflow dependencies not available" in result.stdout

    @patch("asqi.workflow.DBOS")
    def test_evaluate_score_cards_import_error(self, mock_dbos):
        """Test evaluate-score-cards with ImportError for DBOS."""
        with patch.dict("sys.modules", {"asqi.workflow": None}):
            result = self.runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "score_card.yaml",
                ],
            )
            assert result.exit_code == 1
            assert "DBOS workflow dependencies not available" in result.stdout

    @patch(
        "asqi.workflow.start_test_execution", side_effect=Exception("Workflow error")
    )
    @patch("asqi.workflow.DBOS")
    def test_execute_tests_workflow_error(self, mock_dbos, mock_start):
        """Test execute-tests with workflow execution error."""
        result = self.runner.invoke(
            app,
            [
                "execute-tests",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
            ],
        )
        assert result.exit_code == 1
        assert "Test execution failed: Workflow error" in result.stdout

    @patch("asqi.main.load_score_card_file")
    @patch(
        "asqi.workflow.start_test_execution", side_effect=Exception("Workflow error")
    )
    @patch("asqi.workflow.DBOS")
    def test_execute_workflow_error(self, mock_dbos, mock_start, mock_load_score):
        """Test execute with workflow execution error."""
        mock_load_score.return_value = {"score_card_name": "Test"}

        result = self.runner.invoke(
            app,
            [
                "execute",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-r",
                "score_card.yaml",
            ],
        )
        assert result.exit_code == 1
        assert "Execution failed: Workflow error" in result.stdout

    @patch("asqi.main.load_score_card_file")
    @patch(
        "asqi.workflow.start_score_card_evaluation",
        side_effect=Exception("Evaluation error"),
    )
    @patch("asqi.workflow.DBOS")
    def test_evaluate_score_cards_workflow_error(
        self, mock_dbos, mock_start_eval, mock_load_score
    ):
        """Test evaluate-score-cards with workflow execution error."""
        mock_load_score.return_value = {"score_card_name": "Test"}

        result = self.runner.invoke(
            app,
            [
                "evaluate-score-cards",
                "--input-file",
                "input.json",
                "-r",
                "score_card.yaml",
            ],
        )
        assert result.exit_code == 1
        assert "Score card evaluation failed: Evaluation error" in result.stdout


class TestLoadAndValidatePlan:
    """Test load_and_validate_plan function."""

    def test_load_and_validate_plan_file_errors(self):
        """Test load_and_validate_plan with file errors."""
        from asqi.main import load_and_validate_plan

        # Test with missing suite file
        result = load_and_validate_plan(
            "/nonexistent/suite.yaml",
            "/nonexistent/systems.yaml",
            "/nonexistent/manifests/",
        )
        assert result["status"] == "failure"
        assert any(
            "Configuration file not found" in error for error in result["errors"]
        )

    def test_load_and_validate_plan_success_empty_manifests(self):
        """Test load_and_validate_plan with no manifest files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_file = os.path.join(temp_dir, "suite.yaml")
            systems_file = os.path.join(temp_dir, "systems.yaml")
            manifests_dir = os.path.join(temp_dir, "manifests")
            os.makedirs(manifests_dir)

            # Create minimal valid files
            with open(suite_file, "w") as f:
                yaml.dump({"suite_name": "Empty", "test_suite": []}, f)

            with open(systems_file, "w") as f:
                yaml.dump({"systems": {}}, f)

            from asqi.main import load_and_validate_plan

            result = load_and_validate_plan(suite_file, systems_file, manifests_dir)
            assert result["status"] == "success"

    def test_load_and_validate_plan_with_empty_manifest(self):
        """Test load_and_validate_plan with empty manifest file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite_file = os.path.join(temp_dir, "suite.yaml")
            systems_file = os.path.join(temp_dir, "systems.yaml")
            manifests_dir = os.path.join(temp_dir, "manifests", "test_container")
            os.makedirs(manifests_dir)

            with open(suite_file, "w") as f:
                yaml.dump({"suite_name": "Test", "test_suite": []}, f)

            with open(systems_file, "w") as f:
                yaml.dump({"systems": {}}, f)

            # Create empty manifest file
            manifest_file = os.path.join(manifests_dir, "manifest.yaml")
            with open(manifest_file, "w") as f:
                f.write("")  # Empty file

            from asqi.main import load_and_validate_plan

            result = load_and_validate_plan(suite_file, systems_file, temp_dir)
            # With empty suite and systems, it should succeed without errors
            # (empty manifest is skipped but doesn't cause validation failure)
            assert result["status"] == "success"
