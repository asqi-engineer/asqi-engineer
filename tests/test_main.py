import os
import tempfile
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from asqi.config import ContainerConfig, ExecutionMode, ExecutorConfig
from asqi.main import app, load_score_card_file, load_yaml_file
from test_data import MOCK_AUDIT_RESPONSES, MOCK_SCORE_CARD_CONFIG


class TestMainCLI:
    """Test the main CLI with typer subcommands."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_version_flag(self):
        """Test that --version flag displays version information."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "asqi-engineer version" in result.output
        # Check for either format: simple version or version with build info
        assert "asqi-engineer version" in result.output and (
            "build" in result.output
            or "unknown" in result.output
            or result.output.count("asqi-engineer version") == 1
        )

    def test_version_flag_short(self):
        """Test that -V flag displays version information."""
        result = self.runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "asqi-engineer version" in result.output

    def test_version_flag_package_not_found_error(self):
        """
        Test that --version flag raises an error when asqi-engineer is not installed.
        """
        with patch("asqi.main.version", side_effect=PackageNotFoundError):
            result = self.runner.invoke(app, ["--version"])
            assert result.exit_code == 0
            assert "asqi-engineer version: unknown (not installed)" in result.output

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
            execution_mode=ExecutionMode.TESTS_ONLY,
            test_ids=None,
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
            execution_mode=ExecutionMode.END_TO_END,
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
            audit_responses_data=None,
        )
        assert "✅ Loaded grading score card: Test scorecard" in result.stdout
        assert "✨ Execution completed! Workflow ID: workflow-456" in result.stdout

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.main.load_audit_responses_file")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_execute_with_audit_responses(
        self, mock_dbos, mock_load_score, mock_load_audit, mock_start
    ):
        """Test execute command with score card and audit responses."""
        mock_load_score.return_value = MOCK_SCORE_CARD_CONFIG
        mock_load_audit.return_value = MOCK_AUDIT_RESPONSES
        mock_start.return_value = "workflow-audit-1"

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
                "-a",
                "audit_responses.yaml",
                "-o",
                "output_scorecard.json",
            ],
        )

        assert result.exit_code == 0

        mock_load_score.assert_called_once_with("score_card.yaml")
        mock_load_audit.assert_called_once_with("audit_responses.yaml")

        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="output_scorecard.json",
            score_card_configs=[MOCK_SCORE_CARD_CONFIG],
            execution_mode=ExecutionMode.END_TO_END,
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
            audit_responses_data=MOCK_AUDIT_RESPONSES,
        )

        assert "✅ Loaded grading score card: Mock Chatbot Scorecard" in result.stdout
        assert "✨ Execution completed! Workflow ID: workflow-audit-1" in result.stdout

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_execute_with_skip_audit(self, mock_dbos, mock_load_score, mock_start):
        """Test execute command when audit is explicitly skipped."""
        mock_load_score.return_value = MOCK_SCORE_CARD_CONFIG
        mock_start.return_value = "workflow-audit-skip"

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
                "--skip-audit",
                "-o",
                "output_scorecard.json",
            ],
        )

        assert result.exit_code == 0
        mock_load_score.assert_called_once_with("score_card.yaml")

        # Inspect what was passed into start_test_execution
        mock_start.assert_called_once()
        _, kwargs = mock_start.call_args

        cleaned_configs = kwargs["score_card_configs"]
        assert len(cleaned_configs) == 1
        cleaned_card = cleaned_configs[0]
        # All audit indicators should have been removed
        assert all(
            ind.get("type") != "audit" for ind in cleaned_card.get("indicators", [])
        )
        assert kwargs["audit_responses_data"] is None

        assert (
            "✨ Execution completed! Workflow ID: workflow-audit-skip" in result.stdout
        )

    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_execute_audit_required_but_missing(self, mock_dbos, mock_load_score):
        """Test execute errors when score card has audit indicators but no responses or skip flag."""
        mock_load_score.return_value = MOCK_SCORE_CARD_CONFIG

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
                # no -a and no --skip-audit
            ],
        )

        assert result.exit_code == 1
        out = result.stdout

        assert 'selected_outcome: ""  # Choose from:' in out
        assert 'notes: "Optional explanation"' in out

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
            audit_responses_data=None,
        )
        assert "✅ Loaded grading score card: Test scorecard" in result.stdout
        assert (
            "✨ Score card evaluation completed! Workflow ID: workflow-789"
            in result.stdout
        )

    @patch("asqi.workflow.start_score_card_evaluation")
    @patch("asqi.main.load_audit_responses_file")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_evaluate_score_cards_with_audit_responses(
        self, mock_dbos, mock_load_score, mock_load_audit, mock_start_eval
    ):
        """Test evaluate-score-cards with audit responses."""
        mock_load_score.return_value = MOCK_SCORE_CARD_CONFIG
        mock_load_audit.return_value = MOCK_AUDIT_RESPONSES
        mock_start_eval.return_value = "workflow-audit-eval-1"

        result = self.runner.invoke(
            app,
            [
                "evaluate-score-cards",
                "--input-file",
                "input.json",
                "-r",
                "score_card.yaml",
                "-a",
                "audit_responses.yaml",
                "-o",
                "output_scorecard.json",
            ],
        )

        assert result.exit_code == 0

        mock_load_score.assert_called_once_with("score_card.yaml")
        mock_load_audit.assert_called_once_with("audit_responses.yaml")

        mock_start_eval.assert_called_once_with(
            input_path="input.json",
            score_card_configs=[MOCK_SCORE_CARD_CONFIG],
            audit_responses_data=MOCK_AUDIT_RESPONSES,
            output_path="output_scorecard.json",
        )

        assert "✅ Loaded grading score card: Mock Chatbot Scorecard" in result.stdout
        assert (
            "✨ Score card evaluation completed! Workflow ID: workflow-audit-eval-1"
            in result.stdout
        )

    @patch("asqi.workflow.start_score_card_evaluation")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_evaluate_score_cards_with_skip_audit(
        self, mock_dbos, mock_load_score, mock_start_eval
    ):
        """Test evaluate-score-cards when audit indicators are skipped."""
        mock_load_score.return_value = MOCK_SCORE_CARD_CONFIG
        mock_start_eval.return_value = "workflow-audit-eval-skip"

        result = self.runner.invoke(
            app,
            [
                "evaluate-score-cards",
                "--input-file",
                "input.json",
                "-r",
                "score_card.yaml",
                "--skip-audit",
                "-o",
                "output_scorecard.json",
            ],
        )

        assert result.exit_code == 0
        mock_load_score.assert_called_once_with("score_card.yaml")

        mock_start_eval.assert_called_once()
        _, kwargs = mock_start_eval.call_args

        cleaned_configs = kwargs["score_card_configs"]
        assert len(cleaned_configs) == 1
        cleaned_card = cleaned_configs[0]
        assert all(
            ind.get("type") != "audit" for ind in cleaned_card.get("indicators", [])
        )
        assert kwargs["audit_responses_data"] is None

        assert (
            "✨ Score card evaluation completed! Workflow ID: workflow-audit-eval-skip"
            in result.stdout
        )

    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_evaluate_score_cards_audit_required_but_missing(
        self, mock_dbos, mock_load_score
    ):
        """Test evaluate-score-cards errors when audit indicators exist but no responses or skip flag."""
        mock_load_score.return_value = MOCK_SCORE_CARD_CONFIG

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
        out = result.stdout

        # Same expectations: template must be shown
        assert 'selected_outcome: ""  # Choose from:' in out
        assert 'notes: "Optional explanation"' in out

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

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.workflow.DBOS")
    def test_default_output_filenames(self, mock_dbos, mock_start):
        """Test that default output filenames are logical."""
        mock_start.return_value = "workflow-test"

        result = self.runner.invoke(
            app, ["execute-tests", "-t", "suite.yaml", "-s", "systems.yaml"]
        )
        assert result.exit_code == 0
        mock_start.assert_called_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="output.json",
            score_card_configs=None,
            execution_mode=ExecutionMode.TESTS_ONLY,
            test_ids=None,
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
        )

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_execute_default_scorecard_filename(
        self, mock_dbos, mock_load_score, mock_start
    ):
        """Test execute (with score cards) defaults to output_scorecard.json."""
        mock_load_score.return_value = {"score_card_name": "Test"}
        mock_start.return_value = "workflow-test"

        result = self.runner.invoke(
            app,
            ["execute", "-t", "suite.yaml", "-s", "systems.yaml", "-r", "score.yaml"],
        )
        assert result.exit_code == 0
        mock_start.assert_called_with(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            output_path="output_scorecard.json",
            score_card_configs=[{"score_card_name": "Test"}],
            execution_mode=ExecutionMode.END_TO_END,
            executor_config={
                "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
                "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
                "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
            },
            container_config=ContainerConfig.with_streaming(False),
            audit_responses_data=None,
        )

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
    def test_execute_tests_with_test_ids_success(self, mock_dbos, mock_start):
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
                "-tids",
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
            execution_mode=ExecutionMode.TESTS_ONLY,
            test_ids=["t1"],
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
    def test_execute_tests_with_test_ids_failure(self, mock_dbos, mock_start):
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
                "-tids",
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
            execution_mode=ExecutionMode.TESTS_ONLY,
            test_ids=["tes1"],
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

    def test_load_yaml_file_with_interpolation(self):
        """Test YAML file loading with environment variable interpolation."""
        os.environ["TEST_IMAGE"] = "my-registry.com/test-app"
        os.environ["TEST_API_KEY"] = "sk-12345"

        yaml_content = """
        image: "${TEST_IMAGE}:latest"
        params:
          api_key: "${TEST_API_KEY}"
          timeout: "${TIMEOUT:-30}"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = load_yaml_file(temp_path)
            expected = {
                "image": "my-registry.com/test-app:latest",
                "params": {"api_key": "sk-12345", "timeout": "30"},
            }
            assert result == expected
        finally:
            os.unlink(temp_path)
            del os.environ["TEST_IMAGE"]
            del os.environ["TEST_API_KEY"]

    def test_load_score_card_file_success(self):
        """Test successful score card file loading."""
        score_card_data = {
            "score_card_name": "Test Score Card",
            "indicators": [
                {
                    "id": "test_indicator",
                    "name": "test indicator",
                    "apply_to": {"test_id": "test1"},
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


class TestLoadDotenvFunctionality:
    """Test that load_dotenv is functioning correctly in main.py."""

    @patch("dotenv.load_dotenv")
    def test_load_dotenv_loads_env_variables(self, mock_load_dotenv):
        """Tests that DBOS_DATABASE_URL is available in the environment
        after importing main.py.
        """

        # Mock the .env file content by setting environment variables when load_dotenv is called
        def mock_dotenv_side_effect(*args, **kwargs):
            os.environ["DBOS_DATABASE_URL"] = (
                "postgres://test:test@localhost:5432/test_db"
            )
            return True

        mock_load_dotenv.side_effect = mock_dotenv_side_effect

        # Clear the environment variable first to ensure it's not already set
        original_dbos_url = os.environ.pop("DBOS_DATABASE_URL", None)

        try:
            # Import main.py which should trigger load_dotenv() at module level
            import importlib

            import asqi.main

            importlib.reload(asqi.main)

            # Check if environment variables are now available from mocked .env file
            dbos_url = os.environ.get("DBOS_DATABASE_URL")

            # This assertion will fail if load_dotenv() is commented out
            assert dbos_url is not None, (
                "DBOS_DATABASE_URL should be loaded from .env file"
            )
            assert dbos_url == "postgres://test:test@localhost:5432/test_db"

        finally:
            # Restore original environment state
            if original_dbos_url is not None:
                os.environ["DBOS_DATABASE_URL"] = original_dbos_url
            elif "DBOS_DATABASE_URL" in os.environ:
                del os.environ["DBOS_DATABASE_URL"]
