from unittest.mock import patch

from typer.testing import CliRunner

from asqi.main import app


class TestMainCLI:
    """Test the main CLI with typer subcommands."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_validate_missing_suite_file(self):
        """Test that validate command requires suite file."""
        result = self.runner.invoke(app, ["validate"])
        assert result.exit_code == 2
        assert "Missing option '--suite-file'" in result.output

    def test_validate_missing_suts_file(self):
        """Test that validate command requires suts file."""
        result = self.runner.invoke(app, ["validate", "--suite-file", "suite.yaml"])
        assert result.exit_code == 2
        assert "Missing option '--suts-file'" in result.output

    def test_validate_missing_manifests_dir(self):
        """Test that validate command requires manifests dir."""
        result = self.runner.invoke(
            app, ["validate", "--suite-file", "suite.yaml", "--suts-file", "suts.yaml"]
        )
        assert result.exit_code == 2
        assert "Missing option '--manifests-dir'" in result.output

    def test_execute_missing_suite_file(self):
        """Test that execute command requires suite file."""
        result = self.runner.invoke(app, ["execute"])
        assert result.exit_code == 2
        assert "Missing option '--suite-file'" in result.output

    def test_execute_missing_score_card_file(self):
        """Test that execute command requires score card file."""
        result = self.runner.invoke(
            app, ["execute", "--suite-file", "suite.yaml", "--suts-file", "suts.yaml"]
        )
        assert result.exit_code == 2
        assert "Missing option '--score-card-file'" in result.output

    def test_execute_tests_missing_suite_file(self):
        """Test that execute-tests command requires suite file."""
        result = self.runner.invoke(app, ["execute-tests"])
        assert result.exit_code == 2
        assert "Missing option '--suite-file'" in result.output

    def test_execute_tests_missing_suts_file(self):
        """Test that execute-tests command requires suts file."""
        result = self.runner.invoke(
            app, ["execute-tests", "--suite-file", "suite.yaml"]
        )
        assert result.exit_code == 2
        assert "Missing option '--suts-file'" in result.output

    def test_evaluate_score_cards_missing_input_file(self):
        """Test that evaluate-score-cards command requires input file."""
        result = self.runner.invoke(app, ["evaluate-score-cards"])
        assert result.exit_code == 2
        assert "Missing option '--input-file'" in result.output

    def test_evaluate_score_cards_missing_score_card_file(self):
        """Test that evaluate-score-cards command requires score card file."""
        result = self.runner.invoke(
            app, ["evaluate-score-cards", "--input-file", "input.json"]
        )
        assert result.exit_code == 2
        assert "Missing option '--score-card-file'" in result.output

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.workflow.DBOS")
    def test_execute_tests_success(self, mock_dbos, mock_start):
        """Test successful execute-tests command."""
        mock_start.return_value = "workflow-123"

        result = self.runner.invoke(
            app,
            [
                "execute-tests",
                "--suite-file",
                "suite.yaml",
                "--suts-file",
                "suts.yaml",
                "--output-file",
                "output.json",
            ],
        )

        assert result.exit_code == 0
        mock_dbos.launch.assert_called_once()
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            suts_path="suts.yaml",
            output_path="output.json",
            score_card_configs=None,
            execution_mode="tests_only",
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
                "--suite-file",
                "suite.yaml",
                "--suts-file",
                "suts.yaml",
                "--score-card-file",
                "score_card.yaml",
                "--output-file",
                "output.json",
            ],
        )

        assert result.exit_code == 0
        mock_load_score.assert_called_once_with("score_card.yaml")
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            suts_path="suts.yaml",
            output_path="output.json",
            score_card_configs=[{"score_card_name": "Test scorecard"}],
            execution_mode="end_to_end",
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
                "--score-card-file",
                "score_card.yaml",
                "--output-file",
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
                "--suite-file",
                "suite.yaml",
                "--suts-file",
                "suts.yaml",
                "--manifests-dir",
                "manifests/",
            ],
        )

        assert result.exit_code == 0
        mock_validate.assert_called_once_with(
            suite_path="suite.yaml", suts_path="suts.yaml", manifests_path="manifests/"
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
                "--suite-file",
                "suite.yaml",
                "--suts-file",
                "suts.yaml",
                "--manifests-dir",
                "manifests/",
            ],
        )

        assert result.exit_code == 1
        assert "❌ Test Plan Validation Failed:" in result.stderr
        assert "Error 1" in result.stderr
        assert "Error 2" in result.stderr

    @patch("asqi.workflow.start_test_execution")
    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_execute_tests_with_optional_score_card(
        self, mock_dbos, mock_load_score, mock_start
    ):
        """Test execute-tests command with optional score card."""
        mock_load_score.return_value = {"score_card_name": "Test scorecard"}
        mock_start.return_value = "workflow-123"

        result = self.runner.invoke(
            app,
            [
                "execute-tests",
                "--suite-file",
                "suite.yaml",
                "--suts-file",
                "suts.yaml",
                "--score-card-file",
                "score_card.yaml",
            ],
        )

        assert result.exit_code == 0
        mock_load_score.assert_called_once_with("score_card.yaml")
        mock_start.assert_called_once_with(
            suite_path="suite.yaml",
            suts_path="suts.yaml",
            output_path=None,
            score_card_configs=[{"score_card_name": "Test scorecard"}],
            execution_mode="tests_only",
        )
        assert "✅ Loaded grading score card: Test scorecard" in result.stdout

    @patch("asqi.main.load_score_card_file")
    @patch("asqi.workflow.DBOS")
    def test_score_card_config_error(self, mock_dbos, mock_load_score):
        """Test handling score card configuration errors."""
        from asqi.main import ConfigError

        mock_load_score.side_effect = ConfigError("Invalid score card format")

        result = self.runner.invoke(
            app,
            [
                "execute",
                "--suite-file",
                "suite.yaml",
                "--suts-file",
                "suts.yaml",
                "--score-card-file",
                "bad_score_card.yaml",
            ],
        )

        assert result.exit_code == 1
        assert (
            "❌ score card configuration error: Invalid score card format"
            in result.stderr
        )
