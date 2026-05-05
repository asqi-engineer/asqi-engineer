# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""Contract tests for the CLI surface.

Covers each subcommand's documented happy-path behaviour, exit codes, error
mapping, and the cross-subcommand global ``--version`` flag.

We use Typer's CliRunner so we exercise the real argument parsing layer.
The downstream workflow / start_* helpers are mocked because the contract
under test is the CLI shape (flags, exit codes, error messages), not the
workflow itself."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest
import yaml
from asqi.config import ContainerConfig, ExecutionMode, ExecutorConfig
from asqi.errors import AuditResponsesRequiredError
from asqi.main import app
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_yaml(path, data) -> None:
    path.write_text(yaml.safe_dump(data))


# ---------------------------------------------------------------------------
# Global --version flag
# ---------------------------------------------------------------------------


class TestGlobalVersionFlag:
    def test_version_long_flag_exits_zero_with_version_text(self, runner):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "asqi-engineer version" in result.output

    def test_version_short_flag_exits_zero(self, runner):
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "asqi-engineer version" in result.output

    def test_version_when_package_not_installed(self, runner):
        with patch("asqi.main.version", side_effect=PackageNotFoundError):
            result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "asqi-engineer version: unknown (not installed)" in result.output

    def test_version_with_local_build_suffix_includes_git_hash(self, runner):
        """Per the documented global-flag contract: when the installed
        version carries a ``+local`` build suffix
        (e.g. ``0.3.1.dev2+g816449b60.d20251120``) the output is
        ``asqi-engineer version <ver>, build <git-hash>``.

        The ``version_callback`` splits on the first ``+``, then on the
        first ``.`` of the build component to extract the git hash.
        """
        with patch("asqi.main.version", return_value="0.3.1.dev2+g816449b60.d20251120"):
            result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Both the version and the build hash appear in the documented format.
        assert "asqi-engineer version 0.3.1.dev2" in result.output
        assert "build g816449b60" in result.output
        # The trailing ``.d20251120`` chunk is stripped and not surfaced.
        assert "d20251120" not in result.output

    def test_version_with_build_suffix_no_dot_uses_full_build(self, runner):
        """When the ``+`` suffix has no internal ``.``, the entire build
        component is used as the hash (``"." in build`` is False branch)."""
        with patch("asqi.main.version", return_value="0.3.1+g123abc"):
            result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "asqi-engineer version 0.3.1, build g123abc" in result.output


# ---------------------------------------------------------------------------
# ``asqi validate``
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_happy_path_exits_zero_for_valid_plan(self, runner, tmp_path):
        suite = tmp_path / "suite.yaml"
        systems = tmp_path / "systems.yaml"
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        _write_yaml(
            suite,
            {
                "suite_name": "demo",
                "test_suite": [
                    {
                        "name": "t1",
                        "id": "t1",
                        "image": "test/image:latest",
                        "systems_under_test": ["sysA"],
                        "params": {},
                    }
                ],
            },
        )
        _write_yaml(
            systems,
            {
                "systems": {
                    "sysA": {
                        "type": "llm_api",
                        "params": {"base_url": "http://x", "model": "x"},
                    }
                }
            },
        )

        with patch(
            "asqi.main.load_and_validate_plan",
            return_value={"status": "success", "errors": []},
        ):
            result = runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    str(suite),
                    "-s",
                    str(systems),
                    "--manifests-dir",
                    str(manifests_dir),
                ],
            )

        assert result.exit_code == 0
        # Documented stdout markers
        assert "Running Verification" in result.output
        # '[green]✅ IDs verified[/green]' is the documented uniqueness-pass
        # banner emitted before plan validation runs.
        assert "IDs verified" in result.output
        assert "Success" in result.output

    def test_failure_exits_one_and_shows_errors(self, runner, tmp_path):
        suite = tmp_path / "suite.yaml"
        systems = tmp_path / "systems.yaml"
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        _write_yaml(suite, {"suite_name": "x", "test_suite": []})
        _write_yaml(systems, {"systems": {}})

        with patch(
            "asqi.main.load_and_validate_plan",
            return_value={"status": "failure", "errors": ["bad-thing"]},
        ):
            result = runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    str(suite),
                    "-s",
                    str(systems),
                    "--manifests-dir",
                    str(manifests_dir),
                ],
            )

        assert result.exit_code == 1
        # Documented stdout markers
        assert "Test Plan Validation Failed" in result.output
        assert "bad-thing" in result.output

    def test_missing_required_args_exits_two(self, runner):
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 2

    def test_duplicate_id_error_exits_nonzero(self, runner, tmp_path):
        """Documented exit code: ``1`` for any validation failure including
        ``duplicate IDs``. ``_validate_unique_ids`` re-raises
        ``DuplicateIDError`` which Typer surfaces as a non-zero exit (the
        documented ``1`` family — Typer/Click reports it via the SystemExit
        code path).

        Pin per-cause: a ``DuplicateIDError`` from the uniqueness check must
        terminate the command with a non-zero exit and NOT emit the success
        banner. A regression that swallows the uniqueness check (or routes it
        through exit 0) would slip past ``test_failure_exits_one_and_shows_errors``
        because that test only mocks ``load_and_validate_plan``."""
        from asqi.errors import DuplicateIDError

        suite = tmp_path / "suite.yaml"
        systems = tmp_path / "systems.yaml"
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        _write_yaml(suite, {"suite_name": "x", "test_suite": []})
        _write_yaml(systems, {"systems": {}})

        with patch(
            "asqi.main.validate_ids",
            side_effect=DuplicateIDError(
                {
                    "dup": {
                        "id": "dup",
                        "config_type": "test_suite",
                        "occurrences": [
                            {
                                "location": str(suite),
                                "test_suite_name": "x",
                                "test_name": "t",
                            }
                        ],
                    }
                }
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    str(suite),
                    "-s",
                    str(systems),
                    "--manifests-dir",
                    str(manifests_dir),
                ],
            )

        assert result.exit_code != 0
        # The success banner must NOT appear when the uniqueness pass failed.
        assert "Success" not in result.output
        # And the documented duplicate-ID banner is rendered first.
        assert "Found Duplicated IDs" in result.output

    def test_missing_id_field_error_exits_nonzero(self, runner, tmp_path):
        """``MissingIDFieldError`` from the uniqueness pass also terminates
        ``validate`` with a non-zero exit. Pin the second documented cause
        separately from ``DuplicateIDError`` so a regression that catches one
        but not the other surfaces."""
        from asqi.errors import MissingIDFieldError

        suite = tmp_path / "suite.yaml"
        systems = tmp_path / "systems.yaml"
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        _write_yaml(suite, {"suite_name": "x", "test_suite": []})
        _write_yaml(systems, {"systems": {}})

        with patch(
            "asqi.main.validate_ids",
            side_effect=MissingIDFieldError("missing id field in test"),
        ):
            result = runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    str(suite),
                    "-s",
                    str(systems),
                    "--manifests-dir",
                    str(manifests_dir),
                ],
            )

        assert result.exit_code != 0
        assert "Success" not in result.output
        # Documented banner from `_validate_unique_ids` (main.py:303)
        assert "Missing required ID field" in result.output

    def test_failure_paths_emit_documented_error_prefix(self, runner, tmp_path):
        """Documented stdout: each error line is rendered as ``  - <error>``
        (preceded by the red ``❌ Test Plan Validation Failed:`` header).

        Pin the per-line prefix so a regression that drops the bullet
        rendering (or merges all errors onto one line) is caught — the
        documented stdout format is part of the contract for tools that
        scrape the validate output."""
        suite = tmp_path / "suite.yaml"
        systems = tmp_path / "systems.yaml"
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        _write_yaml(suite, {"suite_name": "x", "test_suite": []})
        _write_yaml(systems, {"systems": {}})

        with patch(
            "asqi.main.load_and_validate_plan",
            return_value={
                "status": "failure",
                "errors": ["error-one", "error-two"],
            },
        ):
            result = runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    str(suite),
                    "-s",
                    str(systems),
                    "--manifests-dir",
                    str(manifests_dir),
                ],
            )

        assert result.exit_code == 1
        # Documented per-error prefix and content.
        assert "- error-one" in result.output
        assert "- error-two" in result.output


# ---------------------------------------------------------------------------
# ``asqi execute``
# ---------------------------------------------------------------------------


class TestExecuteCommand:
    def test_happy_path_invokes_start_test_execution_end_to_end(self, runner):
        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {"score_card_name": "Test"}
            mock_start.return_value = "wf-123"
            result = runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "card.yaml",
                ],
            )

        assert result.exit_code == 0
        mock_start.assert_called_once()
        kwargs = mock_start.call_args.kwargs
        assert kwargs["execution_mode"] == ExecutionMode.END_TO_END
        assert kwargs["score_card_configs"] == [{"score_card_name": "Test"}]
        # Documented default output_file is "output_scorecard.json"
        assert kwargs["output_path"] == "output_scorecard.json"
        # Default executor_config from ExecutorConfig
        assert kwargs["executor_config"] == {
            "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
            "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
            "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
        }
        # Audit kwargs default to None when no audit indicators in card
        assert kwargs["audit_responses_data"] is None

    def test_audit_responses_required_error_exits_one(self, runner):
        """If the score card has audit indicators and neither --audit-responses
        nor --skip-audit is given, ``AuditResponsesRequiredError`` is raised
        by ``resolve_audit_options`` and the CLI exits 1 (documented audit
        logic)."""
        with (
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch(
                "asqi.main.resolve_audit_options",
                side_effect=AuditResponsesRequiredError(score_card_name="X", audit_indicators=[{"id": "a1"}]),
            ),
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {
                "score_card_name": "X",
                "indicators": [{"id": "a1", "type": "audit"}],
            }
            result = runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "card.yaml",
                ],
            )

        assert result.exit_code == 1

    def test_score_card_load_error_exits_one(self, runner):
        """A FileNotFoundError on the score card produces exit 1."""
        with (
            patch(
                "asqi.main.load_score_card_file",
                side_effect=FileNotFoundError("no such card"),
            ),
            patch("asqi.workflow.DBOS"),
        ):
            result = runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "card.yaml",
                ],
            )
        assert result.exit_code == 1

    def test_missing_required_arg_exits_two(self, runner):
        result = runner.invoke(app, ["execute", "-t", "suite.yaml"])
        assert result.exit_code == 2

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--concurrent-tests", "0"),
            ("--concurrent-tests", "21"),
            ("--max-failures", "0"),
            ("--max-failures", "11"),
            ("--progress-interval", "0"),
            ("--progress-interval", "11"),
        ],
    )
    def test_range_violations_exit_two(self, runner, flag, value):
        """The int flags are range-clamped (1..20, 1..10, 1..10) per the
        documented contract."""
        result = runner.invoke(
            app,
            [
                "execute",
                "-t",
                "suite.yaml",
                "-s",
                "systems.yaml",
                "-r",
                "card.yaml",
                flag,
                value,
            ],
        )
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# ``asqi execute-tests``
# ---------------------------------------------------------------------------


class TestExecuteTestsCommand:
    def test_happy_path_invokes_start_test_execution_tests_only(self, runner):
        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.workflow.DBOS"),
        ):
            mock_start.return_value = "wf-456"
            result = runner.invoke(
                app,
                [
                    "execute-tests",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock_start.call_args.kwargs
        assert kwargs["execution_mode"] == ExecutionMode.TESTS_ONLY
        assert kwargs["score_card_configs"] is None
        # Documented default output_file is "output.json" (different from `execute`)
        assert kwargs["output_path"] == "output.json"

    def test_test_ids_repeated_flag_collected_as_list(self, runner):
        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.workflow.DBOS"),
        ):
            mock_start.return_value = "wf"
            result = runner.invoke(
                app,
                [
                    "execute-tests",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-tids",
                    "alpha",
                    "-tids",
                    "beta",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock_start.call_args.kwargs
        assert kwargs["test_ids"] == ["alpha", "beta"]

    def test_test_ids_comma_separated_collected_as_list(self, runner):
        """``-tids a,b`` (single flag, comma-separated) is supported.

        Note: the comma-splitting itself happens deeper, inside
        ``start_test_execution``. From the CLI layer, Typer hands the raw
        value through unchanged."""
        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.workflow.DBOS"),
        ):
            mock_start.return_value = "wf"
            result = runner.invoke(
                app,
                [
                    "execute-tests",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-tids",
                    "alpha,beta",
                ],
            )
        assert result.exit_code == 0
        kwargs = mock_start.call_args.kwargs
        assert kwargs["test_ids"] == ["alpha,beta"]

    def test_workflow_dependency_import_error_exits_one(self, runner):
        """Documented exit code: ImportError on DBOS deps -> exit 1."""
        # Patch the local import target inside execute_tests
        import asqi.workflow as workflow_module

        with patch.object(workflow_module, "start_test_execution", side_effect=ImportError):
            result = runner.invoke(
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

    def test_missing_required_arg_exits_two(self, runner):
        result = runner.invoke(app, ["execute-tests"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# ``--test-ids`` matches the ``name`` field, not ``id``
# ---------------------------------------------------------------------------


class TestTestIdsFilterSemantics:
    """Despite the flag name, values are matched against each test's ``name``
    field (case-insensitive), not its ``id`` field.

    These tests target ``start_test_execution`` directly because that is
    where the filter actually executes (workflow.py:1680). The CLI layer
    just hands the raw flag values through."""

    def _suite_yaml(self, tmp_path) -> str:
        """Suite with two tests where id != name on purpose."""
        path = tmp_path / "suite.yaml"
        _write_yaml(
            path,
            {
                "suite_name": "demo",
                "test_suite": [
                    {
                        "id": "id_alpha",
                        "name": "name_alpha",
                        "image": "img:1",
                        "systems_under_test": ["sysA"],
                        "params": {},
                    },
                    {
                        "id": "id_beta",
                        "name": "name_beta",
                        "image": "img:1",
                        "systems_under_test": ["sysA"],
                        "params": {},
                    },
                ],
            },
        )
        return str(path)

    def _systems_yaml(self, tmp_path) -> str:
        path = tmp_path / "systems.yaml"
        _write_yaml(
            path,
            {
                "systems": {
                    "sysA": {
                        "type": "llm_api",
                        "params": {"base_url": "http://x", "model": "m"},
                    }
                }
            },
        )
        return str(path)

    def _call_start(self, suite_path, systems_path, test_ids):
        """Invoke start_test_execution with mocks deep enough to capture the
        filtered suite_config before any real DBOS work happens."""
        from asqi.config import ContainerConfig
        from asqi.workflow import start_test_execution

        captured: dict = {}

        def fake_start_workflow(fn, suite, *args, **kwargs):
            captured["suite_config"] = suite

            class _Handle:
                def get_workflow_id(self):
                    return "wf"

                def get_result(self):
                    # Documented return shape of run_test_suite_workflow.
                    return ({"summary": {}, "results": []}, [])

            return _Handle()

        with (
            patch("asqi.workflow.DBOS") as mock_dbos,
            patch("asqi.workflow.validate_execution_inputs", return_value=None),
        ):
            mock_dbos.start_workflow.side_effect = fake_start_workflow
            mock_dbos.workflow_id = "wf"
            start_test_execution(
                suite_path=suite_path,
                systems_path=systems_path,
                executor_config={
                    "concurrent_tests": 1,
                    "max_failures": 1,
                    "progress_interval": 1,
                },
                container_config=ContainerConfig(),
                execution_mode=ExecutionMode.TESTS_ONLY,
                test_ids=test_ids,
            )
        return captured["suite_config"]

    def test_filter_matches_name_field_not_id_field(self, tmp_path):
        """Pass the test's ``name`` (not its ``id``) — the test should pass through."""
        suite = self._call_start(
            self._suite_yaml(tmp_path),
            self._systems_yaml(tmp_path),
            test_ids=["name_alpha"],
        )
        assert [t["name"] for t in suite["test_suite"]] == ["name_alpha"]

    def test_filter_using_id_value_does_not_match(self, tmp_path):
        """Using the ``id`` value (which differs from name) should miss and
        raise ``ValueError`` — the documented "did you mean" path."""
        with pytest.raises(ValueError) as excinfo:
            self._call_start(
                self._suite_yaml(tmp_path),
                self._systems_yaml(tmp_path),
                test_ids=["id_alpha"],
            )
        # Documented "did you mean" suggestion (workflow.py:1702-1707)
        msg = str(excinfo.value)
        assert "Test not found" in msg
        # difflib should suggest one of the actual names
        assert "name_alpha" in msg or "Did you mean" in msg

    def test_filter_is_case_insensitive(self, tmp_path):
        """Filter is documented as case-insensitive."""
        suite = self._call_start(
            self._suite_yaml(tmp_path),
            self._systems_yaml(tmp_path),
            test_ids=["NAME_ALPHA"],
        )
        assert [t["name"] for t in suite["test_suite"]] == ["name_alpha"]

    def test_filter_supports_comma_separated_single_flag(self, tmp_path):
        """Comma-separated values (-tids a,b) are supported."""
        suite = self._call_start(
            self._suite_yaml(tmp_path),
            self._systems_yaml(tmp_path),
            test_ids=["name_alpha,name_beta"],
        )
        names = sorted(t["name"] for t in suite["test_suite"])
        assert names == ["name_alpha", "name_beta"]

    def test_unknown_name_raises_value_error_with_suggestion(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            self._call_start(
                self._suite_yaml(tmp_path),
                self._systems_yaml(tmp_path),
                test_ids=["name_alphaa"],  # typo
            )
        msg = str(excinfo.value)
        assert "Test not found" in msg
        assert "Did you mean" in msg


# ---------------------------------------------------------------------------
# ``asqi generate-dataset``
# ---------------------------------------------------------------------------


class TestGenerateDatasetCommand:
    def test_happy_path_invokes_start_data_generation(self, runner):
        with (
            patch("asqi.workflow.start_data_generation") as mock_start,
            patch("asqi.workflow.DBOS"),
        ):
            mock_start.return_value = "wf-gen"
            result = runner.invoke(
                app,
                [
                    "generate-dataset",
                    "-t",
                    "gen.yaml",
                ],
            )
        assert result.exit_code == 0
        mock_start.assert_called_once()
        kwargs = mock_start.call_args.kwargs
        # systems_path is optional; default None
        assert kwargs["systems_path"] is None
        # output_path default is "output.json"
        assert kwargs["output_path"] == "output.json"
        assert isinstance(kwargs["container_config"], ContainerConfig)

    def test_missing_required_arg_exits_two(self, runner):
        result = runner.invoke(app, ["generate-dataset"])
        assert result.exit_code == 2

    def test_data_generation_failure_exits_one(self, runner):
        with (
            patch(
                "asqi.workflow.start_data_generation",
                side_effect=RuntimeError("boom"),
            ),
            patch("asqi.workflow.DBOS"),
        ):
            result = runner.invoke(
                app,
                [
                    "generate-dataset",
                    "-t",
                    "gen.yaml",
                ],
            )
        assert result.exit_code == 1

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--concurrent-tests", "0"),
            ("--concurrent-tests", "21"),
            ("--max-failures", "0"),
            ("--max-failures", "11"),
            ("--progress-interval", "0"),
            ("--progress-interval", "11"),
        ],
    )
    def test_range_violations_exit_two(self, runner, flag, value):
        """The int flags are range-clamped (1..20, 1..10, 1..10) — same
        contract as ``execute``."""
        result = runner.invoke(
            app,
            ["generate-dataset", "-t", "gen.yaml", flag, value],
        )
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# ``asqi evaluate-score-cards``
# ---------------------------------------------------------------------------


class TestEvaluateScoreCardsCommand:
    def test_happy_path_invokes_start_score_card_evaluation(self, runner):
        with (
            patch("asqi.workflow.start_score_card_evaluation") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {"score_card_name": "Test"}
            mock_start.return_value = "wf-eval"
            result = runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                ],
            )
        assert result.exit_code == 0
        mock_start.assert_called_once()

    def test_missing_required_arg_exits_two(self, runner):
        result = runner.invoke(app, ["evaluate-score-cards"])
        assert result.exit_code == 2

    def test_score_card_load_error_exits_one(self, runner):
        """Documented exit code: a score-card load failure
        (FileNotFoundError / ValueError / PermissionError) maps to exit 1."""
        with (
            patch(
                "asqi.main.load_score_card_file",
                side_effect=FileNotFoundError("no such card"),
            ),
            patch("asqi.workflow.DBOS"),
        ):
            result = runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                ],
            )
        assert result.exit_code == 1

    def test_dbos_import_error_exits_one(self, runner):
        """Documented exit code: ``ImportError`` on DBOS deps maps to exit 1.

        The DBOS module is imported lazily inside the command body. Setting
        ``sys.modules["asqi.workflow"] = None`` makes the next import of
        that module raise ``ImportError`` (Python's documented sentinel for
        "this module is intentionally unavailable")."""
        import sys

        with patch.dict(sys.modules, {"asqi.workflow": None}):
            result = runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                ],
            )
        assert result.exit_code == 1
        # Confirm the ImportError branch fired, not some other exit-1 path.
        assert "DBOS workflow dependencies not available" in result.output

    def test_audit_responses_required_error_exits_one(self, runner):
        """Same audit logic as ``execute`` — when the score card has audit
        indicators and neither ``--audit-responses`` nor ``--skip-audit`` is
        supplied, ``AuditResponsesRequiredError`` exits 1."""
        with (
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch(
                "asqi.main.resolve_audit_options",
                side_effect=AuditResponsesRequiredError(score_card_name="X", audit_indicators=[{"id": "a1"}]),
            ),
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {
                "score_card_name": "X",
                "indicators": [{"id": "a1", "type": "audit"}],
            }
            result = runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                ],
            )
        assert result.exit_code == 1

    def test_default_output_file_is_output_scorecard_json(self, runner):
        """The documented default --output-file is ``output_scorecard.json``."""
        with (
            patch("asqi.workflow.start_score_card_evaluation") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {"score_card_name": "Test"}
            mock_start.return_value = "ok"
            runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                ],
            )
        kwargs = mock_start.call_args.kwargs
        assert kwargs.get("output_path") == "output_scorecard.json"


# ---------------------------------------------------------------------------
# ``--container-config`` and ``--datasets-config`` flag forwarding
# ---------------------------------------------------------------------------


class TestContainerAndDatasetsConfigFlags:
    """``--container-config`` (when given) is loaded via
    ``ContainerConfig.load_from_yaml`` and the resulting instance is forwarded
    to the workflow start helper. ``--datasets-config`` is forwarded as a path
    (under the ``datasets_config_path`` kwarg)."""

    def test_execute_loads_container_config_via_load_from_yaml(self, runner, tmp_path):
        cfg = tmp_path / "container.yaml"
        cfg.write_text("timeout_seconds: 999\nstream_logs: true\n")

        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {"score_card_name": "Test"}
            mock_start.return_value = "wf"
            result = runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "card.yaml",
                    "--container-config",
                    str(cfg),
                    "-d",
                    "datasets.yaml",
                ],
            )

        assert result.exit_code == 0
        kwargs = mock_start.call_args.kwargs
        # ContainerConfig instance loaded from the YAML
        assert isinstance(kwargs["container_config"], ContainerConfig)
        assert kwargs["container_config"].timeout_seconds == 999
        assert kwargs["container_config"].stream_logs is True
        # --datasets-config forwarded as a path string
        assert kwargs["datasets_config_path"] == "datasets.yaml"

    def test_execute_default_container_config_when_flag_omitted(self, runner):
        """``--container-config`` defaults to None → a fresh
        ``ContainerConfig()`` (default-configured)."""
        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {"score_card_name": "Test"}
            mock_start.return_value = "wf"
            runner.invoke(
                app,
                ["execute", "-t", "suite.yaml", "-s", "systems.yaml", "-r", "card.yaml"],
            )
        kwargs = mock_start.call_args.kwargs
        assert isinstance(kwargs["container_config"], ContainerConfig)
        # Default-configured: documented default timeout
        assert kwargs["container_config"].timeout_seconds == 300
        # --datasets-config not given → None
        assert kwargs["datasets_config_path"] is None

    def test_execute_tests_forwards_container_and_datasets_config(self, runner, tmp_path):
        cfg = tmp_path / "container.yaml"
        cfg.write_text("timeout_seconds: 42\n")

        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.workflow.DBOS"),
        ):
            mock_start.return_value = "wf"
            runner.invoke(
                app,
                [
                    "execute-tests",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "--container-config",
                    str(cfg),
                    "-d",
                    "ds.yaml",
                ],
            )
        kwargs = mock_start.call_args.kwargs
        assert kwargs["container_config"].timeout_seconds == 42
        assert kwargs["datasets_config_path"] == "ds.yaml"

    def test_generate_dataset_forwards_container_and_datasets_config(self, runner, tmp_path):
        cfg = tmp_path / "container.yaml"
        cfg.write_text("timeout_seconds: 7\n")

        with (
            patch("asqi.workflow.start_data_generation") as mock_start,
            patch("asqi.workflow.DBOS"),
        ):
            mock_start.return_value = "wf"
            runner.invoke(
                app,
                [
                    "generate-dataset",
                    "-t",
                    "gen.yaml",
                    "--container-config",
                    str(cfg),
                    "-d",
                    "ds.yaml",
                ],
            )
        kwargs = mock_start.call_args.kwargs
        assert kwargs["container_config"].timeout_seconds == 7
        assert kwargs["datasets_config_path"] == "ds.yaml"


# ---------------------------------------------------------------------------
# Audit flag mutual exclusion
# ---------------------------------------------------------------------------


class TestAuditFlagMutualExclusion:
    """``--audit-responses`` and ``--skip-audit`` cannot be used together."""

    def test_execute_rejects_both_audit_flags(self, runner):
        with (
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {
                "score_card_name": "X",
                "indicators": [{"id": "a1", "type": "audit"}],
            }
            result = runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "card.yaml",
                    "-a",
                    "audit.yaml",
                    "--skip-audit",
                ],
            )
        # resolve_audit_options raises typer.Exit(1) when both are provided
        assert result.exit_code == 1

    def test_execute_forwards_audit_responses_yaml_into_workflow(self, runner, tmp_path):
        """When ``--audit-responses`` is supplied, the loaded responses dict
        must reach ``start_test_execution`` as ``audit_responses_data``. Pin
        the positive path — the negative ``AuditResponsesRequiredError`` case
        is covered by ``test_audit_responses_required_error_exits_one``
        above."""
        audit_yaml = tmp_path / "audit.yaml"
        _write_yaml(
            audit_yaml,
            {
                "responses": [
                    {
                        "indicator_id": "ind_1",
                        "selected_outcome": "A",
                        "notes": "looks good",
                    }
                ]
            },
        )

        with (
            patch("asqi.workflow.start_test_execution") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {
                "score_card_name": "AuditCard",
                "indicators": [
                    {
                        "id": "ind_1",
                        "type": "audit",
                        "name": "Manual review",
                        "assessment": [
                            {"outcome": "A", "description": "Pass"},
                            {"outcome": "B", "description": "Fail"},
                        ],
                    }
                ],
            }
            mock_start.return_value = "wf"
            result = runner.invoke(
                app,
                [
                    "execute",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "-r",
                    "card.yaml",
                    "-a",
                    str(audit_yaml),
                ],
            )

        assert result.exit_code == 0
        kwargs = mock_start.call_args.kwargs
        # The loaded YAML reaches the workflow as audit_responses_data.
        assert kwargs["audit_responses_data"] == {
            "responses": [
                {
                    "indicator_id": "ind_1",
                    "selected_outcome": "A",
                    "notes": "looks good",
                }
            ]
        }

    def test_evaluate_score_cards_forwards_audit_responses_yaml(self, runner, tmp_path):
        """Same audit forwarding contract as ``execute`` — the positive path.
        ``--audit-responses`` is loaded and reaches
        ``start_score_card_evaluation`` as ``audit_responses_data``."""
        audit_yaml = tmp_path / "audit.yaml"
        _write_yaml(
            audit_yaml,
            {
                "responses": [
                    {
                        "indicator_id": "ind_1",
                        "selected_outcome": "B",
                        "notes": "needs follow-up",
                    }
                ]
            },
        )

        with (
            patch("asqi.workflow.start_score_card_evaluation") as mock_start,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {
                "score_card_name": "AuditCard",
                "indicators": [
                    {
                        "id": "ind_1",
                        "type": "audit",
                        "name": "Manual review",
                        "assessment": [
                            {"outcome": "A", "description": "Pass"},
                            {"outcome": "B", "description": "Fail"},
                        ],
                    }
                ],
            }
            mock_start.return_value = "wf"
            result = runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                    "-a",
                    str(audit_yaml),
                ],
            )

        assert result.exit_code == 0
        kwargs = mock_start.call_args.kwargs
        assert kwargs["audit_responses_data"] == {
            "responses": [
                {
                    "indicator_id": "ind_1",
                    "selected_outcome": "B",
                    "notes": "needs follow-up",
                }
            ]
        }

    def test_evaluate_score_cards_rejects_both_audit_flags(self, runner):
        """The same audit mutual-exclusion rule applies on
        ``evaluate-score-cards`` (it shares the ``resolve_audit_options``
        helper with ``execute``)."""
        with (
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.workflow.DBOS"),
        ):
            mock_load_card.return_value = {
                "score_card_name": "X",
                "indicators": [{"id": "a1", "type": "audit"}],
            }
            result = runner.invoke(
                app,
                [
                    "evaluate-score-cards",
                    "--input-file",
                    "input.json",
                    "-r",
                    "card.yaml",
                    "-a",
                    "audit.yaml",
                    "--skip-audit",
                ],
            )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Process-wide signal & atexit handler registration
# ---------------------------------------------------------------------------


class TestResolveAuditOptionsDirectly:
    """``resolve_audit_options`` is the CLI helper that actually constructs
    and raises ``AuditResponsesRequiredError`` when audit indicators exist
    but neither ``--audit-responses`` nor ``--skip-audit`` is supplied.

    The CLI tests above mock the helper at the ``side_effect`` level. Pin the
    helper's own behaviour directly so a regression that drops the raise (or
    mis-shapes the payload) surfaces independently of the CLI plumbing."""

    def test_raises_audit_responses_required_with_documented_payload(self):
        """The raised ``AuditResponsesRequiredError`` must carry the score-card
        name and the filtered list of audit indicators (documented
        attributes)."""
        from asqi.main import resolve_audit_options

        score_card = {
            "score_card_name": "AuditCard",
            "indicators": [
                {
                    "id": "ind_1",
                    "type": "audit",
                    "name": "Manual review",
                    "assessment": [
                        {"outcome": "A", "description": "Pass"},
                        {"outcome": "B", "description": "Fail"},
                    ],
                },
                # Non-audit indicator; must NOT appear in the error payload.
                {
                    "id": "ind_2",
                    "type": "metric",
                    "metric": "accuracy",
                },
            ],
        }

        with pytest.raises(AuditResponsesRequiredError) as excinfo:
            resolve_audit_options(
                score_card_data=score_card,
                audit_responses_path=None,
                skip_audit_flag=False,
            )

        err = excinfo.value
        assert err.score_card_name == "AuditCard"
        # Only the audit-typed indicator is included.
        assert len(err.audit_indicators) == 1
        assert err.audit_indicators[0]["id"] == "ind_1"

    def test_no_audit_indicators_returns_unchanged_card(self):
        """Per the helper docstring: when the card has no audit indicators,
        the helper returns the card unchanged plus ``None`` for responses —
        no error raised, even if neither flag is set."""
        from asqi.main import resolve_audit_options

        card = {
            "score_card_name": "MetricOnly",
            "indicators": [{"id": "i1", "type": "metric", "metric": "m"}],
        }
        out_card, responses = resolve_audit_options(
            score_card_data=card,
            audit_responses_path=None,
            skip_audit_flag=False,
        )
        assert out_card is card
        assert responses is None


class TestValidateCommandDispatchOrder:
    """Documented dispatch order: ``_validate_unique_ids(test_suite_config)``
    runs **before** ``load_and_validate_plan(...)``. Pin the order so a
    refactor that swaps them (or drops the uniqueness pass) surfaces
    immediately."""

    def test_validate_unique_ids_runs_before_load_and_validate_plan(self, runner):
        call_order: list[str] = []

        def fake_unique_ids(*args, **kwargs):
            call_order.append("validate_unique_ids")

        def fake_load_plan(*args, **kwargs):
            call_order.append("load_and_validate_plan")
            return {"status": "success", "errors": []}

        with (
            patch("asqi.main._validate_unique_ids", side_effect=fake_unique_ids),
            patch("asqi.main.load_and_validate_plan", side_effect=fake_load_plan),
        ):
            result = runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "--manifests-dir",
                    "manifests",
                ],
            )

        assert result.exit_code == 0
        assert call_order == ["validate_unique_ids", "load_and_validate_plan"]


class TestStartupSignalHandlers:
    """The startup callback registers ``SIGINT`` and ``SIGTERM`` handlers
    that call ``shutdown_containers()`` and an ``atexit`` cleanup hook.
    Triggered once per process when any subcommand runs."""

    def test_startup_registers_sigint_sigterm_and_atexit(self, runner):
        import signal as _signal

        with (
            patch("asqi.main.signal.signal") as mock_signal,
            patch("asqi.main.atexit.register") as mock_atexit,
            # Short-circuit the actual subcommand so we exercise only the
            # global callback (which owns the registration).
            patch(
                "asqi.main.load_and_validate_plan",
                return_value={"status": "success", "errors": []},
            ),
        ):
            runner.invoke(
                app,
                [
                    "validate",
                    "-t",
                    "suite.yaml",
                    "-s",
                    "systems.yaml",
                    "--manifests-dir",
                    "manifests",
                ],
            )

        # atexit registered exactly once.
        mock_atexit.assert_called_once()
        # Both SIGINT and SIGTERM registered.
        registered_signals = {call.args[0] for call in mock_signal.call_args_list}
        assert _signal.SIGINT in registered_signals
        assert _signal.SIGTERM in registered_signals
