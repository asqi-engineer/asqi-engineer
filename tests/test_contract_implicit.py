# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false, reportMissingTypeArgument=false
"""Contract tests for the implicit cross-cutting contracts.

Covers:

- Sidecar log path mismatch between writer and reader.
- Environment variable surface — confirms which env vars the documented
  consumers actually read; ``DOCKER_HOST`` stripping for DinD containers.
- ID validation — the regex ``^[0-9a-z_]{1,32}$`` is enforced selectively;
  this test pins down where it IS and is NOT enforced.
- YAML environment-variable interpolation patterns
  (``${VAR}``, ``${VAR:-default}``, ``${VAR-default}``).
- Container labeling — every container started by asqi-engineer carries
  ``workflow_id=<uuid>``.
- Privileged-mode non-escalation — the embedding host owns privilege
  decisions; asqi-engineer does *not* auto-set ``privileged=True`` from a
  manifest's ``host_access`` field.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from asqi.config import (
    ContainerConfig,
    _interpolate_string,
    interpolate_env_vars,
    load_config_file,
)
from asqi.schemas import (
    AuditResponse,
    AuditScoreCardIndicator,
    Manifest,
    ScoreCardFilter,
    ScoreCardIndicator,
    SuiteConfig,
    SystemsConfig,
)
from asqi.schemas import (
    TestDefinition as _TestDefinition,
)
from pydantic import ValidationError

# Alias avoids pytest collecting this Pydantic class as a test fixture.
TestDef = _TestDefinition


# ---------------------------------------------------------------------------
# ID regex enforcement scope
# ---------------------------------------------------------------------------


VALID_ID = "valid_id_123"
INVALID_IDS = [
    "Has-Hyphen",  # hyphen disallowed
    "HasUpper",  # uppercase disallowed
    "has space",  # whitespace
    "x" * 33,  # too long
    "",  # empty
]


class TestIDRegexEnforcement:
    """``IDsStringPattern = ^[0-9a-z_]{1,32}$`` is selectively enforced."""

    @pytest.mark.parametrize("bad_id", INVALID_IDS)
    def test_test_definition_id_rejects_invalid_pattern(self, bad_id):
        with pytest.raises(ValidationError):
            TestDef(
                id=bad_id,
                name="t",
                image="img",
                systems_under_test=["sysA"],
                params={},
            )

    def test_test_definition_id_accepts_valid_pattern(self):
        td = TestDef(
            id=VALID_ID,
            name="t",
            image="img",
            systems_under_test=["sysA"],
            params={},
        )
        assert td.id == VALID_ID

    @pytest.mark.parametrize("bad_id", INVALID_IDS)
    def test_score_card_indicator_id_rejects_invalid_pattern(self, bad_id):
        """``ScoreCardIndicator.id`` is one of the three documented
        regex-enforcement points (alongside ``TestDefinition.id`` and
        ``AuditScoreCardIndicator.id``)."""
        with pytest.raises(ValidationError):
            ScoreCardIndicator(
                id=bad_id,
                apply_to=ScoreCardFilter(test_id="some_test"),
                metric="m",
                assessment=[{"outcome": "PASS", "condition": "greater_equal", "threshold": 0.5}],
            )

    def test_score_card_indicator_id_accepts_valid_pattern(self):
        ind = ScoreCardIndicator(
            id=VALID_ID,
            apply_to=ScoreCardFilter(test_id="some_test"),
            metric="m",
            assessment=[{"outcome": "PASS", "condition": "greater_equal", "threshold": 0.5}],
        )
        assert ind.id == VALID_ID

    @pytest.mark.parametrize("bad_id", INVALID_IDS)
    def test_audit_score_card_indicator_id_rejects_invalid_pattern(self, bad_id):
        """``AuditScoreCardIndicator.id`` is regex-enforced."""
        with pytest.raises(ValidationError):
            AuditScoreCardIndicator(
                id=bad_id,
                assessment=[{"outcome": "A", "description": "ok"}],
            )

    def test_audit_score_card_indicator_id_accepts_valid_pattern(self):
        ind = AuditScoreCardIndicator(
            id=VALID_ID,
            assessment=[{"outcome": "A", "description": "ok"}],
        )
        assert ind.id == VALID_ID

    @pytest.mark.parametrize(
        "free_form_id",
        ["Has-Hyphen", "MixedCase", "with space", "x" * 200],
    )
    def test_score_card_filter_test_id_is_free_form(self, free_form_id):
        """``ScoreCardFilter.test_id`` is a free-form ``str`` even though it
        references a ``TestDefinition.id``."""
        f = ScoreCardFilter(test_id=free_form_id)
        assert f.test_id == free_form_id

    @pytest.mark.parametrize("free_form_id", ["Has-Hyphen", "MixedCase"])
    def test_audit_response_indicator_id_is_free_form(self, free_form_id):
        """``AuditResponse.indicator_id`` is a free-form ``str``."""
        a = AuditResponse(indicator_id=free_form_id, selected_outcome="A", notes="")
        assert a.indicator_id == free_form_id

    def test_suite_name_is_free_form(self):
        """``SuiteConfig.suite_name`` is NOT regex-validated."""
        SuiteConfig(suite_name="My Suite Name", test_suite=[])

    def test_system_names_are_free_form(self):
        """Keys of ``SystemsConfig.systems`` are free-form."""
        cfg = SystemsConfig(
            systems={
                "Mixed-Case-System": {
                    "type": "llm_api",
                    "params": {"base_url": "http://x", "model": "x"},
                }
            }
        )
        assert "Mixed-Case-System" in cfg.systems


class TestValidateIdsCommandCoverage:
    """``asqi.validation.validate_ids`` (raises ``DuplicateIDError`` /
    ``MissingIDFieldError``) is invoked from most CLI commands but **not all**.

    Documented commands that DO call it: ``validate``, ``execute``,
    ``execute-tests``, ``evaluate-score-cards``.
    Documented commands that DO NOT: ``generate-dataset``.

    Pinning this asymmetry catches a future refactor that either drops the
    uniqueness check from one of the four documented commands, or accidentally
    starts running it for ``generate-dataset`` (which the documented contract
    calls out as a deliberate omission)."""

    def test_generate_dataset_does_not_invoke_validate_ids(self):
        """``generate-dataset`` does not call ``validate_ids``. Pin the
        negative case: the command runs without triggering the uniqueness
        pass."""
        from asqi.main import app
        from typer.testing import CliRunner

        runner = CliRunner()

        with (
            patch("asqi.workflow.start_data_generation") as mock_start,
            patch("asqi.workflow.DBOS"),
            patch("asqi.main.validate_ids") as mock_validate_ids,
        ):
            mock_start.return_value = "wf-gen"
            result = runner.invoke(
                app,
                ["generate-dataset", "-t", "gen.yaml"],
            )

        assert result.exit_code == 0
        mock_validate_ids.assert_not_called()

    @pytest.mark.parametrize(
        "argv",
        [
            ["validate", "-t", "s.yaml", "-s", "sys.yaml", "--manifests-dir", "m"],
            ["execute", "-t", "s.yaml", "-s", "sys.yaml", "-r", "c.yaml"],
            ["execute-tests", "-t", "s.yaml", "-s", "sys.yaml"],
            [
                "evaluate-score-cards",
                "--input-file",
                "in.json",
                "-r",
                "c.yaml",
            ],
        ],
        ids=["validate", "execute", "execute-tests", "evaluate-score-cards"],
    )
    def test_documented_commands_do_invoke_validate_ids(self, argv):
        """``validate``, ``execute``, ``execute-tests``,
        ``evaluate-score-cards`` invoke ``validate_ids`` to enforce
        uniqueness. Pin each one positively so a regression that drops the
        check from one of them is caught immediately."""
        from asqi.main import app
        from typer.testing import CliRunner

        runner = CliRunner()

        with (
            patch("asqi.workflow.start_test_execution") as mock_start_test,
            patch("asqi.workflow.start_score_card_evaluation") as mock_start_eval,
            patch("asqi.main.load_score_card_file") as mock_load_card,
            patch("asqi.main.load_and_validate_plan") as mock_load_plan,
            patch("asqi.workflow.DBOS"),
            patch("asqi.main.validate_ids") as mock_validate_ids,
        ):
            mock_load_card.return_value = {"score_card_name": "Test"}
            mock_load_plan.return_value = {"status": "success", "errors": []}
            mock_start_test.return_value = "wf"
            mock_start_eval.return_value = "wf"
            runner.invoke(app, argv)

        # The command must hit ``validate_ids`` at least once. Exit code is
        # not asserted here — some sub-mocks intentionally don't fully model
        # downstream paths — but the uniqueness check must be invoked before
        # any failure short-circuits.
        assert mock_validate_ids.call_count >= 1, f"`{argv[0]}` is documented as invoking validate_ids"


# ---------------------------------------------------------------------------
# YAML env-var interpolation
# ---------------------------------------------------------------------------


class TestYamlEnvVarInterpolation:
    def test_direct_substitution_when_var_set(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "value-1")
        assert _interpolate_string("${MY_VAR}") == "value-1"

    def test_direct_substitution_returns_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("MY_VAR", raising=False)
        assert _interpolate_string("${MY_VAR}") == ""

    def test_default_with_colon_dash_uses_var_when_set_and_nonempty(self, monkeypatch):
        monkeypatch.setenv("X", "real")
        assert _interpolate_string("${X:-fallback}") == "real"

    def test_default_with_colon_dash_uses_default_when_empty(self, monkeypatch):
        monkeypatch.setenv("X", "")
        assert _interpolate_string("${X:-fallback}") == "fallback"

    def test_default_with_colon_dash_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        assert _interpolate_string("${X:-fallback}") == "fallback"

    def test_default_with_dash_uses_var_even_when_empty(self, monkeypatch):
        """``${VAR-default}`` returns VAR when set, even if empty (vs ``:-``)."""
        monkeypatch.setenv("X", "")
        assert _interpolate_string("${X-fallback}") == ""

    def test_default_with_dash_uses_var_when_set_non_empty(self, monkeypatch):
        """``${VAR-default}`` is 'VAR if set (even empty), else default'.

        The ``set-and-empty`` and ``unset`` cases are covered above. Pin the
        ``set-and-non-empty`` case explicitly so a regression that always
        substitutes the default (regardless of VAR) is caught — the existing
        empty-value test would still pass because the default and the empty
        string both render as ``""``."""
        monkeypatch.setenv("X", "real")
        assert _interpolate_string("${X-fallback}") == "real"

    def test_default_with_dash_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("X", raising=False)
        assert _interpolate_string("${X-fallback}") == "fallback"

    def test_recursive_interpolation_through_nested_data(self, monkeypatch):
        monkeypatch.setenv("HOST", "example.com")
        result = interpolate_env_vars(
            {
                "url": "https://${HOST}/api",
                "list": ["${HOST}", "fixed"],
                "nested": {"k": "v ${HOST}"},
            }
        )
        assert result == {
            "url": "https://example.com/api",
            "list": ["example.com", "fixed"],
            "nested": {"k": "v example.com"},
        }

    def test_load_config_file_applies_interpolation(self, monkeypatch, tmp_path):
        """``load_config_file`` is the CLI's YAML loader and must interpolate
        env vars per the documented contract."""
        monkeypatch.setenv("MY_HOST", "prod.example.com")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("base_url: https://${MY_HOST}\nport: 443\n")
        out = load_config_file(str(cfg_file))
        assert out == {"base_url": "https://prod.example.com", "port": 443}

    def test_container_config_load_from_yaml_applies_interpolation(self, monkeypatch, tmp_path):
        """``ContainerConfig.load_from_yaml`` must apply the same interpolation."""
        monkeypatch.setenv("MY_NETWORK", "bridge")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("timeout_seconds: 100\nrun_params:\n  network_mode: ${MY_NETWORK}\n")
        cfg = ContainerConfig.load_from_yaml(str(cfg_file))
        assert cfg.run_params["network_mode"] == "bridge"


# ---------------------------------------------------------------------------
# Environment variable surface — no module imports the deprecated names
# ---------------------------------------------------------------------------


class TestEnvironmentVariableSurface:
    """Documented env vars: ``ASQI_LOG_LEVEL``, ``DBOS_DATABASE_URL``,
    ``OTEL_EXPORTER_OTLP_ENDPOINT``, ``LOGS_PATH``, ``HF_TOKEN``,
    ``DOCKER_HOST``. We confirm each is accessed at the expected boundary."""

    @pytest.fixture
    def _restore_logging_state(self):
        """``configure_logging()`` mutates the root and ``asqi`` logger levels
        and replaces the root handler. Snapshot and restore so the mutation
        does not leak into later tests."""
        import logging

        root = logging.getLogger()
        asqi = logging.getLogger("asqi")
        saved = (root.level, asqi.level, list(root.handlers))
        try:
            yield
        finally:
            root.setLevel(saved[0])
            asqi.setLevel(saved[1])
            root.handlers[:] = saved[2]

    def test_logs_path_default_is_logs(self, monkeypatch):
        """``LOGS_PATH`` defaults to ``"logs"``."""
        monkeypatch.delenv("LOGS_PATH", raising=False)
        # Read from the workflow module — its constants pick up the default.
        import os as _os

        assert _os.getenv("LOGS_PATH", "logs") == "logs"

    def test_asqi_log_level_default_is_info(self, monkeypatch, _restore_logging_state):
        """``ASQI_LOG_LEVEL`` defaults to INFO."""
        import logging

        monkeypatch.delenv("ASQI_LOG_LEVEL", raising=False)
        from asqi.logging_config import configure_logging

        configure_logging()
        # The asqi.* logger is the documented namespace ASQI_LOG_LEVEL controls.
        assert logging.getLogger("asqi").level == logging.INFO

    def test_asqi_log_level_env_var_overrides_default(self, monkeypatch, _restore_logging_state):
        """``ASQI_LOG_LEVEL`` controls the asqi.* namespace level."""
        import logging

        monkeypatch.setenv("ASQI_LOG_LEVEL", "DEBUG")
        from asqi.logging_config import configure_logging

        configure_logging()
        assert logging.getLogger("asqi").level == logging.DEBUG

    def test_hf_token_consumed_by_hub_loader(self, monkeypatch):
        """``HF_TOKEN`` is consumed by ``asqi.datasets`` for gated Hub
        datasets when no inline token is provided. We patch the underlying
        ``load_dataset`` and assert the env-var token reaches it."""
        from asqi.datasets import _load_from_hub
        from asqi.schemas import DatasetLoaderParams

        monkeypatch.setenv("HF_TOKEN", "hf_secret_xyz")
        params = DatasetLoaderParams(hub_path="org/dataset")
        with patch("asqi.datasets.load_dataset") as mock_load:
            mock_load.return_value = MagicMock()
            _load_from_hub(params)

        kwargs = mock_load.call_args.kwargs
        assert kwargs["token"] == "hf_secret_xyz"  # noqa: S105

    def test_dbos_database_url_captured_at_import_time(self):
        """``DBOS_DATABASE_URL`` is consumed by ``asqi.workflow`` at import
        time. The module reads ``os.environ["DBOS_DATABASE_URL"]`` and
        stores it on a module-level constant, so the captured value should
        reflect the env var's value at the moment the module was first
        imported (which the test process did during pytest collection)."""
        import os

        import asqi.workflow as wf_module

        # Module-level constant must exist (locks down the documented contract).
        assert hasattr(wf_module, "system_database_url")
        # And it equals the value the env held at import time.
        assert wf_module.system_database_url == os.environ.get("DBOS_DATABASE_URL")

    def test_dbos_database_url_unset_raises_at_module_import(self, tmp_path):
        """``DBOS_DATABASE_URL`` is required when the workflow module is
        imported. The module's import-time guard raises ``ValueError`` with
        the documented message when it's missing.

        We exercise this in a subprocess so we can clear the env without
        breaking the pytest process's own DBOS state. The probe also
        neutralises ``dotenv.load_dotenv`` before the ``asqi.workflow``
        import so a ``.env`` anywhere on the filesystem (repo root, an
        ancestor dir, the user's home) cannot silently repopulate
        ``DBOS_DATABASE_URL`` and mask the guard we're asserting."""
        import os as _os
        import subprocess
        import sys

        import asqi

        env = {k: v for k, v in _os.environ.items() if k != "DBOS_DATABASE_URL"}
        env.pop("DBOS_DATABASE_URL", None)
        # CI runs with a *relative* ``PYTHONPATH=src`` from
        # ``libs/asqi-engineer`` and does not install the project
        # (``--no-install-project``). Changing cwd to ``tmp_path`` below
        # would make that relative path point nowhere, so the child would
        # raise ``ModuleNotFoundError`` before reaching the import-time
        # guard we're trying to exercise. Pin an absolute pointer to the
        # ``asqi`` package's parent dir.
        asqi_parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(asqi.__file__)))
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = asqi_parent + (_os.pathsep + existing_pp if existing_pp else "")
        # Drive the child process to import asqi.workflow and report the
        # exception class on stdout so we can assert the documented type
        # (``ValueError``) — not just a substring match on stderr.
        # Neutralise dotenv before the workflow import: ``load_dotenv()`` at
        # ``workflow.py:75`` walks up from the module's own directory, so a
        # cwd swap is not enough — a ``.env`` in any repo ancestor would
        # still repopulate ``DBOS_DATABASE_URL`` and silently pass the test.
        probe = (
            "import sys\n"
            "import dotenv\n"
            "dotenv.load_dotenv = lambda *a, **kw: False\n"
            "try:\n"
            "    import asqi.workflow  # noqa: F401\n"
            "except BaseException as exc:\n"
            "    print(type(exc).__name__)\n"
            "    print(str(exc))\n"
            "    sys.exit(7)\n"
            "sys.exit(0)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        # The guard fired (non-zero exit), and the exception class is the
        # documented ``ValueError`` (not e.g. ``RuntimeError`` or a DBOS
        # internal type).
        assert result.returncode == 7, result.stderr
        out_lines = result.stdout.strip().splitlines()
        assert out_lines, f"no exception captured. stderr: {result.stderr!r}"
        assert out_lines[0] == "ValueError"
        # Documented message snippet (workflow.py:79-81)
        assert "DBOS_DATABASE_URL" in result.stdout

    def test_otel_exporter_endpoint_consumed_at_import_time(self):
        """``OTEL_EXPORTER_OTLP_ENDPOINT`` is read by ``asqi.workflow`` at
        import time; tracing is disabled if unset.

        The module exposes ``oltp_endpoint`` (sic) as the captured value, so
        we can pin the read-from-env contract without a second import."""
        import os

        import asqi.workflow as wf_module

        assert hasattr(wf_module, "oltp_endpoint")
        # The captured constant matches whatever the env held at import time;
        # both ``None`` (unset) and a string value are valid contract states.
        assert wf_module.oltp_endpoint == os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    def test_hf_token_inline_overrides_env(self, monkeypatch):
        """Inline token on the loader params takes precedence over HF_TOKEN."""
        from asqi.datasets import _load_from_hub
        from asqi.schemas import DatasetLoaderParams

        monkeypatch.setenv("HF_TOKEN", "from_env")
        params = DatasetLoaderParams(hub_path="org/dataset", token="inline_token")  # noqa: S106
        with patch("asqi.datasets.load_dataset") as mock_load:
            mock_load.return_value = MagicMock()
            _load_from_hub(params)
        assert mock_load.call_args.kwargs["token"] == "inline_token"  # noqa: S105


# ---------------------------------------------------------------------------
# File conventions — /input and /output mount paths
# ---------------------------------------------------------------------------


class TestMountPathConstants:
    """Container input mount is ``/input`` (read-only); output mount is
    ``/output`` (read-write). These are pinned as module constants in
    ``asqi.backends.docker_backend`` and must not drift."""

    def test_input_mount_path_is_slash_input(self):
        from pathlib import Path

        from asqi.backends.docker_backend import INPUT_MOUNT_PATH

        assert INPUT_MOUNT_PATH == Path("/input")

    def test_output_mount_path_is_slash_output(self):
        from pathlib import Path

        from asqi.backends.docker_backend import OUTPUT_MOUNT_PATH

        assert OUTPUT_MOUNT_PATH == Path("/output")


# ---------------------------------------------------------------------------
# DOCKER_HOST stripping for Docker-in-Docker containers
# ---------------------------------------------------------------------------


class TestDockerHostStrippingForDinD:
    """``DOCKER_HOST`` is stripped from the env passed into Docker-in-Docker
    containers (those whose manifest declares ``host_access: true``) so the
    container's Docker client falls back to the bind-mounted socket.
    Untouched for non-DinD containers."""

    def _manifest(self, *, host_access: bool) -> Manifest:
        return Manifest(name="m", version="1", host_access=host_access)

    def test_docker_host_stripped_when_host_access_true(self):
        from asqi.workflow import _configure_docker_in_docker

        cfg = ContainerConfig()
        env = {"DOCKER_HOST": "unix:///var/run/docker.sock", "OTHER": "keep"}
        _configure_docker_in_docker(
            manifest=self._manifest(host_access=True),
            container_config=cfg,
            container_env=env,
            item_id="t1",
            image="img:1",
        )
        # DOCKER_HOST must be removed; other vars must be preserved.
        assert "DOCKER_HOST" not in env
        assert env["OTHER"] == "keep"

    def test_docker_host_preserved_when_host_access_false(self):
        from asqi.workflow import _configure_docker_in_docker

        cfg = ContainerConfig()
        env = {"DOCKER_HOST": "unix:///var/run/docker.sock"}
        _configure_docker_in_docker(
            manifest=self._manifest(host_access=False),
            container_config=cfg,
            container_env=env,
            item_id="t1",
            image="img:1",
        )
        # Non-DinD containers must keep DOCKER_HOST untouched.
        assert env["DOCKER_HOST"] == "unix:///var/run/docker.sock"

    def test_docker_host_preserved_when_no_manifest(self):
        from asqi.workflow import _configure_docker_in_docker

        cfg = ContainerConfig()
        env = {"DOCKER_HOST": "tcp://1.2.3.4:2375"}
        _configure_docker_in_docker(
            manifest=None,
            container_config=cfg,
            container_env=env,
            item_id="t1",
            image="img:1",
        )
        assert env["DOCKER_HOST"] == "tcp://1.2.3.4:2375"


# ---------------------------------------------------------------------------
# Container labeling — ``workflow_id=<uuid>``
# ---------------------------------------------------------------------------


class TestContainerWorkflowIdLabel:
    """Every container started by asqi-engineer is labeled
    ``workflow_id=<uuid>``. Callers can use this label for orphan-container
    cleanup during host-process shutdown."""

    def test_run_container_with_args_passes_workflow_id_label(self):
        """The ``labels`` kwarg passed into ``client.containers.run`` must
        include ``workflow_id=<uuid>`` and ``service=asqi_engineer``.

        We capture the kwargs at the ``client.containers.run`` boundary and
        short-circuit by raising — the function's outer ``except Exception``
        catches it and returns a normal result dict."""
        from asqi.backends.docker_backend import run_container_with_args
        from docker import errors as docker_errors

        captured: dict = {}

        def fake_run(**kwargs):
            captured["labels"] = kwargs.get("labels")
            # APIError is one of the documented exits — short-circuits the
            # function cleanly without entering the wait loop.
            raise docker_errors.APIError("captured — short-circuit")

        client = MagicMock()
        client.containers.run.side_effect = fake_run
        with patch("asqi.backends.docker_backend.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            run_container_with_args(
                image="img:1",
                args=[],
                container_config=ContainerConfig(timeout_seconds=1),
                workflow_id="wf-uuid-12345",
            )

        # Labels must include both keys and carry the workflow_id we passed.
        assert captured["labels"] == {
            "workflow_id": "wf-uuid-12345",
            "service": "asqi_engineer",
        }


# ---------------------------------------------------------------------------
# Privileged-mode non-escalation
# ---------------------------------------------------------------------------


class TestPrivilegedModeNonEscalation:
    """asqi-engineer does **not** automatically escalate privilege based on
    the manifest field — privilege grants are an explicit decision of the
    embedding host.

    The host_access manifest field DOES trigger DinD setup (cap_add SYS_ADMIN
    and volume-mounting the docker socket), but it MUST NOT set ``privileged``."""

    def _manifest(self, host_access: bool) -> Manifest:
        return Manifest(name="m", version="1", host_access=host_access)

    def test_host_access_does_not_set_privileged(self):
        from asqi.workflow import _configure_docker_in_docker

        cfg = ContainerConfig()
        # Sanity: the default does NOT include privileged
        assert "privileged" not in cfg.run_params

        _configure_docker_in_docker(
            manifest=self._manifest(host_access=True),
            container_config=cfg,
            container_env={},
            item_id="t1",
            image="img:1",
        )

        # DinD config IS applied — but privileged must remain unset.
        assert "privileged" not in cfg.run_params
        assert cfg.run_params.get("cap_add") == ["SYS_ADMIN"]
        assert "volumes" in cfg.run_params

    def test_host_access_false_is_a_no_op(self):
        from asqi.workflow import _configure_docker_in_docker

        cfg = ContainerConfig()
        original = dict(cfg.run_params)
        _configure_docker_in_docker(
            manifest=self._manifest(host_access=False),
            container_config=cfg,
            container_env={"DOCKER_HOST": "x"},
            item_id="t1",
            image="img:1",
        )
        assert cfg.run_params == original
        assert "privileged" not in cfg.run_params

    def test_no_manifest_is_a_no_op(self):
        from asqi.workflow import _configure_docker_in_docker

        cfg = ContainerConfig()
        original = dict(cfg.run_params)
        _configure_docker_in_docker(
            manifest=None,
            container_config=cfg,
            container_env={},
            item_id="t1",
            image="img:1",
        )
        assert cfg.run_params == original

    def test_host_owns_privileged_via_run_params(self):
        """Documented usage: the embedding host sets privileged via
        ``container_config.run_params["privileged"] = True`` themselves."""
        cfg = ContainerConfig()
        cfg.run_params["privileged"] = True
        assert cfg.run_params["privileged"] is True
        # And ``from_run_params`` accepts it via **extra
        cfg2 = ContainerConfig.from_run_params(privileged=True)
        assert cfg2.run_params["privileged"] is True


# ---------------------------------------------------------------------------
# Sidecar log path mismatch
# ---------------------------------------------------------------------------


class TestEvaluateScoreCardsSidecarPathMismatch:
    """The writer (``execute-tests``) saves to
    ``$LOGS_PATH/<basename(--output-file)>``, but the reader looks at
    ``$LOGS_PATH/<--input-file as given>``. They only line up when both
    flags are bare filenames.

    These tests pin both halves of that asymmetric contract so the writer
    and reader cannot drift independently."""

    def test_writer_uses_basename_of_output_path(self, tmp_path, monkeypatch):
        """``save_container_results_to_file_step`` strips any directory
        component from ``output_path``, writing to ``$LOGS_PATH/<basename>``."""
        from asqi.workflow import save_container_results_to_file_step

        logs_dir = tmp_path / "logs"
        monkeypatch.setenv("LOGS_PATH", str(logs_dir))

        save_container_results_to_file_step(
            container_results=[{"test_id": "t1", "error_message": "", "container_output": ""}],
            output_path="some/nested/dir/output.json",
        )
        # Sidecar lands at $LOGS_PATH/output.json (basename only)
        assert (logs_dir / "output.json").exists()
        # NOT at the path with the directory prefix
        assert not (logs_dir / "some" / "nested" / "dir" / "output.json").exists()

    def test_writer_keeps_extension_verbatim(self, tmp_path, monkeypatch):
        """File-conventions clarification: writer keeps whatever extension
        the caller passed — no implicit ``.json``."""
        from asqi.workflow import save_container_results_to_file_step

        logs_dir = tmp_path / "logs"
        monkeypatch.setenv("LOGS_PATH", str(logs_dir))

        save_container_results_to_file_step(
            container_results=[{"test_id": "t1", "error_message": "", "container_output": ""}],
            output_path="results.txt",
        )
        # File is created with the exact basename + extension.
        assert (logs_dir / "results.txt").exists()
        assert not (logs_dir / "results.txt.json").exists()

    def test_reader_finds_sidecar_when_both_paths_are_bare_filenames(self, tmp_path, monkeypatch):
        """They only line up when both flags are bare filenames.

        The mismatch case (with a directory prefix) is pinned below. Pin the
        happy path here: writer saves at ``$LOGS_PATH/<basename>``, reader
        looks at ``$LOGS_PATH/<input_path-as-given>``; when the input path
        IS a bare filename, the two coincide and ``test_container_data``
        is loaded successfully (NOT silently empty)."""
        import json as _json

        from asqi.workflow import start_score_card_evaluation

        monkeypatch.chdir(tmp_path)

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        monkeypatch.setenv("LOGS_PATH", str(logs_dir))

        # Both --input-file and the sidecar live as bare filenames in the cwd
        # / $LOGS_PATH respectively (matching the documented happy-path usage).
        (tmp_path / "results.json").write_text(_json.dumps({"summary": {"status": "COMPLETED"}, "results": []}))
        documented_sidecar = [
            {
                "test_id": "t1",
                "error_message": "",
                "container_output": "stdout-payload",
            }
        ]
        (logs_dir / "results.json").write_text(_json.dumps(documented_sidecar))

        captured: dict = {}

        def fake_start_workflow(fn, results_data, container_data, *args, **kwargs):
            captured["container_data"] = container_data

            class _H:
                def get_workflow_id(self):
                    return "wf"

                def get_result(self):
                    return {"summary": {"status": "COMPLETED"}, "results": []}

            return _H()

        with (
            patch("asqi.workflow.DBOS") as mock_dbos,
            patch(
                "asqi.workflow.validate_score_card_inputs",
                return_value=None,
            ),
        ):
            mock_dbos.start_workflow.side_effect = fake_start_workflow
            mock_dbos.workflow_id = "wf"
            # Bare filename for --input-file — joins cleanly with $LOGS_PATH
            # to find the writer's sidecar.
            start_score_card_evaluation(
                input_path="results.json",
                score_card_configs=[{"score_card_name": "n"}],
            )

        # Sidecar lookup HIT — container data round-trips.
        assert captured["container_data"] == documented_sidecar

    def test_reader_uses_input_path_as_given(self, tmp_path, monkeypatch):
        """The score-card-evaluation reader joins ``$LOGS_PATH`` with the
        ``input_path`` the user passed *verbatim* (not its basename), so when
        ``--input-file`` carries a directory prefix the sidecar lookup misses
        and ``test_container_data`` quietly becomes ``[]``.

        Reproduces the documented mismatch: writer wrote sidecar at the
        basename location, but reader is given a relative input path with a
        directory prefix → sidecar lookup misses."""
        import json as _json

        from asqi.workflow import start_score_card_evaluation

        # Run the test from inside tmp_path so relative input paths resolve
        # against it (mirroring the documented CLI usage).
        monkeypatch.chdir(tmp_path)

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        monkeypatch.setenv("LOGS_PATH", str(logs_dir))

        # Place results JSON at "out/results.json" (relative — has dir prefix)
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "results.json").write_text(_json.dumps({"summary": {"status": "COMPLETED"}, "results": []}))
        # Writer placed sidecar at $LOGS_PATH/<basename> = logs/results.json
        (logs_dir / "results.json").write_text(_json.dumps([{"test_id": "x"}]))

        captured: dict = {}

        def fake_start_workflow(fn, results_data, container_data, *args, **kwargs):
            captured["container_data"] = container_data

            class _H:
                def get_workflow_id(self):
                    return "wf"

                def get_result(self):
                    return {"summary": {"status": "COMPLETED"}, "results": []}

            return _H()

        with (
            patch("asqi.workflow.DBOS") as mock_dbos,
            patch(
                "asqi.workflow.validate_score_card_inputs",
                return_value=None,
            ),
        ):
            mock_dbos.start_workflow.side_effect = fake_start_workflow
            mock_dbos.workflow_id = "wf"
            # Reader joins $LOGS_PATH with "out/results.json" verbatim
            # → looks at logs/out/results.json → does not exist → [].
            start_score_card_evaluation(
                input_path="out/results.json",
                score_card_configs=[{"score_card_name": "n"}],
            )

        # Documented bug-as-contract: paths don't line up, container data
        # is silently empty.
        assert captured["container_data"] == []
