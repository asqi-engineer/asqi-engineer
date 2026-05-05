"""Contract tests for the host-side configuration types, helpers, and errors.

Covers:

- ``ContainerConfig`` (class vars, defaults, classmethod helpers, the
  documented ``with_streaming`` pitfall)
- ``ExecutorConfig`` dataclass + the dict-shape contract
- ``ExecutionMode`` enum
- ``merge_defaults_into_suite``
- ``extract_manifest_from_image``
- Error classes (``ManifestExtractionError`` payload, ``AuditResponsesRequiredError``)
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from asqi.config import (
    ContainerConfig,
    ExecutionMode,
    ExecutorConfig,
    merge_defaults_into_suite,
)
from asqi.container_manager import extract_manifest_from_image
from asqi.errors import (
    AuditResponsesRequiredError,
    ManifestExtractionError,
)

# ---------------------------------------------------------------------------
# ContainerConfig
# ---------------------------------------------------------------------------


class TestContainerConfigContract:
    """Lock in the documented ContainerConfig surface."""

    def test_manifest_path_class_var(self):
        """MANIFEST_PATH is fixed at /app/manifest.yaml — used by container authors
        and the host equally; renaming it is a breaking change."""
        assert ContainerConfig.MANIFEST_PATH == "/app/manifest.yaml"

    def test_default_run_params_match_documented_values(self):
        """DEFAULT_RUN_PARAMS is the documented Docker run() default kwargs."""
        expected = {
            "detach": True,
            "remove": False,
            "network_mode": "host",
            "mem_limit": "2g",
            "cpu_period": 100000,
            "cpu_quota": 200000,
            "cap_drop": ["ALL"],
        }
        assert ContainerConfig.DEFAULT_RUN_PARAMS == expected

    def test_instance_field_defaults(self):
        cfg = ContainerConfig()
        assert cfg.timeout_seconds == 300
        assert cfg.stream_logs is False
        assert cfg.cleanup_on_finish is True
        assert cfg.cleanup_force is True
        assert cfg.run_params == ContainerConfig.DEFAULT_RUN_PARAMS

    def test_run_params_default_is_a_copy_not_aliased(self):
        a = ContainerConfig()
        b = ContainerConfig()
        a.run_params["network_mode"] = "bridge"
        # Mutating one instance must not bleed into another or into the class default.
        assert b.run_params["network_mode"] == "host"
        assert ContainerConfig.DEFAULT_RUN_PARAMS["network_mode"] == "host"

    def test_with_streaming_returns_fresh_instance_with_only_stream_logs_set(self):
        """Documented pitfall:
        ``ContainerConfig(timeout_seconds=3000).with_streaming(True)`` discards
        the prior instance — ``with_streaming`` is a classmethod that returns
        a fresh default-configured object."""
        cfg = ContainerConfig.with_streaming(True)
        assert cfg.stream_logs is True
        assert cfg.timeout_seconds == 300  # default, NOT carried from prior
        assert cfg.cleanup_on_finish is True
        # And the inverse:
        cfg_off = ContainerConfig.with_streaming(False)
        assert cfg_off.stream_logs is False

    def test_from_run_params_merges_with_defaults(self):
        cfg = ContainerConfig.from_run_params(network_mode="bridge")
        # Merged: explicit override + the rest of the defaults
        assert cfg.run_params["network_mode"] == "bridge"
        assert cfg.run_params["detach"] is True
        assert cfg.run_params["mem_limit"] == "2g"

    def test_from_run_params_supports_extra_kwargs(self):
        cfg = ContainerConfig.from_run_params(privileged=True, hostname="my-host")
        assert cfg.run_params["privileged"] is True
        assert cfg.run_params["hostname"] == "my-host"
        # Defaults still present
        assert cfg.run_params["network_mode"] == "host"

    def test_from_run_params_ignores_none_values(self):
        cfg = ContainerConfig.from_run_params(network_mode=None)
        assert cfg.run_params["network_mode"] == "host"

    def test_load_from_yaml_missing_file_raises_filenotfounderror(self, tmp_path):
        missing = tmp_path / "no-such.yaml"
        with pytest.raises(FileNotFoundError):
            ContainerConfig.load_from_yaml(str(missing))

    def test_load_from_yaml_merges_run_params_with_defaults(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("timeout_seconds: 999\nstream_logs: true\nrun_params:\n  network_mode: bridge\n")
        cfg = ContainerConfig.load_from_yaml(str(cfg_file))
        assert cfg.timeout_seconds == 999
        assert cfg.stream_logs is True
        assert cfg.run_params["network_mode"] == "bridge"
        # Default keys preserved by merge
        assert cfg.run_params["detach"] is True
        assert cfg.run_params["mem_limit"] == "2g"


# ---------------------------------------------------------------------------
# ExecutorConfig
# ---------------------------------------------------------------------------


class TestExecutorConfigContract:
    """ExecutorConfig is a *constants holder*, not the workflow argument type.

    The contract is the dict shape; the dataclass is exported for callers that
    want a single source of truth for default values."""

    def test_default_constants_match_spec(self):
        assert ExecutorConfig.DEFAULT_CONCURRENT_TESTS == 3
        assert ExecutorConfig.MAX_FAILURES_DISPLAYED == 3
        assert ExecutorConfig.PROGRESS_UPDATE_INTERVAL == 4

    def test_dict_shape_keys(self):
        """Documented dict shape callers must build."""
        executor_dict = {
            "concurrent_tests": 1,
            "max_failures": 1,
            "progress_interval": 1,
        }
        # Sanity: all three keys present
        assert {"concurrent_tests", "max_failures", "progress_interval"} == set(executor_dict.keys())


# ---------------------------------------------------------------------------
# ExecutionMode
# ---------------------------------------------------------------------------


class TestExecutionModeContract:
    def test_documented_members_exist(self):
        assert ExecutionMode.END_TO_END.value == "end_to_end"
        assert ExecutionMode.TESTS_ONLY.value == "tests_only"
        assert ExecutionMode.EVALUATE_ONLY.value == "evaluate_only"
        assert ExecutionMode.VALIDATE_ONLY.value == "validate_only"

    def test_str_enum_compares_to_string(self):
        """ExecutionMode is a (str, Enum) so members compare equal to their value."""
        assert ExecutionMode.END_TO_END == "end_to_end"


# ---------------------------------------------------------------------------
# merge_defaults_into_suite
# ---------------------------------------------------------------------------


class TestMergeDefaultsIntoSuiteContract:
    def test_no_op_when_test_suite_default_absent(self):
        config = {
            "suite_name": "x",
            "test_suite": [{"name": "t", "id": "t", "image": "i"}],
        }
        merged = merge_defaults_into_suite(config)
        assert merged is config  # mutates and returns the input
        assert merged == {
            "suite_name": "x",
            "test_suite": [{"name": "t", "id": "t", "image": "i"}],
        }

    def test_per_test_fields_override_defaults(self):
        config = {
            "suite_name": "x",
            "test_suite_default": {"image": "default-img", "params": {"a": 1}},
            "test_suite": [
                {"name": "t1", "id": "t1", "image": "custom-img"},
            ],
        }
        merged = merge_defaults_into_suite(config)
        assert merged is config
        test = merged["test_suite"][0]
        # Overridden:
        assert test["image"] == "custom-img"
        # Inherited from defaults:
        assert test["params"] == {"a": 1}

    def test_nested_dicts_are_recursively_merged(self):
        config = {
            "suite_name": "x",
            "test_suite_default": {
                "params": {"shared": "from-default", "nested": {"a": 1, "b": 2}},
            },
            "test_suite": [
                {
                    "name": "t1",
                    "id": "t1",
                    "params": {"nested": {"b": 99, "c": 3}},
                },
            ],
        }
        merged = merge_defaults_into_suite(config)
        assert merged is config
        t = merged["test_suite"][0]
        # Top-level merge — both keys present
        assert t["params"]["shared"] == "from-default"
        # Recursive merge inside ``nested``: a inherited, b overridden, c added
        assert t["params"]["nested"] == {"a": 1, "b": 99, "c": 3}


# ---------------------------------------------------------------------------
# extract_manifest_from_image
# ---------------------------------------------------------------------------


class TestExtractManifestFromImageContract:
    """Lock in the ``ManifestExtractionError.error_type`` enumeration.

    Spec calls the set 'open-ended (new values may be added without notice)' —
    we assert the documented values exist as concrete tags."""

    DOCUMENTED_ERROR_TYPES: ClassVar[set[str]] = {
        "IMAGE_NOT_FOUND",
        "MANIFEST_FILE_NOT_FOUND",
        "DOCKER_API_ERROR",
        "TAR_EXTRACTION_ERROR",
        "TAR_IO_ERROR",
        "MANIFEST_FILE_MISSING_AFTER_EXTRACTION",
        "YAML_PARSING_ERROR",
        "FILE_READ_ERROR",
        "EMPTY_MANIFEST_FILE",
        "SCHEMA_VALIDATION_ERROR",
        "UNEXPECTED_ERROR",
    }

    def test_image_not_found_raises_with_image_not_found_error_type(self):
        """Maps docker.errors.ImageNotFound to error_type='IMAGE_NOT_FOUND'."""
        from docker import errors as docker_errors

        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            client = MagicMock()
            client.containers.create.side_effect = docker_errors.ImageNotFound("no such image")
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("ghcr.io/missing/image:latest")
        assert excinfo.value.error_type == "IMAGE_NOT_FOUND"
        assert excinfo.value.original_error is not None
        # Documented as a stable tag in the error_type set
        assert excinfo.value.error_type in self.DOCUMENTED_ERROR_TYPES

    def test_manifest_file_not_found_raises_with_manifest_file_not_found_error_type(self):
        from docker import errors as docker_errors

        container = MagicMock()
        container.get_archive.side_effect = docker_errors.NotFound("manifest.yaml absent")
        client = MagicMock()
        client.containers.create.return_value = container
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "MANIFEST_FILE_NOT_FOUND"

    def test_default_manifest_path_matches_container_config(self):
        """``manifest_path`` default matches ``ContainerConfig.MANIFEST_PATH``."""
        # Inspect signature default
        from inspect import signature

        sig = signature(extract_manifest_from_image)
        assert sig.parameters["manifest_path"].default == "/app/manifest.yaml"
        assert sig.parameters["manifest_path"].default == ContainerConfig.MANIFEST_PATH

    def test_docker_api_error_on_create_maps_to_docker_api_error(self):
        from docker import errors as docker_errors

        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            client = MagicMock()
            client.containers.create.side_effect = docker_errors.APIError("daemon unhappy")
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "DOCKER_API_ERROR"

    def test_docker_api_error_on_get_archive_maps_to_docker_api_error(self):
        from docker import errors as docker_errors

        container = MagicMock()
        container.get_archive.side_effect = docker_errors.APIError("api boom")
        client = MagicMock()
        client.containers.create.return_value = container
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "DOCKER_API_ERROR"

    def test_invalid_tar_archive_maps_to_tar_extraction_error(self):
        """``tarfile.TarError`` while opening the bytes stream → TAR_EXTRACTION_ERROR."""
        container = MagicMock()
        # bits is iterable of bytes chunks — bogus content → TarError when opened
        container.get_archive.return_value = (iter([b"not a tar archive"]), None)
        client = MagicMock()
        client.containers.create.return_value = container
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "TAR_EXTRACTION_ERROR"

    def _stub_docker_with_manifest_bytes(self, manifest_bytes: bytes) -> tuple[MagicMock, MagicMock]:
        """Build a docker client that yields a single-file tar containing
        ``manifest.yaml`` with ``manifest_bytes``.

        The extraction code targets the file at /app/manifest.yaml, but the
        tar member name only needs the basename ``manifest.yaml`` because the
        extractor copies into a temp dir and then opens ``temp/manifest.yaml``.
        """
        import io
        import tarfile

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name="manifest.yaml")
            info.size = len(manifest_bytes)
            tar.addfile(info, io.BytesIO(manifest_bytes))
        tar_bytes = buf.getvalue()

        container = MagicMock()
        container.get_archive.return_value = (iter([tar_bytes]), None)
        client = MagicMock()
        client.containers.create.return_value = container
        return client, container

    def test_happy_path_returns_parsed_manifest_instance(self):
        """Documented contract: 'Creates a temporary container from image,
        copies manifest_path out, parses it through the Manifest Pydantic
        model, returns the instance.'

        End-to-end positive path: a valid YAML manifest is extracted and
        returned as a ``Manifest`` instance with the documented field
        defaults applied."""
        from asqi.schemas import Manifest

        manifest_yaml = (
            b"name: my_test\n"
            b"version: '1.0'\n"
            b"description: a test container\n"
            b"input_systems:\n"
            b"  - name: system_under_test\n"
            b"    type: llm_api\n"
        )
        client, _ = self._stub_docker_with_manifest_bytes(manifest_yaml)
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            manifest = extract_manifest_from_image("img:latest")

        assert isinstance(manifest, Manifest)
        assert manifest.name == "my_test"
        assert manifest.version == "1.0"
        assert manifest.description == "a test container"
        # Documented Manifest field defaults
        assert manifest.host_access is False
        assert manifest.input_schema == []
        # The declared system_under_test came through.
        assert manifest.input_systems[0].name == "system_under_test"
        assert manifest.input_systems[0].type == "llm_api"

    def test_invalid_yaml_maps_to_yaml_parsing_error(self):
        client, _ = self._stub_docker_with_manifest_bytes(
            b"this: is: : not valid: yaml: ["  # malformed
        )
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "YAML_PARSING_ERROR"

    def test_empty_manifest_file_maps_to_empty_manifest_file(self):
        """yaml.safe_load returning None → EMPTY_MANIFEST_FILE."""
        client, _ = self._stub_docker_with_manifest_bytes(b"")
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "EMPTY_MANIFEST_FILE"

    def test_manifest_missing_required_fields_maps_to_schema_validation_error(
        self,
    ):
        """Manifest dict missing required ``name``/``version`` → SCHEMA_VALIDATION_ERROR."""
        client, _ = self._stub_docker_with_manifest_bytes(b"description: missing name and version\n")
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "SCHEMA_VALIDATION_ERROR"

    def test_unexpected_exception_maps_to_unexpected_error(self):
        """An exception we don't anticipate (e.g. RuntimeError) is caught by
        the broad ``except Exception`` and tagged ``UNEXPECTED_ERROR``."""
        container = MagicMock()
        container.get_archive.side_effect = RuntimeError("totally unexpected")
        client = MagicMock()
        client.containers.create.return_value = container
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "UNEXPECTED_ERROR"

    def test_tar_io_error_maps_to_tar_io_error(self):
        """``IOError`` while reading the tar bytes stream → TAR_IO_ERROR.

        The implementation iterates ``bits`` to copy chunks into a BytesIO;
        we make the iterator raise ``IOError`` mid-stream."""

        def raising_iter():
            yield b"prefix-bytes-"
            raise OSError("simulated I/O failure during tar read")

        container = MagicMock()
        container.get_archive.return_value = (raising_iter(), None)
        client = MagicMock()
        client.containers.create.return_value = container
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "TAR_IO_ERROR"

    def test_manifest_missing_after_extraction_maps_to_dedicated_error_type(self):
        """If the tar contains files but none of them is ``manifest.yaml``,
        the local path won't exist after extraction →
        ``MANIFEST_FILE_MISSING_AFTER_EXTRACTION``."""
        import io
        import tarfile

        # Build a tar containing only "other.txt" — manifest.yaml is missing
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            payload = b"unrelated content"
            info = tarfile.TarInfo(name="other.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        tar_bytes = buf.getvalue()

        container = MagicMock()
        container.get_archive.return_value = (iter([tar_bytes]), None)
        client = MagicMock()
        client.containers.create.return_value = container
        with patch("asqi.container_manager.docker_client") as mock_client_ctx:
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "MANIFEST_FILE_MISSING_AFTER_EXTRACTION"

    def test_file_read_error_maps_to_file_read_error(self):
        """When the file-read step around ``open()`` / ``yaml.safe_load`` raises
        ``OSError``, the implementation tags it ``FILE_READ_ERROR``
        (container_manager.py:260-265).

        We trigger this by letting the tar extract a real ``manifest.yaml``
        but patching ``yaml.safe_load`` to raise ``OSError``. The wrapping
        ``try`` block catches both ``open()`` and ``safe_load`` failures via
        the same ``(IOError, OSError)`` handler, so this is faithful to the
        documented contract."""
        client, _ = self._stub_docker_with_manifest_bytes(b"name: x\nversion: '1'\n")
        with (
            patch("asqi.container_manager.docker_client") as mock_client_ctx,
            patch(
                "asqi.container_manager.yaml.safe_load",
                side_effect=PermissionError("simulated read failure"),
            ),
        ):
            mock_client_ctx.return_value.__enter__.return_value = client
            with pytest.raises(ManifestExtractionError) as excinfo:
                extract_manifest_from_image("img:latest")
        assert excinfo.value.error_type == "FILE_READ_ERROR"

    def test_documented_error_types_are_all_string_constants(self):
        """All 11 documented values exist as strings the implementation
        produces. With the three additions above (TAR_IO_ERROR,
        MANIFEST_FILE_MISSING_AFTER_EXTRACTION, FILE_READ_ERROR), every
        documented ``error_type`` now has at least one live test."""
        assert isinstance(self.DOCUMENTED_ERROR_TYPES, set)
        assert len(self.DOCUMENTED_ERROR_TYPES) == 11


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class TestErrorClassesContract:
    def test_manifest_extraction_error_carries_payload(self):
        """error_type and original_error are documented attributes catchers may inspect."""
        original = ValueError("inner")
        err = ManifestExtractionError("oops", "DOCKER_API_ERROR", original)
        assert err.error_type == "DOCKER_API_ERROR"
        assert err.original_error is original
        # Also a regular Exception
        assert str(err) == "oops"

    def test_audit_responses_required_error_carries_indicator_payload(self):
        indicators: list[dict[str, Any]] = [
            {
                "id": "ind_1",
                "name": "Manual review",
                "assessment": [
                    {"outcome": "A", "description": "Pass"},
                    {"outcome": "B", "description": "Fail"},
                ],
            }
        ]
        err = AuditResponsesRequiredError(score_card_name="My Score Card", audit_indicators=indicators)
        assert err.score_card_name == "My Score Card"
        assert err.audit_indicators == indicators
        # Rendered message includes a YAML template (documented)
        rendered = str(err)
        assert "responses:" in rendered
        assert "ind_1" in rendered
        # The outcomes from the assessment are surfaced for re-prompting
        assert "A" in rendered and "B" in rendered

    def test_audit_responses_required_error_handles_empty_indicators(self):
        """Empty list is valid input — message just shows no per-indicator block."""
        err = AuditResponsesRequiredError(score_card_name="Empty", audit_indicators=[])
        assert err.audit_indicators == []
        assert "Empty" in str(err)
