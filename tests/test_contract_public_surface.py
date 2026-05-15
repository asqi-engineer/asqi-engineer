# pyright: reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportCallIssue=false
"""Contract tests for the public-symbol inventory and secondary
uncaught-exception surface.

The function-level contract tests (`test_contract_workflows.py`,
`test_contract_sdk.py`, ...) cover *behaviour*. This file pins the
*inventory* itself: every documented module (host-side and in-container
SDK) must export the documented set of symbols, and the documented
secondary exception classes must remain importable from ``asqi.errors``.

A regression where a public symbol is renamed, hidden behind ``_``, or
moved to a different module would fail the inventory checks here even if
each surviving symbol still passes its own behaviour tests.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable

import pytest


def _assert_module_exports(module_name: str, expected_symbols: Iterable[str]) -> None:
    mod = importlib.import_module(module_name)
    for sym in expected_symbols:
        assert hasattr(mod, sym), (
            f"Public-surface inventory: `{module_name}` must export `{sym}` as part of the documented public surface"
        )
        # Also pin: not a private/dunder; the spec lists it as a public name.
        assert not sym.startswith("_"), f"Public symbol `{sym}` must not be private"


# ---------------------------------------------------------------------------
# Host-side library API inventory
# ---------------------------------------------------------------------------


class TestHostSideInventory:
    """Symbols an embedding application imports from `asqi.*` to orchestrate
    workloads. The documented public surface fixes these per-module."""

    def test_asqi_workflow_exports_documented_entry_points(self):
        _assert_module_exports(
            "asqi.workflow",
            [
                "run_test_suite_workflow",
                "evaluate_score_cards_workflow",
                "run_data_generation_workflow",
            ],
        )

    def test_asqi_config_exports_documented_types(self):
        _assert_module_exports(
            "asqi.config",
            [
                "ContainerConfig",
                "ExecutorConfig",
                "ExecutionMode",
                "merge_defaults_into_suite",
            ],
        )

    def test_asqi_backends_exports_docker_backend(self):
        _assert_module_exports(
            "asqi.backends",
            ["ContainerBackend", "DockerBackend"],
        )

    def test_asqi_schemas_exports_public_models(self):
        _assert_module_exports(
            "asqi.schemas",
            [
                "Manifest",
                "SystemInput",
                "InputParameter",
                "InputDataset",
                "DatasetType",
                "DatasetFeature",
            ],
        )

    def test_asqi_errors_exports_documented_host_side_errors(self):
        # These two are the public, structured-payload error classes callers
        # are documented to catch on the host side.
        _assert_module_exports(
            "asqi.errors",
            [
                "ManifestExtractionError",
                "AuditResponsesRequiredError",
            ],
        )


# ---------------------------------------------------------------------------
# In-container SDK inventory
# ---------------------------------------------------------------------------


class TestInContainerSdkInventory:
    """Symbols imported by code running *inside* test/SDG container images.
    A break here ripples through every dependent container image, so the
    inventory bar is at least as strict as the host-side bar."""

    def test_asqi_datasets_exports_documented_symbols(self):
        _assert_module_exports(
            "asqi.datasets",
            [
                "load_hf_dataset",
                "validate_dataset_features",
                "Dataset",  # re-exported from HuggingFace `datasets`
            ],
        )

    def test_asqi_loaders_exports_load_test_cases(self):
        _assert_module_exports(
            "asqi.loaders",
            ["load_test_cases"],
        )

    def test_asqi_response_schemas_exports_documented_symbols(self):
        _assert_module_exports(
            "asqi.response_schemas",
            [
                "ContainerOutput",
                "GeneratedDataset",
                "GeneratedReport",
                "validate_container_output",
            ],
        )

    def test_asqi_utils_exports_get_openai_tracking_kwargs(self):
        _assert_module_exports(
            "asqi.utils",
            ["get_openai_tracking_kwargs"],
        )

    def test_asqi_output_exports_parse_container_json_output(self):
        _assert_module_exports(
            "asqi.output",
            ["parse_container_json_output"],
        )

    def test_asqi_rag_response_schema_exports_documented_symbols(self):
        _assert_module_exports(
            "asqi.rag_response_schema",
            ["RAGCitation", "validate_rag_response"],
        )

    def test_asqi_schemas_manifest_importable_from_container_code(self):
        # Container code (notably SDG entrypoints) is expected to read
        # manifest schemas from `asqi.schemas` directly.
        _assert_module_exports("asqi.schemas", ["Manifest"])


# ---------------------------------------------------------------------------
# Secondary uncaught-exception inventory
# ---------------------------------------------------------------------------


class TestSecondaryExceptionInventory:
    """Exception classes that may propagate uncaught from the documented
    entry points. They carry no extra attributes beyond a plain
    ``Exception``, but their *identity* (class name + module) is part of
    the contract — callers may catch them by name.

    Pin that each is importable from `asqi.errors`, is a subclass of
    `Exception`, and round-trips through raise/except.
    """

    @pytest.mark.parametrize(
        "exc_name",
        [
            "MissingImageError",
            "MountExtractionError",
            "DuplicateIDError",
            "MissingIDFieldError",
        ],
    )
    def test_secondary_exception_class_is_importable(self, exc_name):
        from asqi import errors

        assert hasattr(errors, exc_name), (
            f"`asqi.errors.{exc_name}` must remain importable; "
            f"renaming or hiding it is a breaking change for callers that "
            f"catch by class name"
        )
        cls = getattr(errors, exc_name)
        assert isinstance(cls, type)
        assert issubclass(cls, Exception)

    def test_missing_image_error_round_trips_through_raise(self):
        from asqi.errors import MissingImageError

        with pytest.raises(MissingImageError) as exc_info:
            raise MissingImageError("image not found")
        assert "image not found" in str(exc_info.value)

    def test_mount_extraction_error_round_trips_through_raise(self):
        from asqi.errors import MountExtractionError

        with pytest.raises(MountExtractionError) as exc_info:
            raise MountExtractionError("bad mount")
        assert "bad mount" in str(exc_info.value)

    def test_duplicate_id_error_carries_duplicate_dict_payload(self):
        """``DuplicateIDError`` is documented as a 'plain exception with no
        extra attributes' from the host-API perspective — it isn't part of
        the host surface. But the implementation does carry a
        ``duplicate_dict`` attribute that the CLI relies on for its rendered
        message. Pin both: the class is catchable, AND the documented
        attribute is preserved."""
        from asqi.errors import DuplicateIDError

        payload = {
            "t_dup": {
                "id": "dup",
                "config_type": "test_suite",
                "occurrences": [
                    {"location": "x.yaml", "test_suite_name": "s", "test_name": "t"}
                ],
            }
        }
        with pytest.raises(DuplicateIDError) as exc_info:
            raise DuplicateIDError(payload)
        assert exc_info.value.duplicate_dict == payload
        # The rendered message must mention the duplicate id.
        assert "dup" in str(exc_info.value)

    def test_missing_id_field_error_round_trips_through_raise(self):
        from asqi.errors import MissingIDFieldError

        with pytest.raises(MissingIDFieldError) as exc_info:
            raise MissingIDFieldError("missing id field in test")
        assert "missing id field" in str(exc_info.value)

    def test_each_secondary_exception_is_a_distinct_class(self):
        """Catchers may write ``except DuplicateIDError`` expecting NOT to
        also catch ``MissingIDFieldError``. Pin that the four classes are
        independent (no inheritance between them)."""
        from asqi.errors import (
            DuplicateIDError,
            MissingIDFieldError,
            MissingImageError,
            MountExtractionError,
        )

        classes = [
            MissingImageError,
            MountExtractionError,
            DuplicateIDError,
            MissingIDFieldError,
        ]
        for i, a in enumerate(classes):
            for j, b in enumerate(classes):
                if i == j:
                    continue
                assert not issubclass(a, b), (
                    f"{a.__name__} must not subclass {b.__name__} — catchers rely on these being independent"
                )
