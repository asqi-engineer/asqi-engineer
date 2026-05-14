# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false
"""Contract tests for the in-container SDK.

Locks in the contract that test/SDG containers depend on at runtime:

- ``asqi.datasets`` — ``Dataset`` re-export, ``load_hf_dataset``,
  ``validate_dataset_features``
- ``asqi.response_schemas`` — ``ContainerOutput``, ``GeneratedReport``,
  ``GeneratedDataset``, ``validate_container_output``
- ``asqi.utils.get_openai_tracking_kwargs``
- ``asqi.output.parse_container_json_output``
- ``asqi.rag_response_schema`` — ``RAGCitation``, ``RAGContext``,
  ``validate_rag_response`` (and the documented ``__all__``)
- ``asqi.loaders.load_test_cases``

A breaking change to any of these ripples through every container image."""

from __future__ import annotations

import json

import asqi.rag_response_schema as rag_schema_module
import pytest
from asqi.output import parse_container_json_output
from asqi.rag_response_schema import RAGCitation, RAGContext, validate_rag_response
from asqi.response_schemas import (
    ColumnMetadata,
    ContainerOutput,
    DatasetMetadata,
    GeneratedDataset,
    GeneratedReport,
    validate_container_output,
)
from asqi.utils import get_openai_tracking_kwargs
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# asqi.datasets — public re-exports
# ---------------------------------------------------------------------------


class TestDatasetsModuleSurface:
    """The Dataset symbol must be importable from ``asqi.datasets``."""

    def test_dataset_class_reexported_from_asqi_datasets(self):
        from asqi.datasets import Dataset  # re-exported from HuggingFace

        # It is the HuggingFace Dataset, not a wrapper.
        from datasets import Dataset as HFDataset

        assert Dataset is HFDataset

    def test_load_hf_dataset_and_validate_features_are_public(self):
        # Importable: the contract says they are part of the in-container SDK.
        from asqi.datasets import load_hf_dataset, validate_dataset_features

        assert callable(load_hf_dataset)
        assert callable(validate_dataset_features)


class TestValidateDatasetFeaturesContract:
    """Behavior contract for ``validate_dataset_features``.

    Documented to raise ``ValueError`` aggregating missing features and type
    mismatches into a single multi-line error, and to no-op on streaming
    datasets (where ``column_names`` / ``features`` are ``None``).
    """

    def _make_dataset(self, **cols):
        from datasets import Dataset

        return Dataset.from_dict(cols)

    def test_no_op_on_streaming_dataset(self):
        """No-op when column_names / features are None (per docstring)."""
        from unittest.mock import MagicMock

        from asqi.datasets import validate_dataset_features
        from asqi.schemas import DatasetFeature

        streaming = MagicMock()
        streaming.column_names = None
        streaming.features = None
        # Should not raise even with required features specified
        validate_dataset_features(
            streaming, [DatasetFeature(name="q", dtype="string", required=True)]
        )

    def test_missing_required_feature_raises_value_error(self):
        from asqi.datasets import validate_dataset_features
        from asqi.schemas import DatasetFeature

        ds = self._make_dataset(query=["a", "b"])
        with pytest.raises(ValueError) as excinfo:
            validate_dataset_features(
                ds,
                [DatasetFeature(name="answer", dtype="string", required=True)],
                dataset_name="rag_eval",
            )
        msg = str(excinfo.value)
        # Documented: aggregates into one multi-line error mentioning the dataset name
        assert "rag_eval" in msg
        assert "answer" in msg

    def test_optional_missing_feature_not_an_error(self):
        from asqi.datasets import validate_dataset_features
        from asqi.schemas import DatasetFeature

        ds = self._make_dataset(query=["a"])
        # Required defaults to False on DatasetFeature → optional, no raise
        validate_dataset_features(ds, [DatasetFeature(name="answer", dtype="string")])

    def test_dtype_mismatch_raises_value_error(self):
        from asqi.datasets import validate_dataset_features
        from asqi.schemas import DatasetFeature

        ds = self._make_dataset(score=[1, 2, 3])  # int64 in HF
        with pytest.raises(ValueError) as excinfo:
            validate_dataset_features(
                ds,
                [DatasetFeature(name="score", dtype="string", required=True)],
                dataset_name="d",
            )
        # The error mentions both expected and got dtypes per implementation
        assert "string" in str(excinfo.value)


class TestLoadHfDatasetContract:
    """Behavior contract for ``load_hf_dataset``."""

    def test_invalid_dataset_config_dict_raises_validation_error(self):
        """A dict failing Pydantic validation → ``ValidationError``."""
        from asqi.datasets import load_hf_dataset

        with pytest.raises(ValidationError):
            load_hf_dataset({"not": "valid"})

    def test_local_csv_load_with_input_mount_path(self, tmp_path):
        """Local mode with builder_name+data_files; ``input_mount_path`` is
        prepended to relative paths."""
        from asqi.datasets import load_hf_dataset

        mount = tmp_path / "input"
        mount.mkdir()
        (mount / "rows.csv").write_text("query,answer\nq1,a1\nq2,a2\n")

        ds = load_hf_dataset(
            {
                "type": "huggingface",
                "loader_params": {
                    "builder_name": "csv",
                    "data_files": "rows.csv",
                },
            },
            input_mount_path=mount,
        )
        # Returned dataset exposes the documented HF surface
        assert "query" in ds.column_names
        assert "answer" in ds.column_names
        assert ds.num_rows == 2

    def test_mapping_renames_columns(self, tmp_path):
        """``mapping`` on the config renames columns after load."""
        from asqi.datasets import load_hf_dataset

        f = tmp_path / "rows.csv"
        f.write_text("q,a\nq1,a1\n")

        ds = load_hf_dataset(
            {
                "type": "huggingface",
                "loader_params": {
                    "builder_name": "csv",
                    "data_files": str(f),
                },
                "mapping": {"q": "query", "a": "answer"},
            },
        )
        assert "query" in ds.column_names
        assert "answer" in ds.column_names
        assert "q" not in ds.column_names

    def test_hub_path_dispatches_to_hub_loader(self):
        """When ``loader_params.hub_path`` is set the loader dispatches to
        the Hub branch (``_load_from_hub``) instead of local.

        We patch ``_load_from_hub`` to confirm dispatch *and* return a non-
        IterableDataset so feature validation (when supplied) runs."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        from asqi.datasets import load_hf_dataset
        from datasets import Dataset

        fake_ds = Dataset.from_dict({"query": ["q1"], "answer": ["a1"]})
        with (
            _patch("asqi.datasets._load_from_hub", return_value=fake_ds) as mock_hub,
            _patch("asqi.datasets._load_from_local") as mock_local,
        ):
            ds = load_hf_dataset(
                {
                    "type": "huggingface",
                    "loader_params": {"hub_path": "org/dataset"},
                }
            )
        # Hub branch was used; local branch was not.
        mock_hub.assert_called_once()
        mock_local.assert_not_called()
        # The hub branch's return value flows through unchanged.
        assert ds is fake_ds
        # Sanity: return type matches the documented surface.
        assert isinstance(ds, Dataset) or isinstance(ds, MagicMock)

    def test_streaming_iterable_dataset_skips_feature_validation(self):
        """``load_hf_dataset`` skips feature validation for
        ``IterableDataset`` (streaming) loads — ``column_names`` /
        ``features`` are not materialized for streaming.

        Currently this branch is only proven via ``MagicMock`` substitution
        on ``validate_dataset_features``. Pin it via a real ``IterableDataset``
        returned from ``_load_from_hub`` so the ``isinstance(...,
        IterableDataset)`` check itself is exercised. With ``expected_features``
        supplied, the call must NOT raise even though the iterable carries no
        column metadata."""
        from unittest.mock import patch as _patch

        from asqi.datasets import load_hf_dataset
        from asqi.schemas import DatasetFeature
        from datasets import Dataset, IterableDataset

        # Build a real IterableDataset by streaming an in-memory Dataset.
        materialized = Dataset.from_dict({"query": ["q1"], "answer": ["a1"]})
        streaming_ds = materialized.to_iterable_dataset()
        # Sanity: it really is an IterableDataset (the type the skip branch
        # checks for).
        assert isinstance(streaming_ds, IterableDataset)

        captured = {}

        def fake_validate(dataset, expected_features, dataset_name="dataset"):
            captured["called"] = True

        with (
            _patch(
                "asqi.datasets._load_from_hub", return_value=streaming_ds
            ) as mock_hub,
            _patch(
                "asqi.datasets.validate_dataset_features",
                side_effect=fake_validate,
            ),
        ):
            ds = load_hf_dataset(
                {
                    "type": "huggingface",
                    "loader_params": {
                        "hub_path": "org/dataset",
                        "streaming": True,
                    },
                },
                expected_features=[
                    DatasetFeature(name="answer", dtype="string", required=True)
                ],
            )

        # Hub branch was used and the iterable flowed through unchanged.
        mock_hub.assert_called_once()
        assert ds is streaming_ds
        # validate_dataset_features must NOT have been called — the documented
        # skip branch fires for IterableDataset even when expected_features is
        # supplied.
        assert "called" not in captured

    def test_expected_features_validation_runs_on_load(self, tmp_path):
        """When ``expected_features`` is supplied, validation runs after load
        and raises ``ValueError`` on mismatch."""
        from asqi.datasets import load_hf_dataset
        from asqi.schemas import DatasetFeature

        f = tmp_path / "rows.csv"
        f.write_text("query\nq1\n")

        with pytest.raises(ValueError):
            load_hf_dataset(
                {
                    "type": "huggingface",
                    "loader_params": {
                        "builder_name": "csv",
                        "data_files": str(f),
                    },
                },
                expected_features=[
                    DatasetFeature(name="answer", dtype="string", required=True)
                ],
            )


# ---------------------------------------------------------------------------
# asqi.response_schemas — ContainerOutput rules
# ---------------------------------------------------------------------------


class TestContainerOutputContract:
    def test_supports_results_field_recommended(self):
        out = ContainerOutput(results={"success": True})
        assert out.get_results() == {"success": True}

    def test_supports_test_results_legacy_field(self):
        out = ContainerOutput(test_results={"success": True})
        assert out.get_results() == {"success": True}

    def test_results_takes_precedence_over_test_results(self):
        out = ContainerOutput(results={"new": 1}, test_results={"old": 2})
        assert out.get_results() == {"new": 1}

    def test_get_results_returns_empty_dict_when_neither_set(self):
        out = ContainerOutput()
        assert out.get_results() == {}

    def test_extra_fields_allowed_for_forward_compatibility(self):
        # ``model_config = {"extra": "allow"}`` is the documented contract
        ContainerOutput(results={"success": True}, future_field="hello")

    def test_empty_results_dict_rejected(self):
        with pytest.raises(ValidationError):
            ContainerOutput(results={})

    def test_empty_test_results_dict_rejected(self):
        with pytest.raises(ValidationError):
            ContainerOutput(test_results={})

    def test_validate_container_output_requires_at_least_one_results_field(self):
        """Stricter than the bare constructor."""
        # Constructor accepts an empty body
        ContainerOutput()
        # validate_container_output rejects it
        with pytest.raises(ValueError):
            validate_container_output({})

    def test_validate_container_output_accepts_results(self):
        out = validate_container_output({"results": {"success": True}})
        assert isinstance(out, ContainerOutput)
        assert out.get_results() == {"success": True}

    def test_validate_container_output_accepts_test_results_alias(self):
        out = validate_container_output({"test_results": {"success": True}})
        assert out.get_results() == {"success": True}

    def test_validate_container_output_does_not_enforce_success_field(self):
        """Documented carve-out: 'the schema's docstring requires it to
        "contain at least 'success' field," but the validator only rejects
        an empty ``{}``.' Pin the explicit accept-without-success case so
        the validator cannot silently tighten to require ``success``.

        Containers that emit only domain-specific keys (e.g. ``{"score": 0.9,
        "details": ...}``) currently validate; tightening to require
        ``success`` would break them."""
        # Non-empty results without ``success`` — accepted by the validator.
        out = validate_container_output({"results": {"score": 0.9, "details": "ok"}})
        assert out.get_results() == {"score": 0.9, "details": "ok"}
        # Same for the legacy ``test_results`` alias.
        out2 = validate_container_output({"test_results": {"latency_ms": 42}})
        assert out2.get_results() == {"latency_ms": 42}


class TestGeneratedReportContract:
    def test_minimum_required_fields(self):
        rep = GeneratedReport(
            report_name="r", report_type="html", report_path="/output/r.html"
        )
        assert rep.metadata is None

    @pytest.mark.parametrize("report_type", ["html", "pdf", "json"])
    def test_documented_types_accepted(self, report_type):
        GeneratedReport(
            report_name="x",
            report_type=report_type,
            report_path="/output/x",
        )

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            GeneratedReport(report_name="x", report_type="csv", report_path="/output/x")

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_report_name_rejected(self, blank):
        with pytest.raises(ValidationError):
            GeneratedReport(
                report_name=blank, report_type="html", report_path="/output/x"
            )

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_report_path_rejected(self, blank):
        with pytest.raises(ValidationError):
            GeneratedReport(report_name="r", report_type="html", report_path=blank)


class TestGeneratedDatasetContract:
    @pytest.mark.parametrize("dataset_type", ["huggingface", "pdf", "txt"])
    def test_documented_types_accepted(self, dataset_type):
        GeneratedDataset(
            dataset_name="x",
            dataset_type=dataset_type,
            dataset_path="/output/x",
        )

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            GeneratedDataset(
                dataset_name="x", dataset_type="csv", dataset_path="/output/x"
            )

    def test_metadata_accepts_dataset_metadata_object(self):
        meta = DatasetMetadata(columns=[{"name": "a", "dtype": "string"}], row_count=10)
        ds = GeneratedDataset(
            dataset_name="x",
            dataset_type="huggingface",
            dataset_path="/output/x",
            metadata=meta,
        )
        assert isinstance(ds.metadata, DatasetMetadata)

    def test_metadata_accepts_plain_dict_for_backwards_compat(self):
        ds = GeneratedDataset(
            dataset_name="x",
            dataset_type="pdf",
            dataset_path="/output/x.pdf",
            metadata={"anything": "ok"},
        )
        assert ds.metadata == {"anything": "ok"}


class TestColumnMetadataContract:
    """``ColumnMetadata`` is the structured shape that
    ``DatasetMetadata.columns`` accepts. ``name`` and ``dtype`` are required;
    ``description`` defaults to ``None``."""

    def test_required_fields_only(self):
        col = ColumnMetadata(name="query", dtype="string")
        assert col.name == "query"
        assert col.dtype == "string"
        assert col.description is None

    def test_description_defaults_to_none(self):
        col = ColumnMetadata(name="q", dtype="string")
        assert col.description is None

    @pytest.mark.parametrize("missing", ["name", "dtype"])
    def test_missing_required_field_rejected(self, missing):
        kwargs = {"name": "q", "dtype": "string"}
        del kwargs[missing]
        with pytest.raises(ValidationError):
            ColumnMetadata(**kwargs)


class TestDatasetMetadataConstraints:
    """``row_count`` and ``size_bytes`` carry ``ge=0`` constraints and
    ``columns`` is a required list."""

    def test_minimum_required_fields(self):
        meta = DatasetMetadata(
            columns=[ColumnMetadata(name="q", dtype="string")], row_count=0
        )
        assert meta.row_count == 0
        assert meta.size_bytes is None

    def test_row_count_negative_rejected(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(
                columns=[ColumnMetadata(name="q", dtype="string")],
                row_count=-1,
            )

    def test_size_bytes_negative_rejected(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(
                columns=[ColumnMetadata(name="q", dtype="string")],
                row_count=10,
                size_bytes=-5,
            )

    def test_size_bytes_zero_accepted(self):
        meta = DatasetMetadata(
            columns=[ColumnMetadata(name="q", dtype="string")],
            row_count=10,
            size_bytes=0,
        )
        assert meta.size_bytes == 0

    def test_columns_required(self):
        with pytest.raises(ValidationError):
            DatasetMetadata(row_count=0)  # missing columns


# ---------------------------------------------------------------------------
# asqi.utils.get_openai_tracking_kwargs
# ---------------------------------------------------------------------------


class TestGetOpenaiTrackingKwargsContract:
    def test_none_metadata_returns_no_user_with_empty_tags(self):
        kwargs = get_openai_tracking_kwargs(None)
        assert "user" not in kwargs
        assert "extra_body" not in kwargs

    def test_user_id_only_set_when_truthy(self):
        kwargs = get_openai_tracking_kwargs({"user_id": "u-1"})
        assert kwargs["user"] == "u-1"

        # Empty user_id => no "user" key
        kwargs = get_openai_tracking_kwargs({"user_id": ""})
        assert "user" not in kwargs

    def test_tags_dict_converted_to_list_of_key_value_strings(self):
        kwargs = get_openai_tracking_kwargs(
            {"tags": {"job_id": "j-1", "experiment": "e-1"}}
        )
        tags = kwargs["extra_body"]["metadata"]["tags"]
        assert sorted(tags) == sorted(["job_id:j-1", "experiment:e-1"])

    def test_other_top_level_metadata_keys_preserved(self):
        kwargs = get_openai_tracking_kwargs(
            {"user_id": "u", "custom_field": "v", "tags": {}}
        )
        meta = kwargs["extra_body"]["metadata"]
        assert meta["custom_field"] == "v"
        # user_id and tags are NOT under "metadata" — user_id is at top level,
        # tags is the converted list
        assert "user_id" not in meta

    def test_safe_when_tags_not_a_dict(self):
        # Should not crash; non-dict tags are converted defensively.
        kwargs = get_openai_tracking_kwargs({"tags": "raw-string"})
        assert "tags" in kwargs["extra_body"]["metadata"]

    def test_execution_metadata_pydantic_input_accepted(self):
        """Documented signature is
        ``metadata: Optional[Union[Dict[str, Any], ExecutionMetadata]] = None``.

        The dict arm is covered above; pin the ``ExecutionMetadata`` arm so a
        regression that drops the ``isinstance(metadata, ExecutionMetadata)``
        branch (utils.py:64) — and silently treats the model as a non-dict —
        is caught. Containers receive the metadata as the typed Pydantic model
        in some code paths, so this arm is part of the wire contract."""
        from asqi.schemas import ExecutionMetadata, ExecutionTags

        meta = ExecutionMetadata(
            tags=ExecutionTags(parent_id="parent-1", job_type="test", job_id="job-1"),
            user_id="user-99",
        )
        kwargs = get_openai_tracking_kwargs(meta)

        # Documented output shape: user at top level, tags as ["k:v", ...].
        assert kwargs["user"] == "user-99"
        tags = kwargs["extra_body"]["metadata"]["tags"]
        assert sorted(tags) == sorted(
            ["parent_id:parent-1", "job_type:test", "job_id:job-1"]
        )

    def test_execution_metadata_with_no_user_id_omits_user_key(self):
        """``user`` is only set when ``user_id`` is truthy.
        ``ExecutionMetadata.user_id`` defaults to ``None`` — pin that the
        Pydantic arm honors the same omission rule as the dict arm."""
        from asqi.schemas import ExecutionMetadata, ExecutionTags

        meta = ExecutionMetadata(
            tags=ExecutionTags(parent_id="p", job_type="t", job_id="j"),
        )
        kwargs = get_openai_tracking_kwargs(meta)
        assert "user" not in kwargs

    def test_execution_metadata_extra_fields_preserved(self):
        """``ExecutionMetadata`` allows extras (model_config = {"extra": "allow"}).
        Per the documented output shape, top-level metadata keys other than
        ``user_id`` and ``tags`` land under ``extra_body.metadata``. Pin that
        the dict produced by ``model_dump()`` carries those extras through
        unchanged."""
        from asqi.schemas import ExecutionMetadata, ExecutionTags

        meta = ExecutionMetadata(
            tags=ExecutionTags(parent_id="p", job_type="t", job_id="j"),
            user_id="u",
            experiment_id="exp-42",  # extra field, allowed by model_config
        )
        kwargs = get_openai_tracking_kwargs(meta)
        assert kwargs["extra_body"]["metadata"]["experiment_id"] == "exp-42"


# ---------------------------------------------------------------------------
# asqi.output.parse_container_json_output
# ---------------------------------------------------------------------------


class TestParseContainerJsonOutputContract:
    def test_direct_json_object(self):
        out = parse_container_json_output('{"success": true}')
        assert out == {"success": True}

    def test_trailing_json_after_log_lines(self):
        raw = "log line one\nINFO some progress\n" + json.dumps({"success": True})
        out = parse_container_json_output(raw)
        assert out == {"success": True}

    def test_picks_last_json_block(self):
        raw = json.dumps({"old": 1}) + "\n" + json.dumps({"latest": True})
        out = parse_container_json_output(raw)
        assert out == {"latest": True}

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError, match="Empty container output"):
            parse_container_json_output("")

    def test_whitespace_only_input_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_container_json_output("   \n\t  ")

    def test_no_json_in_output_raises_value_error(self):
        with pytest.raises(ValueError, match="No valid JSON found"):
            parse_container_json_output("plain text only, no JSON")

    def test_no_json_error_includes_100_char_preview(self):
        """The ``ValueError`` message includes a 100-char preview of the
        container output. Catchers / log scrapers may rely on the truncation
        contract (with ``...`` when the output is longer)."""
        long_garbage = "X" * 250  # > 100 chars, no JSON
        with pytest.raises(ValueError) as excinfo:
            parse_container_json_output(long_garbage)
        msg = str(excinfo.value)
        # First 100 chars present
        assert "X" * 100 in msg
        # Truncation marker appended when the output exceeds 100 chars
        assert "..." in msg
        # The full 250-char output is NOT in the message
        assert "X" * 250 not in msg

    def test_no_json_error_omits_ellipsis_when_short(self):
        """When the output is <= 100 chars, no ``...`` truncation marker."""
        short_garbage = "no json here"
        with pytest.raises(ValueError) as excinfo:
            parse_container_json_output(short_garbage)
        msg = str(excinfo.value)
        assert "no json here" in msg
        # Implementation only appends ``...`` when len > 100
        assert "..." not in msg


# ---------------------------------------------------------------------------
# asqi.rag_response_schema
# ---------------------------------------------------------------------------


class TestRagResponseSchemaContract:
    def test_explicit_all_export_list(self):
        """``__all__`` is a stable contract."""
        assert set(rag_schema_module.__all__) == {
            "RAGCitation",
            "RAGContext",
            "validate_rag_response",
        }

    def test_validate_rag_response_happy_path(self):
        response = {
            "choices": [
                {
                    "message": {
                        "context": {
                            "citations": [
                                {
                                    "retrieved_context": "Some text",
                                    "document_id": "doc-1",
                                    "score": 0.9,
                                }
                            ]
                        }
                    }
                }
            ]
        }
        citations = validate_rag_response(response)
        assert len(citations) == 1
        assert isinstance(citations[0], RAGCitation)
        assert citations[0].document_id == "doc-1"

    @pytest.mark.parametrize(
        "broken_response",
        [
            # Empty choices -> IndexError unified to KeyError per documented contract
            {"choices": []},
            # Missing message
            {"choices": [{}]},
            # Missing context
            {"choices": [{"message": {}}]},
            # message is not a dict
            {"choices": [{"message": "string-not-dict"}]},
        ],
    )
    def test_validate_rag_response_unifies_navigation_errors_into_key_error(
        self, broken_response
    ):
        with pytest.raises(KeyError):
            validate_rag_response(broken_response)

    def test_validate_rag_response_validation_error_for_invalid_citation(self):
        """Pydantic ValidationError when the context dict exists but doesn't
        match the schema (e.g. score outside [0, 1])."""
        response = {
            "choices": [
                {
                    "message": {
                        "context": {
                            "citations": [
                                {
                                    "retrieved_context": "Some text",
                                    "score": 1.5,  # > 1.0
                                }
                            ]
                        }
                    }
                }
            ]
        }
        with pytest.raises(ValidationError):
            validate_rag_response(response)

    def test_validate_rag_response_blank_retrieved_context_rejected(self):
        response = {
            "choices": [
                {"message": {"context": {"citations": [{"retrieved_context": ""}]}}}
            ]
        }
        with pytest.raises(ValidationError):
            validate_rag_response(response)

    def test_rag_context_default_citations_empty_list(self):
        ctx = RAGContext()
        assert ctx.citations == []

    def test_validate_rag_response_score_below_zero_rejected(self):
        """``RAGCitation.score`` carries ``ge=0.0, le=1.0``. The upper bound
        is covered above; pin the lower bound symmetrically so a regression
        that drops ``ge=0.0`` (allowing nonsensical negative scores) is
        caught."""
        response = {
            "choices": [
                {
                    "message": {
                        "context": {
                            "citations": [
                                {
                                    "retrieved_context": "Some text",
                                    "score": -0.1,  # < 0.0
                                }
                            ]
                        }
                    }
                }
            ]
        }
        with pytest.raises(ValidationError):
            validate_rag_response(response)

    def test_rag_citation_optional_fields_default_to_none(self):
        """``document_id``, ``score``, and ``source_id`` are all optional.
        Pin the documented default (``None``) so a future tightening that
        makes any of them required is caught."""
        cit = RAGCitation(retrieved_context="some text")
        assert cit.document_id is None
        assert cit.score is None
        assert cit.source_id is None


# ---------------------------------------------------------------------------
# asqi.loaders.load_test_cases
# ---------------------------------------------------------------------------


class _RagCase(BaseModel):
    query: str
    answer: str


class TestLoadTestCasesContract:
    def test_jsonl_happy_path(self, tmp_path):
        from asqi.loaders import load_test_cases

        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            json.dumps({"query": "q1", "answer": "a1"})
            + "\n"
            + json.dumps({"query": "q2", "answer": "a2"})
            + "\n"
        )

        out = list(load_test_cases(str(jsonl), _RagCase))
        assert len(out) == 2
        assert all(isinstance(o, _RagCase) for o in out)
        assert out[0].query == "q1"

    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        from asqi.loaders import load_test_cases

        with pytest.raises(FileNotFoundError):
            list(load_test_cases(str(tmp_path / "no.jsonl"), _RagCase))

    def test_validation_failure_raises_value_error_with_row_index(self, tmp_path):
        from asqi.loaders import load_test_cases

        jsonl = tmp_path / "data.jsonl"
        # Row 0 valid; row 1 missing 'answer'
        jsonl.write_text(
            json.dumps({"query": "q1", "answer": "a1"})
            + "\n"
            + json.dumps({"query": "q2"})  # missing answer
            + "\n"
        )

        gen = load_test_cases(str(jsonl), _RagCase)
        next(gen)  # row 0 OK
        with pytest.raises(ValueError) as excinfo:
            next(gen)
        # Documented: message includes row index, target class, and field name
        msg = str(excinfo.value)
        assert "Row 1" in msg
        assert "_RagCase" in msg
        assert "answer" in msg

    def test_input_mount_path_prepended_to_relative_paths(self, tmp_path):
        """When ``input_mount_path`` is given, relative paths resolve against it."""
        from asqi.loaders import load_test_cases

        mount = tmp_path / "input"
        mount.mkdir()
        (mount / "data.jsonl").write_text(
            json.dumps({"query": "q1", "answer": "a1"}) + "\n"
        )

        # Note: the file lives at <mount>/data.jsonl; we pass just "data.jsonl"
        out = list(load_test_cases("data.jsonl", _RagCase, input_mount_path=mount))
        assert len(out) == 1
        assert out[0].query == "q1"

    def test_json_format_supported(self, tmp_path):
        """``.json`` is one of the documented file formats."""
        from asqi.loaders import load_test_cases

        f = tmp_path / "data.json"
        f.write_text(
            json.dumps(
                [{"query": "q1", "answer": "a1"}, {"query": "q2", "answer": "a2"}]
            )
        )
        out = list(load_test_cases(str(f), _RagCase))
        assert len(out) == 2
        assert out[0].query == "q1"

    def test_csv_format_supported(self, tmp_path):
        """``.csv`` is one of the documented file formats."""
        from asqi.loaders import load_test_cases

        f = tmp_path / "data.csv"
        f.write_text("query,answer\nq1,a1\nq2,a2\n")
        out = list(load_test_cases(str(f), _RagCase))
        assert len(out) == 2
        assert out[0].answer == "a1"

    def test_parquet_format_supported(self, tmp_path):
        """``.parquet`` is one of the documented file formats."""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        from asqi.loaders import load_test_cases

        table = pa.table({"query": ["q1", "q2"], "answer": ["a1", "a2"]})
        f = tmp_path / "data.parquet"
        pq.write_table(table, str(f))

        out = list(load_test_cases(str(f), _RagCase))
        assert len(out) == 2
        assert out[1].query == "q2"

    def test_dict_config_path_with_local_csv_and_mapping(self, tmp_path):
        """When ``path`` is a dict / ``HFDatasetDefinition``, loads via the
        HF dataset loader, with ``mapping`` rename applied."""
        from asqi.loaders import load_test_cases

        f = tmp_path / "rows.csv"
        f.write_text("q,a\nq1,a1\nq2,a2\n")

        config = {
            "type": "huggingface",
            "loader_params": {
                "builder_name": "csv",
                "data_files": str(f),
            },
            "mapping": {"q": "query", "a": "answer"},
        }
        out = list(load_test_cases(config, _RagCase))
        assert len(out) == 2
        assert out[0].query == "q1"
        assert out[0].answer == "a1"

    def test_dict_config_with_hub_path_dispatches_to_hub_loader(self):
        """When ``path`` is a dict / ``HFDatasetDefinition`` whose
        ``loader_params.hub_path`` is set, ``load_test_cases`` loads via the
        Hub branch (and runs each row through the Pydantic schema)."""
        from unittest.mock import patch as _patch

        from asqi.loaders import load_test_cases
        from datasets import Dataset

        fake_ds = Dataset.from_dict({"query": ["q1", "q2"], "answer": ["a1", "a2"]})
        with (
            _patch("asqi.datasets._load_from_hub", return_value=fake_ds) as mock_hub,
            _patch("asqi.datasets._load_from_local") as mock_local,
        ):
            out = list(
                load_test_cases(
                    {
                        "type": "huggingface",
                        "loader_params": {"hub_path": "org/dataset"},
                    },
                    _RagCase,
                )
            )

        mock_hub.assert_called_once()
        mock_local.assert_not_called()
        assert len(out) == 2
        assert all(isinstance(o, _RagCase) for o in out)
        assert out[0].query == "q1"

    def test_dict_config_with_hub_path_applies_mapping_rename(self):
        """When ``path`` is a dict / ``HFDatasetDefinition`` and ``mapping``
        is set, columns are renamed *before* iteration — for both the
        local-files branch and the Hub branch.

        The local-CSV variant of this is covered by
        ``test_dict_config_path_with_local_csv_and_mapping``. Pin the same
        rename for the Hub branch so a future divergence between the two
        loaders (where Hub forgets to rename) cannot slip through."""
        from unittest.mock import patch as _patch

        from asqi.loaders import load_test_cases
        from datasets import Dataset

        # Hub returns a dataset with the original (un-renamed) column names —
        # `q`/`a` rather than the schema-required `query`/`answer`. Without
        # the mapping rename, validation against `_RagCase` would fail.
        fake_ds = Dataset.from_dict({"q": ["q1", "q2"], "a": ["a1", "a2"]})
        with (
            _patch("asqi.datasets._load_from_hub", return_value=fake_ds) as mock_hub,
            _patch("asqi.datasets._load_from_local") as mock_local,
        ):
            out = list(
                load_test_cases(
                    {
                        "type": "huggingface",
                        "loader_params": {"hub_path": "org/dataset"},
                        "mapping": {"q": "query", "a": "answer"},
                    },
                    _RagCase,
                )
            )

        # Hub branch was used; local branch was not.
        mock_hub.assert_called_once()
        mock_local.assert_not_called()
        # Rename succeeded — rows validate against the schema's required fields.
        assert len(out) == 2
        assert out[0].query == "q1"
        assert out[0].answer == "a1"
        assert out[1].query == "q2"
        assert out[1].answer == "a2"

    def test_pathlib_path_input_supported(self, tmp_path):
        """Documented signature:
        ``path: str | Path | dict | HFDatasetDefinition``. Pin that
        ``pathlib.Path`` (not just ``str``) is accepted on the file-path
        arm, since callers that already use ``Path`` would otherwise be
        broken by a future signature narrow to ``str`` only."""
        from pathlib import Path

        from asqi.loaders import load_test_cases

        f = tmp_path / "data.jsonl"
        f.write_text(json.dumps({"query": "q1", "answer": "a1"}) + "\n")

        out = list(load_test_cases(Path(f), _RagCase))
        assert len(out) == 1
        assert isinstance(out[0], _RagCase)
        assert out[0].query == "q1"

    def test_unsupported_extension_falls_through_to_loader(self, tmp_path):
        """Per implementation: extensions outside the documented set are
        passed straight to ``load_dataset`` as the builder name. We don't
        promise this works for arbitrary extensions — but a missing file
        still raises ``FileNotFoundError`` (the documented behaviour)."""
        from asqi.loaders import load_test_cases

        with pytest.raises(FileNotFoundError):
            list(load_test_cases(str(tmp_path / "nope.csv"), _RagCase))
