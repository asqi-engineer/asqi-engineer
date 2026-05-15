# pyright: reportCallIssue=false, reportArgumentType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingTypeArgument=false
"""Contract tests for the public schema models.

These models form the wire contract for ``manifest.yaml`` files embedded in
test/SDG container images and for the public schema imports the spec lists.

Renaming or tightening any of these is a breaking change for every test
container.

Pyright suppression note: many tests in this file intentionally call models
with only the *documented* required fields, to pin the spec's "minimum
required" surface. Pyright's strict call-arg analysis misreads Pydantic
field defaults as required arguments, so we disable that one diagnostic at
file scope.
"""

from __future__ import annotations

import pytest
from asqi.schemas import (
    DatasetFeature,
    DatasetType,
    InputDataset,
    InputParameter,
    Manifest,
    SystemInput,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifestContract:
    """Documented field set, defaults, and required-vs-optional surface."""

    def test_minimum_required_fields(self):
        """`name` and `version` are the only documented required fields."""
        m = Manifest(name="my-test", version="1.0.0")
        assert m.name == "my-test"
        assert m.version == "1.0.0"

    def test_documented_field_defaults(self):
        m = Manifest(name="x", version="1")
        assert m.description is None
        assert m.host_access is False
        assert m.input_systems == []
        assert m.input_schema == []
        assert m.input_datasets == []
        assert m.output_metrics == []
        assert m.output_artifacts is None
        assert m.environment_variables == []
        assert m.output_reports == []
        assert m.output_datasets == []

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            Manifest(version="1")  # missing name
        with pytest.raises(ValidationError):
            Manifest(name="x")  # missing version

    def test_output_metrics_accepts_list_of_strings(self):
        """``output_metrics`` is documented as ``list[str] | list[OutputMetric]``.
        Pin both arms of the union — the simple ``list[str]`` form first."""
        m = Manifest(name="x", version="1", output_metrics=["accuracy", "f1"])
        assert m.output_metrics == ["accuracy", "f1"]

    def test_output_metrics_accepts_list_of_output_metric_objects(self):
        """Structured ``OutputMetric`` form. Each entry carries ``name``,
        ``type``, and an optional ``description``. Pinning the structured arm
        catches a regression that narrows the type to ``list[str]`` only."""
        from asqi.schemas import OutputMetric

        metrics = [
            OutputMetric(name="accuracy", type="float", description="0..1"),
            OutputMetric(name="passes", type="integer"),
        ]
        m = Manifest(name="x", version="1", output_metrics=metrics)
        assert len(m.output_metrics) == 2
        assert isinstance(m.output_metrics[0], OutputMetric)
        assert m.output_metrics[0].name == "accuracy"
        assert m.output_metrics[0].type == "float"
        assert m.output_metrics[1].description is None

    def test_output_metrics_accepts_list_of_dicts_coerced_to_output_metric(self):
        """Dicts matching the ``OutputMetric`` shape coerce into instances —
        the documented YAML form callers actually write in manifest.yaml."""
        from asqi.schemas import OutputMetric

        m = Manifest(
            name="x",
            version="1",
            output_metrics=[{"name": "accuracy", "type": "float"}],
        )
        # Pydantic coerces the dict into an OutputMetric instance.
        assert isinstance(m.output_metrics[0], OutputMetric)
        assert m.output_metrics[0].name == "accuracy"

    def test_output_artifacts_populated_arm(self):
        """``output_artifacts: list[OutputArtifact] | None``. Default ``None``
        is covered above; pin the populated arm so a regression narrowing
        the type (e.g. to ``list[str]``) is caught at parse time."""
        from asqi.schemas import OutputArtifact

        m = Manifest(
            name="x",
            version="1",
            output_artifacts=[
                OutputArtifact(name="report", path="/output/report.html"),
            ],
        )
        assert m.output_artifacts is not None
        assert len(m.output_artifacts) == 1
        assert isinstance(m.output_artifacts[0], OutputArtifact)
        assert m.output_artifacts[0].name == "report"
        assert m.output_artifacts[0].path == "/output/report.html"

    def test_environment_variables_populated_arm(self):
        """``environment_variables: list[EnvironmentVariable]`` with default
        ``[]``. Pin that the populated arm coerces dict YAML entries into
        ``EnvironmentVariable`` instances and preserves the documented
        ``required`` default."""
        from asqi.schemas import EnvironmentVariable

        m = Manifest(
            name="x",
            version="1",
            environment_variables=[
                {"name": "OPENAI_API_KEY"},
                {"name": "OPTIONAL_VAR", "required": False},
            ],
        )
        assert len(m.environment_variables) == 2
        assert all(isinstance(e, EnvironmentVariable) for e in m.environment_variables)
        assert m.environment_variables[0].name == "OPENAI_API_KEY"
        assert m.environment_variables[0].required is True  # documented default
        assert m.environment_variables[1].required is False

    def test_output_reports_populated_arm(self):
        """``output_reports: list[OutputReports]`` with default ``[]``. Pin
        the populated arm and the ``Literal['pdf', 'html']`` constraint on
        the report ``type`` field."""
        from asqi.schemas import OutputReports

        m = Manifest(
            name="x",
            version="1",
            output_reports=[
                {"name": "summary", "type": "pdf"},
                {"name": "details", "type": "html"},
            ],
        )
        assert len(m.output_reports) == 2
        assert all(isinstance(r, OutputReports) for r in m.output_reports)
        assert m.output_reports[0].type == "pdf"
        assert m.output_reports[1].type == "html"

    def test_output_datasets_populated_arm(self):
        """``output_datasets: list[OutputDataset]`` with default ``[]``. Pin
        the populated arm so a regression that drops the field from the
        schema (or narrows its element type) surfaces at parse time."""
        from asqi.schemas import OutputDataset

        m = Manifest(
            name="x",
            version="1",
            output_datasets=[{"name": "synthesized_qa", "type": "huggingface"}],
        )
        assert len(m.output_datasets) == 1
        assert isinstance(m.output_datasets[0], OutputDataset)
        assert m.output_datasets[0].name == "synthesized_qa"


# ---------------------------------------------------------------------------
# SystemInput
# ---------------------------------------------------------------------------


class TestSystemInputContract:
    def test_required_fields_only(self):
        s = SystemInput(name="system_under_test", type="llm_api")
        assert s.required is True  # documented default
        assert s.description is None

    def test_type_accepts_single_string(self):
        s = SystemInput(name="x", type="rag_api")
        assert s.type == "rag_api"

    def test_type_accepts_list_of_strings(self):
        """`type` accepts ``str | list[str]`` per the documented contract."""
        s = SystemInput(name="x", type=["llm_api", "vlm_api"])
        assert s.type == ["llm_api", "vlm_api"]

    @pytest.mark.parametrize(
        "documented_type",
        [
            "llm_api",
            "rest_api",
            "rag_api",
            "image_generation_api",
            "image_editing_api",
            "vlm_api",
            "agent_cli",
        ],
    )
    def test_documented_types_round_trip(self, documented_type):
        """The seven documented type tags must remain accepted strings."""
        s = SystemInput(name="x", type=documented_type)
        assert s.type == documented_type

    def test_type_field_is_loose_str_not_enum(self):
        """``type`` is annotated ``str | list[str]`` (not a Literal / Enum).
        The seven documented values are *descriptive* — the schema does not
        reject unknown strings. Pin the loose contract so a future tightening
        to ``Literal[...]`` (which would silently break any container using
        a custom type tag) forces a deliberate decision."""
        # An arbitrary tag the spec does not list is still accepted.
        s = SystemInput(name="x", type="custom_future_type")
        assert s.type == "custom_future_type"
        # And in list form.
        s2 = SystemInput(name="x", type=["llm_api", "custom_future_type"])
        assert s2.type == ["llm_api", "custom_future_type"]


# ---------------------------------------------------------------------------
# InputParameter
# ---------------------------------------------------------------------------


class TestInputParameterContract:
    @pytest.mark.parametrize(
        "doc_type",
        ["string", "integer", "float", "boolean", "list", "object", "enum"],
    )
    def test_documented_type_literal_values(self, doc_type):
        """The seven documented type values must remain accepted."""
        kwargs: dict = {"name": "p", "type": doc_type}
        # `enum` requires `choices`; `list` allows any items.
        if doc_type == "enum":
            kwargs["choices"] = ["a", "b"]
        InputParameter(**kwargs)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            InputParameter(name="p", type="not-a-real-type")

    def test_items_only_allowed_for_list(self):
        """`items` may only be set when type='list' (model validator)."""
        with pytest.raises(ValidationError):
            InputParameter(name="p", type="string", items="string")

    def test_properties_only_allowed_for_object(self):
        with pytest.raises(ValidationError):
            InputParameter(
                name="p",
                type="string",
                properties=[InputParameter(name="x", type="string")],
            )

    def test_choices_only_allowed_for_enum(self):
        with pytest.raises(ValidationError):
            InputParameter(name="p", type="string", choices=["a"])

    def test_enum_requires_choices(self):
        with pytest.raises(ValidationError):
            InputParameter(name="p", type="enum")

    def test_enum_default_must_be_in_choices(self):
        with pytest.raises(ValidationError):
            InputParameter(name="p", type="enum", choices=["a", "b"], default="z")

    def test_recursive_items(self):
        """The contract allows ``items`` to itself be an ``InputParameter``."""
        nested = InputParameter(name="inner", type="string")
        outer = InputParameter(name="outer", type="list", items=nested)
        assert isinstance(outer.items, InputParameter)
        assert outer.items.name == "inner"

    def test_ui_config_is_arbitrary_dict(self):
        p = InputParameter(
            name="p", type="string", ui_config={"widget": "textarea", "rows": 4}
        )
        assert p.ui_config == {"widget": "textarea", "rows": 4}


# ---------------------------------------------------------------------------
# InputDataset + DatasetType
# ---------------------------------------------------------------------------


class TestDatasetTypeContract:
    def test_documented_members(self):
        assert DatasetType.HUGGINGFACE.value == "huggingface"
        assert DatasetType.PDF.value == "pdf"
        assert DatasetType.TXT.value == "txt"

    def test_str_enum_compares_to_string(self):
        assert DatasetType.HUGGINGFACE == "huggingface"


class TestInputDatasetContract:
    def test_pdf_only_does_not_require_features(self):
        d = InputDataset(name="docs", type=DatasetType.PDF)
        assert d.features is None
        assert d.required is True  # documented default

    def test_huggingface_requires_features(self):
        """``_validate_features_for_huggingface`` raises ``ValueError`` if
        ``huggingface`` is in ``type`` but ``features`` is empty."""
        with pytest.raises(ValidationError):
            InputDataset(name="d", type=DatasetType.HUGGINGFACE)

    def test_huggingface_with_features_ok(self):
        d = InputDataset(
            name="d",
            type=DatasetType.HUGGINGFACE,
            features=[DatasetFeature(name="q", dtype="string")],
        )
        assert d.features is not None and len(d.features) == 1

    def test_huggingface_in_type_list_still_requires_features(self):
        """The 'required when ``type`` includes ``huggingface``' rule applies
        to the list form too."""
        with pytest.raises(ValidationError):
            InputDataset(name="d", type=[DatasetType.HUGGINGFACE, DatasetType.PDF])

    def test_type_accepts_list_of_dataset_types(self):
        d = InputDataset(name="d", type=[DatasetType.PDF, DatasetType.TXT])
        assert d.type == [DatasetType.PDF, DatasetType.TXT]


# ---------------------------------------------------------------------------
# DatasetFeature
# ---------------------------------------------------------------------------


class TestDatasetFeatureContract:
    def test_required_fields(self):
        f = DatasetFeature(name="col", dtype="string")
        assert f.required is False  # documented default
        assert f.description is None

    def test_missing_dtype_rejected(self):
        with pytest.raises(ValidationError):
            DatasetFeature(name="col")

    @pytest.mark.parametrize("doc_dtype", ["string", "int64", "float32", "bool"])
    def test_documented_dtype_examples(self, doc_dtype):
        DatasetFeature(name="col", dtype=doc_dtype)
