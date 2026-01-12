from typing import get_args

import pytest
from datasets.features import Value
from pydantic import ValidationError

from asqi.schemas import DatasetFeature, HFDtype, InputDataset


class TestHFDtypeLiteral:
    """Test that our HFDtype Literal matches actual HuggingFace Value types."""

    def test_all_literal_values_are_valid_hf_types(self):
        """Test that every value in our HFDtype Literal can create a valid HuggingFace Value."""
        literal_dtypes = get_args(HFDtype)

        failed_dtypes = []
        for dtype in literal_dtypes:
            try:
                # Try to create a Value with this dtype
                Value(dtype=dtype)
            except Exception as e:
                failed_dtypes.append((dtype, str(e)))

        if failed_dtypes:
            error_msg = "The following dtypes in HFDtype Literal are not valid HuggingFace Value types:\n"
            for dtype, error in failed_dtypes:
                error_msg += f"  - '{dtype}': {error}\n"
            pytest.fail(error_msg)

    def test_common_hf_types_are_in_literal(self):
        """Test that common HuggingFace types are included in our Literal."""
        # Test the most common types that users will need
        common_types = [
            "string",
            "int64",
            "int32",
            "float64",
            "float32",
            "bool",
            "null",
            "binary",
            "timestamp[s]",
            "date32",
        ]

        literal_dtypes = get_args(HFDtype)

        missing_types = []
        for dtype in common_types:
            if dtype not in literal_dtypes:
                missing_types.append(dtype)

        if missing_types:
            pytest.fail(
                f"Common HuggingFace types missing from HFDtype Literal: {missing_types}"
            )


class TestDatasetFeature:
    """Test the DatasetFeature model with dtype validation."""

    def test_create_with_string_dtype(self):
        """Test creating DatasetFeature with valid string dtype."""
        feature = DatasetFeature(
            name="user_id", dtype="int64", description="Unique user identifier"
        )
        assert feature.name == "user_id"
        assert feature.dtype == "int64"
        assert feature.description == "Unique user identifier"

    def test_create_multiple_features(self):
        """Test creating multiple features with different dtypes."""
        features = [
            DatasetFeature(name="id", dtype="int64"),
            DatasetFeature(name="name", dtype="string"),
            DatasetFeature(name="score", dtype="float32"),
            DatasetFeature(name="active", dtype="bool"),
            DatasetFeature(name="created_at", dtype="timestamp[s]"),
        ]

        assert len(features) == 5
        assert features[0].dtype == "int64"
        assert features[1].dtype == "string"
        assert features[4].dtype == "timestamp[s]"

    def test_invalid_dtype_raises_validation_error(self):
        """Test that invalid dtypes raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetFeature(name="bad", dtype="invalid_type")

        error = exc_info.value
        assert "dtype" in str(error)

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            DatasetFeature(dtype="string")
        assert "name" in str(exc_info.value)

        # Missing dtype
        with pytest.raises(ValidationError) as exc_info:
            DatasetFeature(name="test")
        assert "dtype" in str(exc_info.value)

    def test_optional_description(self):
        """Test that description is optional."""
        feature = DatasetFeature(name="test", dtype="string")
        assert feature.description is None

        feature_with_desc = DatasetFeature(
            name="test", dtype="string", description="Test feature"
        )
        assert feature_with_desc.description == "Test feature"

    def test_all_common_dtypes(self):
        """Test all common HuggingFace dtypes are accepted."""
        common_dtypes = [
            "null",
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "string",
            "binary",
        ]

        for dtype in common_dtypes:
            feature = DatasetFeature(name=f"test_{dtype}", dtype=dtype)
            assert feature.dtype == dtype

    def test_dtype_aliases(self):
        """Test that dtype aliases are accepted."""
        # 'float' alias for 'float32'
        feature = DatasetFeature(name="value", dtype="float")
        assert feature.dtype == "float"

        # 'double' alias for 'float64'
        feature = DatasetFeature(name="value", dtype="double")
        assert feature.dtype == "double"

    def test_parametric_dtypes(self):
        """Test parametric dtypes (timestamp, time, duration)."""
        parametric_dtypes = [
            "timestamp[s]",
            "timestamp[ms]",
            "timestamp[us]",
            "timestamp[ns]",
            "time32[s]",
            "time32[ms]",
            "time64[us]",
            "time64[ns]",
            "duration[s]",
            "duration[ms]",
            "duration[us]",
            "duration[ns]",
        ]

        for dtype in parametric_dtypes:
            feature = DatasetFeature(name=f"test_{dtype}", dtype=dtype)
            assert feature.dtype == dtype


class TestInputDatasetWithFeatures:
    """Test InputDataset with DatasetFeature validation."""

    def test_input_dataset_with_valid_features(self):
        """Test creating InputDataset with validated features."""
        dataset = InputDataset(
            name="user_activity",
            required=True,
            type="huggingface",
            description="User activity dataset",
            features=[
                DatasetFeature(name="user_id", dtype="int64"),
                DatasetFeature(name="activity", dtype="string"),
                DatasetFeature(name="timestamp", dtype="timestamp[s]"),
            ],
        )

        assert dataset.name == "user_activity"
        assert dataset.required is True
        assert dataset.type == "huggingface"
        assert len(dataset.features) == 3
        assert dataset.features[0].dtype == "int64"
        assert dataset.features[1].dtype == "string"
        assert dataset.features[2].dtype == "timestamp[s]"

    def test_input_dataset_features_must_be_defined_for_huggingface(self):
        """Test that HuggingFace datasets require features to be defined."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="test_dataset",
                required=True,
                type="huggingface",
                features=None,  # Missing features
            )
        assert "Features must be defined for HuggingFace dataset" in str(exc_info.value)

    def test_input_dataset_with_invalid_feature_dtype(self):
        """Test that invalid feature dtypes are caught."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="test_dataset",
                required=True,
                type="huggingface",
                features=[
                    DatasetFeature(name="valid", dtype="string"),
                    DatasetFeature(name="invalid", dtype="not_a_type"),
                ],
            )
        error_str = str(exc_info.value)
        assert "dtype" in error_str

    def test_input_dataset_pdf_type_no_features(self):
        """Test that PDF datasets don't require features."""
        dataset = InputDataset(
            name="document",
            required=True,
            type="pdf",
            description="PDF document",
            features=None,
        )
        assert dataset.name == "document"
        assert dataset.type == "pdf"
        assert dataset.features is None

    def test_input_dataset_multiple_types(self):
        """Test that InputDataset can accept multiple types."""
        dataset = InputDataset(
            name="knowledge_base",
            required=True,
            type=["pdf", "txt"],
            description="Knowledge base - accepts PDF or TXT",
            features=None,
        )
        assert dataset.name == "knowledge_base"
        assert dataset.type == ["pdf", "txt"]
        assert dataset.features is None

    def test_input_dataset_multiple_types_with_huggingface(self):
        """Test that HuggingFace in multi-type list doesn't require features."""
        dataset = InputDataset(
            name="flexible_data",
            required=True,
            type=["huggingface", "pdf", "txt"],
            description="Flexible input - accepts multiple formats",
            features=None,  # Features not required when multiple types accepted
        )
        assert dataset.name == "flexible_data"
        assert dataset.type == ["huggingface", "pdf", "txt"]
        assert dataset.features is None

    def test_input_dataset_multiple_types_with_features(self):
        """Test that multi-type datasets can still have features defined."""
        dataset = InputDataset(
            name="flexible_data",
            required=True,
            type=["huggingface", "pdf", "txt"],
            description="Flexible input with optional features",
            features=[
                DatasetFeature(name="text", dtype="string"),
            ],
        )
        assert dataset.name == "flexible_data"
        assert dataset.type == ["huggingface", "pdf", "txt"]
        assert len(dataset.features) == 1

    def test_input_dataset_single_huggingface_still_requires_features(self):
        """Test that single HuggingFace type still requires features (backward compatibility)."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="hf_only",
                required=True,
                type="huggingface",  # Single type
                features=None,  # Should fail
            )
        assert "Features must be defined when only 'huggingface' type is accepted" in str(exc_info.value)


class TestDatasetFeatureDtypeExamples:
    """Test real-world usage examples of DatasetFeature dtypes."""

    def test_nlp_dataset_features(self):
        """Test typical NLP dataset feature types."""
        features = [
            DatasetFeature(name="text", dtype="string"),
            DatasetFeature(name="label", dtype="int64"),
            DatasetFeature(name="tokens", dtype="string"),
        ]
        assert all(f.dtype in ["string", "int64"] for f in features)

    def test_numeric_dataset_features(self):
        """Test typical numeric dataset feature types."""
        features = [
            DatasetFeature(name="id", dtype="int64"),
            DatasetFeature(name="value", dtype="float64"),
            DatasetFeature(name="count", dtype="int32"),
            DatasetFeature(name="percentage", dtype="float32"),
        ]
        assert len(features) == 4
        assert features[1].dtype == "float64"

    def test_temporal_dataset_features(self):
        """Test temporal data types."""
        features = [
            DatasetFeature(name="event_timestamp", dtype="timestamp[s]"),
            DatasetFeature(name="created_date", dtype="date32"),
            DatasetFeature(name="duration", dtype="duration[ms]"),
            DatasetFeature(name="event_time", dtype="time64[us]"),
        ]
        assert len(features) == 4
        assert "timestamp" in features[0].dtype
