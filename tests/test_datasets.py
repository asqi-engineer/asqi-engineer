import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, Features, Value
from datasets import Sequence as HFSequence
from datasets.features import Image as HFImage

from asqi.datasets import load_hf_dataset, validate_dataset_features
from asqi.schemas import (
    DatasetFeature,
    DatasetLoaderParams,
    HFDatasetDefinition,
    ImageFeature,
    ListFeature,
    ValueFeature,
)


class TestValidateDatasetFeatures:
    """Test the validate_dataset_features function."""

    def test_validate_with_all_required_features_present(self):
        """Test validation passes when all required features are present."""
        dataset = Dataset.from_dict(
            {"prompt": ["test"], "response": ["answer"], "score": [0.9]}
        )
        features = [
            ValueFeature(name="prompt", dtype="string", required=True),
            ValueFeature(name="response", dtype="string", required=True),
            ValueFeature(
                name="score", dtype="float64", required=True
            ),  # float64 is default dtype for float
        ]

        # Should not raise
        validate_dataset_features(dataset, features, dataset_name="test_dataset")

    def test_validate_with_optional_features_missing(self):
        """Test validation passes when optional features are missing."""
        dataset = Dataset.from_dict({"prompt": ["test"], "response": ["answer"]})
        features = [
            ValueFeature(name="prompt", dtype="string", required=True),
            ValueFeature(name="response", dtype="string", required=True),
            ValueFeature(
                name="metadata", dtype="string", required=False
            ),  # Optional, not in dataset
        ]

        # Should not raise even though metadata is missing
        validate_dataset_features(dataset, features, dataset_name="test_dataset")

    def test_validate_with_required_feature_missing(self):
        """Test validation fails when required feature is missing."""
        dataset = Dataset.from_dict({"prompt": ["test"]})
        features = [
            ValueFeature(name="prompt", dtype="string", required=True),
            ValueFeature(name="response", dtype="string", required=True),  # Missing!
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_dataset_features(dataset, features, dataset_name="test_dataset")

        error_msg = str(exc_info.value)
        assert "test_dataset" in error_msg
        assert "response" in error_msg
        assert "Available columns: prompt" in error_msg
        assert "Check your dataset mapping configuration" in error_msg

    def test_validate_with_multiple_missing_features(self):
        """Test validation reports all missing required features."""
        dataset = Dataset.from_dict({"id": [1]})
        features = [
            ValueFeature(name="id", dtype="int32", required=True),
            ValueFeature(name="prompt", dtype="string", required=True),
            ValueFeature(name="response", dtype="string", required=True),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_dataset_features(dataset, features, dataset_name="test_dataset")

        error_msg = str(exc_info.value)
        assert "prompt" in error_msg
        assert "response" in error_msg

    def test_validate_with_complex_feature_types(self):
        """Test validation works with complex feature types."""
        dataset = Dataset.from_dict(
            {
                "text": ["sample"],
                "tags": [["a", "b"]],
                "image": [[1, 2, 3]],
            },
            features=Features(
                {
                    "text": Value("string"),
                    "tags": HFSequence(Value("string")),
                    "image": HFImage(),
                }
            ),
        )
        features = [
            ValueFeature(name="text", dtype="string", required=True),
            ListFeature(name="tags", feature="string", required=True),
            ImageFeature(name="image", required=True),
        ]

        # Should not raise
        validate_dataset_features(dataset, features, dataset_name="test_dataset")

    def test_validate_with_extra_columns_in_dataset(self):
        """Test validation passes when dataset has extra columns not in schema."""
        dataset = Dataset.from_dict(
            {
                "prompt": ["test"],
                "response": ["answer"],
                "extra_field": ["bonus"],
            }
        )
        features = [
            ValueFeature(name="prompt", dtype="string", required=True),
            ValueFeature(name="response", dtype="string", required=True),
        ]

        # Should not raise - extra columns are fine
        validate_dataset_features(dataset, features, dataset_name="test_dataset")

    def test_validate_with_default_required_false(self):
        """Test that features default to required=False for backward compatibility."""
        dataset = Dataset.from_dict({"prompt": ["test"]})
        # Create feature without explicitly setting required (should default to False)
        features = [
            ValueFeature(name="prompt", dtype="string"),
            ValueFeature(
                name="optional_field", dtype="string"
            ),  # Missing, but required defaults to False
        ]

        # Should not raise because required defaults to False
        validate_dataset_features(dataset, features, dataset_name="test_dataset")


class TestLoadHFDatasetWithValidation:
    """Test load_hf_dataset function with validation."""

    def test_load_without_validation(self):
        """Test loading dataset without validation (backward compatible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"text": ["sample1", "sample2"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )
            loaded = load_hf_dataset(dataset_config)

            assert len(loaded) == 2
            assert "text" in loaded.column_names

    def test_load_with_validation_passes(self):
        """Test loading dataset with validation that passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"prompt": ["q1"], "response": ["a1"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ValueFeature(name="prompt", dtype="string", required=True),
                ValueFeature(name="response", dtype="string", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )
            loaded = load_hf_dataset(dataset_config)
            validate_dataset_features(
                loaded, expected_features, dataset_name="test_data"
            )

            assert len(loaded) == 1
            assert "prompt" in loaded.column_names
            assert "response" in loaded.column_names

    def test_load_with_validation_fails(self):
        """Test loading dataset with validation that fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"wrong_column": ["data"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ValueFeature(name="prompt", dtype="string", required=True),
                ValueFeature(name="response", dtype="string", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )
            loaded = load_hf_dataset(dataset_config)

            with pytest.raises(ValueError) as exc_info:
                validate_dataset_features(
                    loaded, expected_features, dataset_name="test_data"
                )

            error_msg = str(exc_info.value)
            assert "test_data" in error_msg
            assert "prompt" in error_msg
            assert "response" in error_msg

    def test_load_with_validation_and_mapping(self):
        """Test loading dataset with validation after column mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"question": ["q1"], "answer": ["a1"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ValueFeature(name="prompt", dtype="string", required=True),
                ValueFeature(name="response", dtype="string", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
                mapping={"question": "prompt", "answer": "response"},
            )
            loaded = load_hf_dataset(dataset_config)

            validate_dataset_features(
                loaded, expected_features, dataset_name="test_data"
            )

            # Verify mapping worked correctly
            assert "prompt" in loaded.column_names
            assert "response" in loaded.column_names
            assert "question" not in loaded.column_names
            assert "answer" not in loaded.column_names

    def test_load_with_optional_features(self):
        """Test loading dataset with optional features that are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"prompt": ["q1"], "response": ["a1"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            # Define schema with required and optional features
            expected_features = [
                ValueFeature(name="prompt", dtype="string", required=True),
                ValueFeature(name="response", dtype="string", required=True),
                ValueFeature(
                    name="metadata", dtype="string", required=False
                ),  # Optional
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )
            loaded = load_hf_dataset(dataset_config)

            # Validate - should pass even though metadata is missing
            validate_dataset_features(
                loaded, expected_features, dataset_name="test_data"
            )

            assert len(loaded) == 1
            assert "prompt" in loaded.column_names
            assert "response" in loaded.column_names
            assert "metadata" not in loaded.column_names

    def test_load_with_integrated_validation(self):
        """Test loading dataset with integrated validation in single call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"text": ["sample"], "label": ["cat"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ValueFeature(name="text", dtype="string", required=True),
                ValueFeature(name="label", dtype="string", required=True),
            ]

            # Load and validate in one call
            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )
            loaded = load_hf_dataset(
                dataset_config,
                expected_features=expected_features,
                dataset_name="source_data",
            )

            assert len(loaded) == 1
            assert "text" in loaded.column_names
            assert "label" in loaded.column_names

    def test_load_with_integrated_validation_fails(self):
        """Test integrated validation fails with missing required features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"text": ["sample"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            # Define expected features including missing column
            expected_features = [
                ValueFeature(name="text", dtype="string", required=True),
                ValueFeature(name="label", dtype="string", required=True),  # Missing!
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            # Should fail during load
            with pytest.raises(ValueError) as exc_info:
                load_hf_dataset(
                    dataset_config,
                    expected_features=expected_features,
                    dataset_name="source_data",
                )

            error_msg = str(exc_info.value)
            assert "source_data" in error_msg
            assert "label" in error_msg

    def test_load_with_dataset_feature_validation(self):
        """Test validation with DatasetFeature (simple scalar features)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"text": ["sample"], "label": ["cat"]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                DatasetFeature(name="text", dtype="string"),
                DatasetFeature(name="label", dtype="string"),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            # Load and validate - DatasetFeature doesn't have required field,
            # so validation should pass even though required defaults to False
            loaded = load_hf_dataset(
                dataset_config,
                expected_features=expected_features,
                dataset_name="test_data",
            )

            assert len(loaded) == 1
            assert "text" in loaded.column_names
            assert "label" in loaded.column_names


class TestTypeValidation:
    """Test type validation functionality."""

    def test_validate_scalar_types_pass(self):
        """Test type validation passes with correct scalar dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict(
                {"text": ["sample"], "count": [42], "score": [0.95], "is_valid": [True]}
            )
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ValueFeature(name="text", dtype="string", required=True),
                ValueFeature(name="count", dtype="int64", required=True),
                ValueFeature(name="score", dtype="float64", required=True),
                ValueFeature(name="is_valid", dtype="bool", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            # Should pass with type validation
            loaded = load_hf_dataset(
                dataset_config,
                expected_features=expected_features,
            )

            assert len(loaded) == 1

    def test_validate_scalar_types_fail(self):
        """Test type validation fails with wrong scalar dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"count": [42]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            # Expect string instead of int64
            expected_features = [
                ValueFeature(name="count", dtype="string", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            # Should fail with type mismatch
            with pytest.raises(ValueError) as exc_info:
                load_hf_dataset(
                    dataset_config,
                    expected_features=expected_features,
                )

            error_msg = str(exc_info.value)
            assert "Type mismatches" in error_msg
            assert "count" in error_msg

    def test_load_without_validation(self):
        """Test loading without validation skips type checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"count": [42]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            # Should pass without validation (expected_features=None)
            loaded = load_hf_dataset(dataset_config)

            assert len(loaded) == 1
            assert "count" in loaded.column_names

    def test_validate_image_feature_type(self):
        """Test type validation for Image features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict(
                {"image": [[1, 2, 3]]}, features=Features({"image": HFImage()})
            )
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ImageFeature(name="image", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            loaded = load_hf_dataset(
                dataset_config,
                expected_features=expected_features,
            )

            assert len(loaded) == 1

    def test_validate_list_feature_type(self):
        """Test type validation for List features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Dataset.from_dict({"tags": [["tag1", "tag2"]]})
            data_file = Path(tmpdir) / "data.parquet"
            dataset.to_parquet(data_file)

            expected_features = [
                ListFeature(name="tags", feature="string", required=True),
            ]

            dataset_config = HFDatasetDefinition(
                type="huggingface",
                loader_params=DatasetLoaderParams(
                    builder_name="parquet",
                    data_files=str(data_file),
                ),
            )

            loaded = load_hf_dataset(
                dataset_config,
                expected_features=expected_features,
            )

            assert len(loaded) == 1
