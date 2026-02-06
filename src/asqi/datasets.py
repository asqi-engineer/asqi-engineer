import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from datasets import Dataset, IterableDataset, Value, load_dataset
from datasets import Sequence as HFSequence
from datasets.features import Audio, ClassLabel, Image, Video

from asqi.schemas import (
    AudioFeature,
    ClassLabelFeature,
    DatasetFeature,
    DictFeature,
    HFDatasetDefinition,
    HFFeature,
    ImageFeature,
    ListFeature,
    ValueFeature,
    VideoFeature,
)


def _validate_scalar_dtype(
    expected: Union[ValueFeature, DatasetFeature], actual: Value
) -> Optional[str]:
    """Validate scalar dtype matches expected type.

    Args:
        expected: Expected feature with dtype
        actual: Actual HuggingFace Value feature

    Returns:
        Error message if validation fails, None if passes
    """
    # Normalize dtype for comparison
    expected_dtype = str(expected.dtype)
    actual_dtype = str(actual.dtype)

    # Handle dtype aliases (float->float32, double->float64)
    dtype_aliases = {"float": "float32", "double": "float64"}
    expected_dtype = dtype_aliases.get(expected_dtype, expected_dtype)
    actual_dtype = dtype_aliases.get(actual_dtype, actual_dtype)

    if expected_dtype != actual_dtype:
        return f"expected dtype '{expected_dtype}', got '{actual_dtype}'"
    return None


# Mapping of feature types to their expected HuggingFace types
_FEATURE_TYPE_MAP = {
    ValueFeature: (Value, "scalar type"),
    DatasetFeature: (Value, "scalar type"),
    ImageFeature: (Image, "Image"),
    AudioFeature: (Audio, "Audio"),
    VideoFeature: (Video, "Video"),
    ClassLabelFeature: (ClassLabel, "ClassLabel"),
    ListFeature: (HFSequence, "List/Sequence"),
    DictFeature: (dict, "Dict"),
}


def _validate_feature_type(
    expected: Union[DatasetFeature, HFFeature], actual
) -> Optional[str]:
    """Validate that actual HuggingFace feature matches expected schema.

    Args:
        expected: Expected feature from manifest
        actual: Actual feature from loaded dataset

    Returns:
        Error message if validation fails, None if validation passes

    Note:
        Complex nested structures are not fully validated.
    """
    if isinstance(expected, DictFeature):
        # Basic validation: ensure actual is a dict
        if not isinstance(actual, dict):
            return f"expected Dict, got {type(actual).__name__}"
        # Note: Nested field validation is complex, left for future enhancement
        return None

    # Check feature type using mapping
    for feature_class, (expected_type, type_name) in _FEATURE_TYPE_MAP.items():
        if isinstance(expected, feature_class):
            if not isinstance(actual, expected_type):
                # For scalar types, show the expected dtype in error
                if isinstance(expected, (ValueFeature, DatasetFeature)):
                    return f"expected scalar type '{expected.dtype}', got {type(actual).__name__}"
                return f"expected {type_name}, got {type(actual).__name__}"

            # Additional validation for scalar types
            if isinstance(expected, (ValueFeature, DatasetFeature)):
                return _validate_scalar_dtype(expected, actual)

            return None

    # Unknown feature type - skip validation
    return None


def validate_dataset_features(
    dataset: Dataset,
    expected_features: Sequence[Union[DatasetFeature, HFFeature]],
    dataset_name: str = "dataset",
) -> None:
    """Validate that the loaded dataset has all required features and types.

    Args:
        dataset: The loaded HuggingFace dataset to validate.
        expected_features: List of feature definitions from InputDataset or OutputDataset.
        dataset_name: Name of the dataset for error messages (default: "dataset").

    Raises:
        ValueError: If required features are missing or types don't match.

    Note:
        Validates both feature existence and types (dtypes for scalars, feature types
        for Image/Audio/Video/List/ClassLabel). Complex nested structures (DictFeature
        fields) are not fully validated.
    """
    if dataset.column_names is None or dataset.features is None:
        logging.getLogger(__name__).debug(
            "Skipping feature validation for dataset '%s': "
            "column_names or features not available (streaming dataset).",
            dataset_name,
        )
        return

    dataset_columns = set(dataset.column_names)
    missing_features = []
    type_mismatches = []

    for feature in expected_features:
        # All feature types have required field (defaults to False)
        required = feature.required

        if feature.name not in dataset_columns:
            if required:
                missing_features.append(feature.name)
        else:
            # Column exists, validate type
            actual_feature = dataset.features[feature.name]
            type_error = _validate_feature_type(feature, actual_feature)
            if type_error:
                type_mismatches.append(f"  - {feature.name}: {type_error}")

    # Report errors
    errors = []
    if missing_features:
        available_columns = ", ".join(sorted(dataset_columns))
        missing_columns = ", ".join(missing_features)
        errors.append(
            f"Missing required features: {missing_columns}\n"
            f"Available columns: {available_columns}"
        )

    if type_mismatches:
        errors.append("Type mismatches:\n" + "\n".join(type_mismatches))

    if errors:
        error_msg = f"Dataset '{dataset_name}' validation failed:\n" + "\n\n".join(
            errors
        )
        error_msg += (
            "\n\nHint: Check your dataset mapping configuration and feature types."
        )
        raise ValueError(error_msg)


def _load_from_hub(
    loader_params: "DatasetLoaderParams",
) -> Dataset | IterableDataset:
    """Load a dataset from HuggingFace Hub.

    Args:
        loader_params: Configuration containing hub_path and optional name, split, revision.

    Returns:
        Dataset: Loaded HuggingFace dataset.

    Security Note:
        hub_path allows loading arbitrary Hub datasets. The trust_remote_code
        flag controls whether remote code execution is allowed.
    """
    token = loader_params.token or os.environ.get("HF_TOKEN")

    return load_dataset(  # nosec B615
        path=loader_params.hub_path,
        name=loader_params.name,
        split=loader_params.split or "train",
        revision=loader_params.revision,
        trust_remote_code=loader_params.trust_remote_code,
        streaming=loader_params.streaming,
        token=token,
    )


def _load_from_local(
    loader_params: "DatasetLoaderParams", input_mount_path: Path | None
) -> Dataset | IterableDataset:
    """Load a dataset from local files.

    Args:
        loader_params: Configuration containing builder_name and data_dir/data_files.
        input_mount_path: Optional path to prepend to relative paths.

    Returns:
        Dataset: Loaded HuggingFace dataset.
    """
    data_dir = loader_params.data_dir
    data_files = loader_params.data_files

    if input_mount_path:
        if data_dir:
            data_dir = (input_mount_path / Path(data_dir)).as_posix()
        elif data_files:
            if isinstance(data_files, list):
                data_files = [
                    (input_mount_path / Path(file)).as_posix() for file in data_files
                ]
            else:
                data_files = (input_mount_path / Path(data_files)).as_posix()

    return load_dataset(  # nosec B615
        path=loader_params.builder_name,
        data_dir=data_dir,
        data_files=data_files,
        split=loader_params.split or "train",
        streaming=loader_params.streaming,
    )


def load_hf_dataset(
    dataset_config: Union[dict, HFDatasetDefinition],
    input_mount_path: Path | None = None,
    expected_features: Sequence[Union[DatasetFeature, HFFeature]] | None = None,
    dataset_name: str = "dataset",
) -> Dataset | IterableDataset:
    # TODO: consider using load_from_disk for caching purposes
    """Load a HuggingFace dataset from Hub or local files.

    Supports two modes:
    - Hub mode: Load from HuggingFace Hub using `hub_path` in loader_params
    - Local mode: Load from local files using `builder_name` with `data_dir` or `data_files`

    Args:
        dataset_config: Configuration for loading the HuggingFace dataset.
        input_mount_path: Optional path to prepend to relative data_files/data_dir paths.
            Typically used in containers to resolve paths relative to the input mount point.
            Only used in local file mode; ignored for Hub datasets.
        expected_features: Optional list of features to validate after loading.
            If provided, validates both feature existence and types (dtypes for scalars,
            feature types for Image/Audio/Video/List/ClassLabel).
        dataset_name: Name of the dataset for validation error messages (default: "dataset").
            Only used when expected_features is provided.

    Returns:
        Dataset: Loaded HuggingFace dataset.

    Raises:
        ValidationError: If dataset_config dict fails Pydantic validation.
        ValueError: If expected_features is provided and validation fails.
    """
    if isinstance(dataset_config, dict):
        dataset_config = HFDatasetDefinition(**dataset_config)
    loader_params = dataset_config.loader_params
    mapping = dataset_config.mapping

    if loader_params.hub_path is not None:
        dataset = _load_from_hub(loader_params)
    else:
        dataset = _load_from_local(loader_params, input_mount_path)

    if mapping:
        dataset = dataset.rename_columns(mapping)

    # Skip feature validation for streaming datasets (IterableDataset) because
    # column_names and features are None until data is materialized.
    if expected_features is not None and not isinstance(dataset, IterableDataset):
        validate_dataset_features(dataset, expected_features, dataset_name)

    return dataset


def verify_txt_file(file_path: str) -> str:
    """Verify that the provided file path points to a valid .txt file.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        str: The validated file path.

    Raises:
        ValueError: If the file is not a .txt file.
    """
    if not file_path.lower().endswith(".txt"):
        raise ValueError(
            f"Unsupported file type: {file_path}. Only .txt files are supported."
        )
    return file_path


def verify_pdf_file(file_path: str) -> str:
    """Verify that the provided file path points to a valid .pdf file.

    Args:
        file_path (str): Path to the .pdf file.

    Returns:
        str: The validated file path.

    Raises:
        ValueError: If the file is not a .pdf file.
    """
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(
            f"Unsupported file type: {file_path}. Only .pdf files are supported."
        )
    return file_path
