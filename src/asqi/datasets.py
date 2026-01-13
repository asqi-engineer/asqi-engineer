from pathlib import Path
from typing import Union

from datasets import Dataset, load_dataset

from asqi.schemas import HFDatasetDefinition


def load_hf_dataset(
    dataset_config: Union[dict, HFDatasetDefinition],
    input_mount_path: Path | None = None,
) -> Dataset:
    # TODO: consider using load_from_disk for caching purposes
    """Load a HuggingFace dataset using the provided loader parameters.

    Args:
        dataset_config: Configuration for loading the HuggingFace dataset.
        input_mount_path: Optional path to prepend to relative data_files/data_dir paths.
            Typically used in containers to resolve paths relative to the input mount point.
            Absolute paths in the dataset config are not modified.

    Returns:
        Dataset: Loaded HuggingFace dataset.

    Raises:
        ValidationError: If dataset_config dict fails Pydantic validation.

    Security Note:
        This function uses local file loaders (json, csv, parquet, etc.) via
        builder_name constrained by Literal types in DatasetLoaderParams.
        The revision parameter is provided for forward compatibility with HF Hub
        datasets, but current usage is limited to local files only.
    """
    if isinstance(dataset_config, dict):
        dataset_config = HFDatasetDefinition(**dataset_config)
    loader_params = dataset_config.loader_params
    mapping = dataset_config.mapping
    # B615: Only local file loaders (json, csv, parquet, etc.) are used via
    # builder_name constrained by Literal type. revision provided for future
    # compatibility with HF Hub datasets but not required for local files.
    #
    # NOTE: split="train" is hardcoded because:
    # 1. For local files (json, parquet, etc.), load_dataset with split=None returns
    #    a DatasetDict containing a single "train" split
    # 2. We want this function to always return a Dataset (not DatasetDict) for simplicity
    # 3. The "train" split is the default convention for single-split datasets in HuggingFace
    if input_mount_path:
        if loader_params.data_dir:
            loader_params.data_dir = (
                input_mount_path / Path(loader_params.data_dir)
            ).as_posix()
        elif loader_params.data_files:
            if isinstance(loader_params.data_files, list):
                loader_params.data_files = [
                    (input_mount_path / Path(file)).as_posix()
                    for file in loader_params.data_files
                ]
            else:
                loader_params.data_files = (
                    input_mount_path / Path(loader_params.data_files)
                ).as_posix()
    dataset = load_dataset(  # nosec B615
        path=loader_params.builder_name,
        data_dir=loader_params.data_dir,
        data_files=loader_params.data_files,
        revision=loader_params.revision,
        split="train",
    )
    dataset = dataset.rename_columns(mapping)
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
