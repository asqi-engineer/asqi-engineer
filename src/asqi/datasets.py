from datasets import Dataset, load_dataset

from asqi.schemas import HFDatasetDefinition


def load_hf_dataset(dataset_config: dict) -> Dataset:
    # TODO: consider using load_from_disk for caching purposes
    """Load a HuggingFace dataset using the provided loader parameters.

    Args:
        dataset_config (dict): Configuration dict for loading the HuggingFace dataset,
                              will be parsed as HFDatasetDefinition.

    Returns:
        Dataset: Loaded HuggingFace dataset.

    Security Note:
        This function uses local file loaders (json, csv, parquet, etc.) via
        builder_name constrained by Literal types in DatasetLoaderParams.
        The revision parameter is provided for forward compatibility with HF Hub
        datasets, but current usage is limited to local files only.
    """
    dataset_config = HFDatasetDefinition(**dataset_config)
    loader_params = dataset_config.loader_params
    mapping = dataset_config.mapping
    # B615: Only local file loaders (json, csv, parquet, etc.) are used via
    # builder_name constrained by Literal type. revision provided for future
    # compatibility with HF Hub datasets but not required for local files.
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
