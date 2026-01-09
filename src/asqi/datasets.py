from datasets import Dataset, load_dataset

from asqi.schemas import HFDatasetConfig

def load_hf_dataset(dataset_config: HFDatasetConfig) -> Dataset:
    # TODO: consider using load_from_disk for caching purposes
    """Load a HuggingFace dataset using the provided loader parameters.

    Args:
        dataset_config (HFDatasetConfig): Configuration for loading the HuggingFace dataset.

    Returns:
        Dataset: Loaded HuggingFace dataset.
    """
    loader_params = dataset_config.loader_params
    mapping = dataset_config.mapping
    dataset = load_dataset(
        path=loader_params.builder_name,
        data_dir=loader_params.data_dir,
        data_files=loader_params.data_files,
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