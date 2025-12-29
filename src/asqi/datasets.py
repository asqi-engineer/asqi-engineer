from datasets import load_dataset, Dataset


def load_hf_dataset(loader_params: dict[str, str]) -> Dataset:
    # TODO: consider using load_from_disk for caching purposes
    """Load a HuggingFace dataset using the provided loader parameters.

    Args:
        loader_params (dict[str, str]): Keyword arguments for datasets.load_dataset function.

    Returns:
        Dataset: Loaded HuggingFace dataset.
    """
    dataset = load_dataset(**loader_params)
    return dataset
