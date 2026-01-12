"""
Example Data Generator - Demonstrates Synthetic Data Generation (SDG) workflow.

This container shows:
1. How to load input datasets using ASQI's infrastructure
2. How to transform/augment data (simple text variations)
3. How to output generated datasets
4. Proper manifest structure for data generation containers
5. Working without required systems (systems are optional)

The container creates simple variations of input text using multiple
transformation strategies: uppercase, prefixes, punctuation, separators, and
word reversal
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# HuggingFace datasets for loading/saving
from datasets import Dataset, load_dataset


def load_input_dataset(
    dataset_config: Dict[str, Any], input_mount_path: Path
) -> Dataset:
    """
    Load a HuggingFace dataset from the input mount.

    This function follows the same pattern as load_hf_dataset() in asqi.datasets,
    but handles path resolution for containerized environments.

    Args:
        dataset_config: Dataset configuration from generation-params containing
                       'loader_params' with builder_name, data_files, etc.
        input_mount_path: Path to the input mount where dataset files are located

    Returns:
        Loaded HuggingFace Dataset

    Raises:
        ValueError: If required loader_params fields are missing

    Note:
        Always returns a Dataset (not DatasetDict) by using split="train".
        This matches the behavior of load_hf_dataset() for consistency.
    """
    loader_params = dataset_config.get("loader_params", {})
    builder_name = loader_params.get("builder_name")
    data_files = loader_params.get("data_files")
    data_dir = loader_params.get("data_dir")

    if not builder_name:
        raise ValueError("Dataset loader_params must include 'builder_name'")

    # Resolve paths relative to input mount
    resolved_data_files = None
    if data_files:
        if isinstance(data_files, str):
            resolved_data_files = str(input_mount_path / data_files)
        elif isinstance(data_files, list):
            resolved_data_files = [str(input_mount_path / f) for f in data_files]

    resolved_data_dir = str(input_mount_path / data_dir) if data_dir else None

    print(f"Loading dataset: builder_name={builder_name}")
    if resolved_data_files:
        print(f"  data_files={resolved_data_files}")
    if resolved_data_dir:
        print(f"  data_dir={resolved_data_dir}")

    # Load with split="train" to always return Dataset (not DatasetDict)
    # This matches load_hf_dataset() behavior for consistency
    dataset = load_dataset(  # nosec B615
        path=builder_name,
        data_dir=resolved_data_dir,
        data_files=resolved_data_files,
        split="train",
    )

    # Apply column mapping if provided
    mapping = dataset_config.get("mapping", {})
    if mapping:
        dataset = dataset.rename_columns(mapping)

    print(f"Loaded {len(dataset)} samples")
    return dataset


def simple_augmentation(text: str, label: str, variation_num: int) -> Dict[str, Any]:
    """
    Create a simple variation of the input text.

    This is a basic demonstration. Real implementations might use:
    - LLM-based paraphrasing
    - Back-translation
    - Synonym replacement
    - More sophisticated NLP techniques

    Args:
        text: Original text
        label: Original label
        variation_num: Which variation (0, 1, 2, etc.)

    Returns:
        Dictionary with augmented sample
    """
    strategies = [
        lambda t: t.upper(),  # Convert to uppercase
        lambda t: f"Example: {t}",  # Add prefix
        lambda t: t.replace(".", "!"),  # Change punctuation
        lambda t: t.replace(" ", " - "),  # Add separators
        lambda t: " ".join(reversed(t.split())),  # Reverse words
    ]

    strategy = strategies[variation_num % len(strategies)]
    augmented_text = strategy(text)

    return {"text": augmented_text, "label": label}


def generate_augmented_dataset(source_dataset: Dataset, num_variations: int) -> Dataset:
    """
    Generate augmented dataset from source data.

    Creates variations of each input sample, marking which are original
    vs synthetic, and tracking the source of each synthetic sample.

    Args:
        source_dataset: Original dataset
        num_variations: How many variations to create per sample

    Returns:
        Augmented dataset with original and synthetic samples
    """
    augmented_samples = []

    print(f"Generating {num_variations} variations per sample...")
    print(f"Source dataset has {len(source_dataset)} samples")

    for idx, sample in enumerate(source_dataset):
        text = sample["text"]
        label = sample["label"]

        # Add original sample
        augmented_samples.append(
            {
                "text": text,
                "label": label,
                "is_synthetic": False,
                "source_index": idx,
            }
        )

        # Generate variations using simple augmentation strategies
        for var_idx in range(num_variations):
            augmented = simple_augmentation(text, label, var_idx)

            augmented_samples.append(
                {
                    "text": augmented["text"],
                    "label": augmented["label"],
                    "is_synthetic": True,
                    "source_index": idx,
                }
            )

    print(f"Generated {len(augmented_samples)} total samples")
    return Dataset.from_list(augmented_samples)


def save_output_dataset(
    dataset: Dataset, output_mount_path: Path, dataset_name: str
) -> Dict[str, Any]:
    """
    Save generated dataset to output mount.

    This demonstrates the expected output format for generated datasets.
    The dataset is saved and metadata is returned for the container output JSON.

    Args:
        dataset: Dataset to save
        output_mount_path: Path to output mount
        dataset_name: Name of the dataset

    Returns:
        Dataset metadata dict for container output
    """
    # Create datasets directory
    datasets_dir = output_mount_path / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset as Parquet file
    dataset_path = datasets_dir / f"{dataset_name}.parquet"
    dataset.to_parquet(str(dataset_path))

    print(f"Saved dataset to: {dataset_path}")

    # Return metadata in expected format following GeneratedDataset schema
    return {
        "dataset_name": dataset_name,
        "dataset_type": "huggingface",
        "dataset_path": str(dataset_path),
        "format": "parquet",
        "metadata": {
            "num_rows": len(dataset),
            "num_columns": len(dataset.column_names),
            "columns": dataset.column_names,
        },
    }


def main():
    """
    Main entrypoint demonstrating data generation container interface.

    This example performs pure data transformation without requiring any LLM systems,
    showcasing that SDG containers can work independently when using non-LLM
    augmentation techniques.

    Expected arguments:
    - --generation-params: JSON string with generation parameters and input datasets

    Expected output:
    - JSON with test_results (metrics) and generated_datasets (list of dataset info)
    """
    parser = argparse.ArgumentParser(
        description="Example data generation container demonstrating SDG workflow"
    )
    parser.add_argument(
        "--generation-params",
        required=True,
        help="Generation parameters as JSON string (includes input_datasets)",
    )

    args = parser.parse_args()
    start_time = time.time()

    try:
        # Parse inputs
        generation_params = json.loads(args.generation_params)

        # Get parameters
        num_variations = generation_params.get("num_variations", 2)

        # Get input datasets from generation_params
        input_datasets = generation_params.get("input_datasets", {})
        if "source_data" not in input_datasets:
            raise ValueError(
                "Missing required input dataset 'source_data' in generation_params"
            )

        # Get mount paths from environment
        input_mount_path = Path(os.environ.get("INPUT_MOUNT_PATH", "/input"))
        output_mount_path = Path(os.environ.get("OUTPUT_MOUNT_PATH", "/output"))

        print("=" * 60)
        print("Example Data Generator - Starting")
        print("=" * 60)
        print(f"Num variations: {num_variations}")
        print(f"Input mount: {input_mount_path}")
        print(f"Output mount: {output_mount_path}")
        print("=" * 60)

        # Step 1: Load input dataset
        print("\n[1/3] Loading input dataset...")
        source_config = input_datasets["source_data"]
        source_dataset = load_input_dataset(source_config, input_mount_path)
        print(f"Loaded {len(source_dataset)} samples from source_data")

        # Step 2: Generate augmented data
        print("\n[2/3] Generating augmented dataset...")
        augmented_dataset = generate_augmented_dataset(source_dataset, num_variations)
        original_count = len(source_dataset)
        generated_count = len(augmented_dataset) - original_count
        total_count = len(augmented_dataset)

        # Step 3: Save output dataset
        print("\n[3/3] Saving output dataset...")
        dataset_metadata = save_output_dataset(
            augmented_dataset, output_mount_path, "augmented_data"
        )

        execution_time = time.time() - start_time

        # Prepare output in expected format (matches manifest output_metrics)
        output = {
            "test_results": {
                "success": True,
                "total_count": total_count,
            },
            "generated_datasets": [dataset_metadata],
        }

        print("\n" + "=" * 60)
        print("Data Generation Complete!")
        print("=" * 60)
        print(f"Original samples: {original_count}")
        print(f"Generated samples: {generated_count}")
        print(f"Total samples: {total_count}")
        print(f"Execution time: {execution_time:.2f}s")
        print("=" * 60)

        # Output results as JSON
        print(json.dumps(output, indent=2))
        sys.exit(0)

    except Exception as e:
        error_output = {
            "test_results": {
                "success": False,
                "total_count": 0,
                "error": str(e),
            },
            "generated_datasets": [],
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
