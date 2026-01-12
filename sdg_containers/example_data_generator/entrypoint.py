"""
Example Data Generator - Demonstrates Synthetic Data Generation (SDG) workflow.

This container shows:
1. How to load input datasets using ASQI's infrastructure
2. How to transform/augment data (simple text variations)
3. How to output generated datasets
4. Proper manifest structure for data generation containers
5. Working without required systems (systems are optional)

The container creates simple variations of input text by:
- Converting to uppercase/lowercase
- Adding prefixes
- Creating paraphrases (simple pattern-based)
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

    This demonstrates how containers should load datasets passed by ASQI.
    The dataset configuration comes from generation-params and includes
    the loader parameters (builder_name, data_files, etc.).

    Args:
        dataset_config: Dataset configuration from generation-params
        input_mount_path: Path to the input mount

    Returns:
        Loaded HuggingFace Dataset
    """
    loader_params = dataset_config.get("loader_params", {})
    builder_name = loader_params.get("builder_name")
    data_files = loader_params.get("data_files")
    data_dir = loader_params.get("data_dir")

    if not builder_name:
        raise ValueError("Dataset loader_params must include 'builder_name'")

    # Construct full paths relative to input mount
    load_kwargs = {"path": builder_name}

    if data_files:
        if isinstance(data_files, str):
            load_kwargs["data_files"] = str(input_mount_path / data_files)
        elif isinstance(data_files, list):
            load_kwargs["data_files"] = [str(input_mount_path / f) for f in data_files]

    if data_dir:
        load_kwargs["data_dir"] = str(input_mount_path / data_dir)

    print(f"Loading dataset with: {load_kwargs}")
    dataset = load_dataset(**load_kwargs)

    # Handle DatasetDict (load_dataset can return dict or Dataset)
    if hasattr(dataset, "keys"):
        # If it's a DatasetDict, take the first split
        first_split = list(dataset.keys())[0]
        print(f"Using split: {first_split}")
        dataset = dataset[first_split]

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


def generate_augmented_dataset(
    source_dataset: Dataset, num_variations: int, augmentation_type: str
) -> Dataset:
    """
    Generate augmented dataset from source data.

    Creates variations of each input sample, marking which are original
    vs synthetic, and tracking the source of each synthetic sample.

    Args:
        source_dataset: Original dataset
        num_variations: How many variations to create per sample
        augmentation_type: Type of augmentation to use

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

        # Generate variations
        for var_idx in range(num_variations):
            if augmentation_type == "llm":
                # In a real implementation, this would call an LLM
                # For this example, we'll use simple augmentation
                # to keep it working without required systems
                augmented = simple_augmentation(text, label, var_idx)
            else:
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

    # Save dataset
    dataset_path = datasets_dir / dataset_name
    dataset.save_to_disk(str(dataset_path))

    print(f"Saved dataset to: {dataset_path}")

    # Return metadata in expected format
    return {
        "dataset_name": dataset_name,
        "dataset_type": "huggingface",
        "dataset_path": str(dataset_path),
        "num_rows": len(dataset),
        "num_columns": len(dataset.column_names),
        "columns": dataset.column_names,
    }


def main():
    """
    Main entrypoint demonstrating data generation container interface.

    Expected arguments:
    - --generation-params: JSON string with generation parameters and input datasets
    - --systems-params: JSON string with system configurations (OPTIONAL)

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
    parser.add_argument(
        "--systems-params",
        required=False,
        help="Systems parameters as JSON string (OPTIONAL - shows systems can be optional)",
    )

    args = parser.parse_args()
    start_time = time.time()

    try:
        # Parse inputs
        generation_params = json.loads(args.generation_params)
        systems_params = json.loads(args.systems_params) if args.systems_params else {}

        # Get parameters
        num_variations = generation_params.get("num_variations", 2)
        augmentation_type = generation_params.get("augmentation_type", "simple")

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
        print(f"Augmentation type: {augmentation_type}")
        print(f"Input mount: {input_mount_path}")
        print(f"Output mount: {output_mount_path}")
        print(f"Systems provided: {list(systems_params.keys())}")
        print("=" * 60)

        # Step 1: Load input dataset
        print("\n[1/3] Loading input dataset...")
        source_config = input_datasets["source_data"]
        source_dataset = load_input_dataset(source_config, input_mount_path)
        print(f"Loaded {len(source_dataset)} samples from source_data")

        # Step 2: Generate augmented data
        print("\n[2/3] Generating augmented dataset...")
        augmented_dataset = generate_augmented_dataset(
            source_dataset, num_variations, augmentation_type
        )
        original_count = len(source_dataset)
        generated_count = len(augmented_dataset) - original_count
        total_count = len(augmented_dataset)

        # Step 3: Save output dataset
        print("\n[3/3] Saving output dataset...")
        dataset_metadata = save_output_dataset(
            augmented_dataset, output_mount_path, "augmented_data"
        )

        execution_time = time.time() - start_time

        # Prepare output in expected format
        output = {
            "test_results": {
                "success": True,
                "original_count": original_count,
                "generated_count": generated_count,
                "total_count": total_count,
                "augmentation_type": augmentation_type,
                "execution_time_seconds": round(execution_time, 2),
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
                "error": str(e),
                "original_count": 0,
                "generated_count": 0,
                "total_count": 0,
                "augmentation_type": "none",
                "execution_time_seconds": time.time() - start_time,
            },
            "generated_datasets": [],
        }
        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
