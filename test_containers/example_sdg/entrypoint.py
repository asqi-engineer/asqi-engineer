#!/usr/bin/env python3
"""
Example Synthetic Data Generation (SDG) Container

This container demonstrates the complete SDG workflow:
1. Loading input datasets
2. Performing data augmentation/generation
3. Saving generated datasets
4. Returning proper metadata

This is a simplified educational example - real SDG would use LLMs or
more sophisticated augmentation techniques.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset


def main():
    """Main entrypoint demonstrating the SDG container interface."""
    parser = argparse.ArgumentParser(
        description="Example Synthetic Data Generation container"
    )
    parser.add_argument(
        "--generation-params",
        required=True,
        help="Generation parameters as JSON string",
    )
    parser.add_argument(
        "--systems-params",
        required=False,
        help="Systems parameters as JSON string (optional for this example)",
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        generation_params = json.loads(args.generation_params)
        systems_params = json.loads(args.systems_params) if args.systems_params else {}

        # Extract configuration parameters
        augmentation_factor = generation_params.get("augmentation_factor", 2)
        transformation_type = generation_params.get("transformation_type", "paraphrase")

        # Load input dataset
        print("Loading input dataset...", file=sys.stderr)
        input_dataset = load_input_dataset("training_examples")
        original_count = len(input_dataset)
        print(f"Loaded {original_count} original examples", file=sys.stderr)

        # Generate synthetic data
        print(f"Generating synthetic data (factor: {augmentation_factor})...", file=sys.stderr)
        augmented_dataset = generate_synthetic_data(
            input_dataset,
            augmentation_factor=augmentation_factor,
            transformation_type=transformation_type,
        )
        synthetic_count = len(augmented_dataset) - original_count
        total_count = len(augmented_dataset)

        # Save the generated dataset
        print("Saving augmented dataset...", file=sys.stderr)
        dataset_metadata = save_output_dataset(
            augmented_dataset, "augmented_training_data"
        )

        # Calculate metrics
        augmentation_ratio = synthetic_count / original_count if original_count > 0 else 0

        metrics_data = {
            "original_count": original_count,
            "synthetic_count": synthetic_count,
            "total_count": total_count,
            "augmentation_ratio": round(augmentation_ratio, 2),
        }

        # Prepare output in the expected format
        output = {
            "test_results": metrics_data,
            "generated_datasets": [dataset_metadata],
        }

        # Output results as JSON to stdout
        print(json.dumps(output, indent=2))
        sys.exit(0)

    except json.JSONDecodeError as e:
        error_output = {
            "test_results": {"error": f"Invalid JSON in arguments: {e}"},
            "generated_datasets": [],
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    except Exception as e:
        error_output = {
            "test_results": {"error": f"Unexpected error: {str(e)}"},
            "generated_datasets": [],
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


def load_input_dataset(dataset_name: str) -> Dataset:
    """
    Load input dataset from the mounted INPUT_MOUNT_PATH.

    Args:
        dataset_name: Name of the dataset to load (matches manifest input_datasets name)

    Returns:
        Loaded HuggingFace Dataset

    Note:
        The ASQI framework automatically mounts input datasets to INPUT_MOUNT_PATH
        based on the manifest's input_datasets configuration.
    """
    input_mount_path = os.environ.get("INPUT_MOUNT_PATH")
    if not input_mount_path:
        raise ValueError("INPUT_MOUNT_PATH environment variable not set")

    dataset_path = Path(input_mount_path) / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Ensure the dataset is properly configured in the generation config."
        )

    # Load the dataset (assumes JSON format as configured in the test config)
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(dataset_path / "*.json"), split="train")
    return dataset


def generate_synthetic_data(
    dataset: Dataset,
    augmentation_factor: int = 2,
    transformation_type: str = "paraphrase",
) -> Dataset:
    """
    Generate synthetic data from the input dataset.

    This is a simplified demonstration. In production, you would:
    - Use LLM APIs to generate realistic paraphrases
    - Apply more sophisticated augmentation techniques
    - Ensure diversity and quality of synthetic data

    Args:
        dataset: Input dataset to augment
        augmentation_factor: How many synthetic examples per original
        transformation_type: Type of transformation to apply

    Returns:
        Augmented dataset with original + synthetic examples
    """
    all_examples = []

    # Add original examples with metadata
    for example in dataset:
        all_examples.append({
            "text": example["text"],
            "label": example["label"],
            "is_synthetic": False,
            "source": "original",
        })

    # Generate synthetic examples
    for example in dataset:
        for i in range(augmentation_factor):
            synthetic_text = apply_transformation(
                example["text"], transformation_type, i
            )
            all_examples.append({
                "text": synthetic_text,
                "label": example["label"],
                "is_synthetic": True,
                "source": transformation_type,
            })

    # Create new dataset from augmented examples
    return Dataset.from_list(all_examples)


def apply_transformation(text: str, transformation_type: str, iteration: int) -> str:
    """
    Apply a simple transformation to generate synthetic text.

    In a real SDG system, this would call an LLM API to generate
    natural paraphrases. This demo uses simple string operations.

    Args:
        text: Original text
        transformation_type: Type of transformation
        iteration: Which iteration (for variation)

    Returns:
        Transformed text
    """
    if transformation_type == "uppercase":
        return text.upper()
    elif transformation_type == "reverse":
        return text[::-1]
    elif transformation_type == "paraphrase":
        # Simple pseudo-paraphrase for demo purposes
        # In production: Use LLM API like OpenAI, Anthropic, etc.
        prefixes = [
            "In other words: ",
            "To put it differently: ",
            "Another way to say this: ",
        ]
        prefix = prefixes[iteration % len(prefixes)]
        return f"{prefix}{text}"
    else:
        return f"[{transformation_type}] {text}"


def save_output_dataset(dataset: Dataset, dataset_name: str) -> Dict[str, Any]:
    """
    Save the generated dataset to OUTPUT_MOUNT_PATH and return metadata.

    Args:
        dataset: The dataset to save
        dataset_name: Name of the output dataset (matches manifest output_datasets name)

    Returns:
        Dictionary with dataset metadata for the generated_datasets output field
    """
    output_mount_path = os.environ.get("OUTPUT_MOUNT_PATH")
    if not output_mount_path:
        raise ValueError("OUTPUT_MOUNT_PATH environment variable not set")

    # Create datasets directory in output mount
    datasets_dir = Path(output_mount_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset as JSON (HuggingFace datasets format)
    dataset_dir = datasets_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    output_file = dataset_dir / "data.json"
    dataset.to_json(output_file)

    # Return metadata in the format expected by ASQI
    return {
        "dataset_name": dataset_name,
        "dataset_type": "huggingface",
        "dataset_path": str(dataset_dir),
        "num_examples": len(dataset),
        "features": list(dataset.features.keys()),
    }


if __name__ == "__main__":
    main()
