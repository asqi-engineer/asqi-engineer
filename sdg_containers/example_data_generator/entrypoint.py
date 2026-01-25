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

import yaml
from datasets import Dataset

from asqi.datasets import load_hf_dataset, validate_dataset_features
from asqi.response_schemas import ContainerOutput
from asqi.schemas import Manifest


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
        text = sample["text"]  # type: ignore[index]
        label = sample["label"]  # type: ignore[index]

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


def main():
    """
    Main entrypoint demonstrating data generation container interface.

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
        manifest_path = Path("/app/manifest.yaml")
        with open(manifest_path) as f:
            manifest_dict = yaml.safe_load(f)
        manifest = Manifest(**manifest_dict)

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

        # Step 1: Load and validate input dataset
        print("\n[1/4] Loading and validating input dataset...")
        source_config = input_datasets["source_data"]
        input_spec = manifest.input_datasets[0]  # "source_data"

        source_dataset = load_hf_dataset(
            source_config,
            input_mount_path=input_mount_path,
            expected_features=input_spec.features,  # Used for validation if provided
            dataset_name="source_data",
        )
        print(
            f"✓ Loaded and validated {len(source_dataset)} samples from 'source_data'"
        )

        # Step 2: Generate augmented data
        print("\n[2/4] Generating augmented dataset...")
        augmented_dataset = generate_augmented_dataset(source_dataset, num_variations)
        original_count = len(source_dataset)
        generated_count = len(augmented_dataset) - original_count
        total_count = len(augmented_dataset)

        # Step 3: Validate and save output dataset
        print("\n[3/4] Validating and saving output dataset...")
        output_spec = manifest.output_datasets[0]  # "augmented_data"

        # Validate against manifest schema
        if output_spec.features is not None:
            validate_dataset_features(
                augmented_dataset,
                expected_features=output_spec.features,
                dataset_name="augmented_data",
            )
            print("✓ Validated output dataset against manifest schema")

        # Save dataset
        datasets_dir = output_mount_path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = datasets_dir / "augmented_data.parquet"
        augmented_dataset.to_parquet(str(dataset_path))
        print(f"✓ Saved dataset to: {dataset_path}")

        # Create metadata for output
        from asqi.response_schemas import GeneratedDataset

        dataset_metadata = GeneratedDataset(
            dataset_name="augmented_data",
            dataset_type="huggingface",
            dataset_path=str(dataset_path),
            format="parquet",
            metadata={
                "num_rows": len(augmented_dataset),
                "num_columns": len(augmented_dataset.column_names),
                "columns": augmented_dataset.column_names,
            },
        )

        # Step 4: Prepare output
        print("\n[4/4] Preparing output...")
        execution_time = time.time() - start_time

        # Prepare output using ASQI's ContainerOutput model for type safety
        container_output = ContainerOutput(  # type: ignore[call-arg]
            results={
                "success": True,
                "total_count": total_count,
                "execution_time": execution_time,
            },
            generated_datasets=[dataset_metadata],
        )

        print("\n" + "=" * 60)
        print("✓ Data Generation Complete!")
        print("=" * 60)
        print(f"Original samples: {original_count}")
        print(f"Generated samples: {generated_count}")
        print(f"Total samples: {total_count}")
        print(f"Execution time: {execution_time:.2f}s")
        print("=" * 60)

        # Output results as JSON using model_dump()
        print(json.dumps(container_output.model_dump(), indent=2))
        sys.exit(0)

    except Exception as e:
        error_output = ContainerOutput(  # type: ignore[call-arg]
            results={
                "success": False,
                "total_count": 0,
                "error": str(e),
            },
            generated_datasets=[],
        )
        print(json.dumps(error_output.model_dump(), indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
