# Example Data Generator Container

A reference implementation demonstrating the Synthetic Data Generation (SDG) workflow in ASQI Engineer. 

## Purpose

This example demonstrates:

1. **Input Dataset Loading** - How to declare and load datasets using ASQI's infrastructure
2. **Data Transformation** - Simple augmentation techniques (can be extended with LLM-based generation)
3. **Output Dataset Creation** - Proper format for saving and returning generated datasets
4. **Container Interface** - Standard interface for data generation containers

## What It Does

The container takes a small dataset of product reviews and creates augmented versions by:
- Converting text to uppercase/lowercase
- Adding prefixes
- Modifying punctuation
- Creating simple variations

Each output sample is marked as either original or synthetic, with tracking of which original sample it was derived from.

## Container Structure

### Files

- `manifest.yaml` - Declares inputs (datasets, parameters, optional systems) and outputs (datasets, metrics)
- `entrypoint.py` - Python script implementing the data generation logic using ASQI library utilities
- `Dockerfile` - Container image definition using `uv` for dependency management
- `pyproject.toml` - Modern Python project configuration with `uv`

## Usage

### Prerequisites

1. Build the container:
```bash
cd sdg_containers/example_data_generator
docker build -t asqiengineer/sdg-container:example_data_generator-latest .
```

**Note:** The container uses `uv` for fast dependency management and installs the unpublished `asqi-engineer` library directly from GitHub. The Dockerfile includes git installation to support this.

### Minimal Usage (No Systems Required)

The container works without any systems - pure data transformation:

```bash
asqi generate-dataset \
  --generation-config config/generation/example_generator.yaml \
  --datasets-config config/datasets/demo_datasets.yaml
```

### With Optional Systems (Not Required for This Example)

This example demonstrates pure data transformation without LLM systems. If you want to add optional systems, you can provide a systems config:

```bash
asqi generate-dataset \
  --generation-config config/generation/example_generator.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --datasets-config config/datasets/demo_datasets.yaml
```

## Configuration Files

### Sample Input Data

Location: `config/datasets/sample_reviews.json`

A small dataset with 5 product reviews (text + label):
```json
[
  {
    "text": "The product quality exceeded my expectations.",
    "label": "positive"
  },
  ...
]
```

### Datasets Configuration

Location: `config/datasets/demo_datasets.yaml`

Defines the reusable dataset:
```yaml
datasets:
  sample_reviews:
    type: "huggingface"
    description: "Small sample dataset of product reviews for demonstration"
    loader_params:
      builder_name: "json"
      data_files: "sample_reviews.json"
    tags:
      - "example"
      - "reviews"
```

### Generation Configuration

Location: `config/generation/example_generator.yaml`

Defines the generation job:
```yaml
job_name: "Example Data Augmentation"

generation_jobs:
  - id: "review_augmentation"
    name: "Review Data Augmentation"
    image: "asqiengineer/sdg-container:example_data_generator-latest"
    input_datasets:
      source_data: sample_reviews  # Maps to dataset registry
    params:
      num_variations: 2
      augmentation_type: "simple"
    volumes:
      input: "config/datasets/"
      output: "output/example_data_generator/"
```

## Output

### Generated Datasets

The container saves augmented data to the output volume and returns metadata:

```json
{
  "test_results": {
    "success": true,
    "original_count": 5,
    "generated_count": 10,
    "total_count": 15,
    "augmentation_type": "simple",
    "execution_time_seconds": 0.45
  },
  "generated_datasets": [
    {
      "dataset_name": "augmented_data",
      "dataset_type": "huggingface",
      "dataset_path": "/output/datasets/augmented_data.parquet",
      "format": "parquet",
      "num_rows": 15,
      "num_columns": 4,
      "metadata": {
        "num_rows": 15,
        "num_columns": 4,
        "columns": [
          "text",
          "label",
          "is_synthetic",
          "source_index"
        ]
      }
    }
  ]
}
```

### Output Dataset Schema

The generated HuggingFace dataset includes:
- `text`: Original or augmented text
- `label`: Category/sentiment label
- `is_synthetic`: Boolean flag (False for original, True for generated)
- `source_index`: Index of the original sample this was derived from

## Key Concepts for Container Authors

### 1. Loading Input Datasets

This example uses ASQI's `load_hf_dataset` utility for consistent dataset loading. The dataset paths are already resolved by ASQI's workflow system:

```python
from asqi.datasets import load_hf_dataset

generation_params = json.loads(args.generation_params)
input_datasets = generation_params.get("input_datasets", {})
source_config = input_datasets["source_data"]

# Use ASQI's utility - paths are already resolved
dataset = load_hf_dataset(source_config)
```

### 2. Saving Output Datasets

Use ASQI's `GeneratedDataset` schema for type-safe output:

```python
from asqi.response_schemas import GeneratedDataset

output_mount_path = Path(os.environ["OUTPUT_MOUNT_PATH"])
datasets_dir = output_mount_path / "datasets"
dataset_path = datasets_dir / "augmented_data.parquet"

# Save dataset
dataset.to_parquet(str(dataset_path))

# Return using GeneratedDataset schema
dataset_metadata = GeneratedDataset(
    dataset_name="augmented_data",
    dataset_type="huggingface",
    dataset_path=str(dataset_path),
    format="parquet",
    metadata={
        "num_rows": len(dataset),
        "num_columns": len(dataset.column_names),
        "columns": dataset.column_names,
    }
)
```

### 3. Container Output Format

Use ASQI's `ContainerOutput` model for type-safe, validated output:

```python
from asqi.response_schemas import ContainerOutput

# Prepare output using ContainerOutput model
container_output = ContainerOutput(  # type: ignore[call-arg]
    results={
        "success": True,
        "total_count": total_count,
        "execution_time": execution_time,
    },
    generated_datasets=[dataset_metadata],  # List of GeneratedDataset objects
)

# Output as JSON
print(json.dumps(container_output.model_dump(), indent=2))
```

The `# type: ignore[call-arg]` comment is needed due to backward compatibility typing (both `results` and `test_results` are optional, but at least one must be provided at runtime).

### 4. Optional vs Required Systems

Systems can be optional by setting `required: false` in manifest. Check if provided:

```python
systems_params = json.loads(args.systems_params) if args.systems_params else {}

if "generation_system" in systems_params:
    # Use LLM-based augmentation
else:
    # Fall back to simple augmentation
```

## Extending This Example

### LLM-Based Augmentation

To implement real LLM-based data generation:

1. Make `generation_system` required in manifest
2. Extract system parameters from `systems_params`
3. Use OpenAI client to call the LLM:

```python
from openai import OpenAI

system_params = systems_params["generation_system"]
client = OpenAI(
    base_url=system_params["base_url"],
    api_key=system_params["api_key"]
)

response = client.chat.completions.create(
    model=system_params["model"],
    messages=[{
        "role": "user",
        "content": f"Generate a variation of: {text}"
    }]
)
```
