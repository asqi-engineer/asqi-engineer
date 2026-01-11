# Example Data Generator Container

A reference implementation demonstrating the Synthetic Data Generation (SDG) workflow in ASQI Engineer. This container shows best practices for creating data generation containers that load input datasets, transform/augment data, and output generated datasets.

## Purpose

This example demonstrates:

1. **Input Dataset Loading** - How to declare and load datasets using ASQI's infrastructure
2. **Data Transformation** - Simple augmentation techniques (can be extended with LLM-based generation)
3. **Output Dataset Creation** - Proper format for saving and returning generated datasets
4. **Container Interface** - Standard interface for data generation containers
5. **Optional Systems** - How containers can work without required systems (pure data transformation)

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
- `entrypoint.py` - Python script implementing the data generation logic
- `Dockerfile` - Container image definition
- `requirements.txt` - Python dependencies (datasets library)

### Manifest Highlights

**Input Datasets:**
```yaml
input_datasets:
  - name: "source_data"
    type: "huggingface"
    required: true
    features:
      - name: "text"
        dtype: "string"
      - name: "label"
        dtype: "string"
```

**Output Datasets:**
```yaml
output_datasets:
  - name: "augmented_data"
    type: "huggingface"
    features:
      - name: "text"
        dtype: "string"
      - name: "label"
        dtype: "string"
      - name: "is_synthetic"
        dtype: "bool"
      - name: "source_index"
        dtype: "int32"
```

**Optional Systems:**
```yaml
input_systems:
  - name: "generation_system"
    type: "llm_api"
    required: false  # Container works without it
```

## Usage

### Prerequisites

1. Build the container:
```bash
cd test_containers/example_data_generator
docker build -t example_data_generator:latest .
```

### Minimal Usage (No Systems Required)

The container works without any systems - pure data transformation:

```bash
asqi generate-dataset \
  --generation-config config/examples/generation_suite.yaml \
  --datasets-config config/examples/datasets.yaml
```

### With Optional Systems

You can provide systems even though they're optional:

```bash
asqi generate-dataset \
  --generation-config config/examples/generation_suite.yaml \
  --systems-config config/examples/systems.yaml \
  --datasets-config config/examples/datasets.yaml
```

## Configuration Files

### Sample Input Data

Location: `test_data/example_data_generator/sample_reviews.json`

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

Location: `config/examples/datasets.yaml`

Defines the reusable dataset:
```yaml
datasets:
  sample_reviews:
    type: "huggingface"
    loader_params:
      builder_name: "json"
      data_files: "sample_reviews.json"
```

### Generation Suite Configuration

Location: `config/examples/generation_suite.yaml`

Defines the generation job:
```yaml
generation_jobs:
  - id: "review_augmentation"
    image: "example_data_generator:latest"
    input_datasets:
      source_data: sample_reviews  # Maps to dataset registry
    params:
      num_variations: 2
      augmentation_type: "simple"
    volumes:
      input: "test_data/example_data_generator/"
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
      "dataset_path": "/output/datasets/augmented_data",
      "num_rows": 15,
      "num_columns": 4,
      "columns": ["text", "label", "is_synthetic", "source_index"]
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

Input datasets are passed via `--generation-params` with loader configuration:

```python
generation_params = json.loads(args.generation_params)
input_datasets = generation_params.get("input_datasets", {})
source_config = input_datasets["source_data"]

# Load using HuggingFace datasets library
loader_params = source_config.get("loader_params", {})
dataset = load_dataset(
    path=loader_params["builder_name"],
    data_files=str(input_mount_path / loader_params["data_files"])
)
```

### 2. Saving Output Datasets

Save to `/output` mount and return metadata:

```python
output_mount_path = Path(os.environ["OUTPUT_MOUNT_PATH"])
datasets_dir = output_mount_path / "datasets"
dataset_path = datasets_dir / "augmented_data"

# Save dataset
dataset.save_to_disk(str(dataset_path))

# Return in generated_datasets
{
    "dataset_name": "augmented_data",
    "dataset_type": "huggingface", 
    "dataset_path": str(dataset_path),
    "num_rows": len(dataset),
    ...
}
```

### 3. Container Output Format

Must return JSON with two fields:

```python
output = {
    "test_results": {
        # Metrics matching manifest output_metrics
        "success": True,
        "original_count": 5,
        ...
    },
    "generated_datasets": [
        # List of dataset metadata dicts
        {...}
    ]
}
```

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

### Different Dataset Types

The example uses HuggingFace datasets, but you can also work with:
- PDF documents (`type: "pdf"`)
- Text files (`type: "txt"`)

### Advanced Features

- Multiple input datasets
- Multiple output datasets
- Complex transformations
- Quality metrics and validation
- Progress reporting for long-running jobs

## Related Documentation

- [Data Generation Pipeline](../../../docs/data-generation.md) - Full pipeline documentation
- [Container Development Guide](../../../docs/container-development.md) - How to build containers
- [Manifest Schema](../../src/asqi/schemas/asqi_manifest.schema.json) - Complete schema reference

## License

Same as parent project (see LICENSE file in repository root).
