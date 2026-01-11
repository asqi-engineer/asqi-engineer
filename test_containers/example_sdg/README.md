# Example SDG Container

This container demonstrates the complete **Synthetic Data Generation (SDG)** workflow in ASQI Engineer. It serves as an educational reference for building custom data generation containers.

## Overview

The example-sdg container showcases four core SDG capabilities:

1. **Input Dataset Loading** - Declares and loads input datasets through the manifest and generation config
2. **Data Augmentation** - Performs simple transformations to generate synthetic data
3. **Output Dataset Creation** - Saves generated datasets with proper metadata
4. **Container Interface** - Implements the standard SDG container interface

## What This Container Does

This container takes a small dataset of labeled text examples and augments it by generating synthetic variants. While this example uses simple string transformations for clarity, real-world SDG containers would use LLM APIs to generate high-quality synthetic data.

### Transformation Types

- **paraphrase** (default): Adds simple prefixes to simulate paraphrasing
- **uppercase**: Converts text to uppercase
- **reverse**: Reverses the text string

> **Note**: In production, you would replace these with LLM-based paraphrasing, back-translation, or other sophisticated augmentation techniques.

## Container Structure

```
example_sdg/
├── manifest.yaml              # Container manifest with input/output schemas
├── entrypoint.py             # Main container logic
├── Dockerfile                # Container image definition
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── example_data/             # Sample input data
│   └── training_examples/
│       └── data.json
└── example_configs/          # Example configuration files
    ├── generation_config.yaml
    └── datasets_config.yaml
```

## Manifest Configuration

The `manifest.yaml` defines:

### Input Datasets
```yaml
input_datasets:
  - name: "training_examples"
    type: "huggingface"
    required: true
    features:
      - name: "text"
        dtype: "string"
      - name: "label"
        dtype: "string"
```

### Output Datasets
```yaml
output_datasets:
  - name: "augmented_training_data"
    type: "huggingface"
    features:
      - name: "text"
        dtype: "string"
      - name: "label"
        dtype: "string"
      - name: "is_synthetic"
        dtype: "bool"
      - name: "source"
        dtype: "string"
```

### Generation Parameters
```yaml
input_schema:
  - name: "augmentation_factor"
    type: "integer"
    description: "How many synthetic examples per original (default: 2)"
  - name: "transformation_type"
    type: "string"
    description: "Type of transformation: 'paraphrase', 'uppercase', 'reverse'"
```

### Output Metrics
The container reports:
- `original_count`: Number of original examples
- `synthetic_count`: Number of generated examples
- `total_count`: Total examples in output
- `augmentation_ratio`: Ratio of synthetic to original

## Container Interface

### Command-line Arguments

**Required:**
- `--generation-params`: JSON string with generation parameters

**Optional:**
- `--systems-params`: JSON string with system configurations (for LLM-based generation)

### Environment Variables

**Required:**
- `INPUT_MOUNT_PATH`: Path where input datasets are mounted
- `OUTPUT_MOUNT_PATH`: Path where output datasets should be saved

### Output Format

The container outputs JSON to stdout:

```json
{
  "test_results": {
    "original_count": 5,
    "synthetic_count": 10,
    "total_count": 15,
    "augmentation_ratio": 2.0
  },
  "generated_datasets": [
    {
      "dataset_name": "augmented_training_data",
      "dataset_type": "huggingface",
      "dataset_path": "/path/to/output/datasets/augmented_training_data",
      "num_examples": 15,
      "features": ["text", "label", "is_synthetic", "source"]
    }
  ]
}
```

## Usage

### 1. Build the Container

```bash
cd test_containers/example_sdg
docker build -t example_sdg:latest .
```

Or use the build script:

```bash
./test_containers/build_all.sh
```

### 2. Configure Datasets

Create a `datasets_config.yaml`:

```yaml
datasets:
  example_training_data:
    type: "huggingface"
    description: "Sample training examples"
    loader_params:
      builder_name: "json"
      data_files: "test_containers/example_sdg/example_data/training_examples/data.json"
    mapping:
      text: "text"
      label: "label"
```

### 3. Create Generation Config

Create a `generation_config.yaml`:

```yaml
job_name: "Example SDG - Text Augmentation"

generation_jobs:
  - id: "text_augmentation_demo"
    name: "Text Augmentation Demo"
    image: "example_sdg:latest"

    input_datasets:
      training_examples: example_training_data

    output_datasets:
      augmented_training_data:
        description: "Augmented training data with synthetic examples"

    params:
      augmentation_factor: 2
      transformation_type: "paraphrase"
```

### 4. Run the Generation Job

```bash
asqi generate-dataset \
  --generation-config test_containers/example_sdg/example_configs/generation_config.yaml \
  --datasets-config test_containers/example_sdg/example_configs/datasets_config.yaml
```

## Output

The container creates:

```
output/
└── datasets/
    └── augmented_training_data/
        └── data.json
```

The output dataset includes:
- All original examples (with `is_synthetic: false`)
- Generated synthetic examples (with `is_synthetic: true`)
- Metadata fields for tracking data provenance

### Sample Output Data

```json
[
  {
    "text": "The weather is nice today.",
    "label": "weather",
    "is_synthetic": false,
    "source": "original"
  },
  {
    "text": "In other words: The weather is nice today.",
    "label": "weather",
    "is_synthetic": true,
    "source": "paraphrase"
  },
  ...
]
```

## Extending This Example

To build a production SDG container:

### 1. Add LLM Integration

Replace the simple transformations with LLM API calls:

```python
import openai

def generate_paraphrase(text, system_params):
    client = openai.OpenAI(
        base_url=system_params["base_url"],
        api_key=os.getenv("API_KEY")
    )

    response = client.chat.completions.create(
        model=system_params["model"],
        messages=[
            {"role": "system", "content": "Paraphrase the following text naturally:"},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content
```

### 2. Add Quality Validation

Implement checks to ensure synthetic data quality:
- Semantic similarity scoring
- Label consistency validation
- Diversity metrics

### 3. Support Multiple Output Formats

Save datasets in various formats:
- HuggingFace datasets (Parquet)
- CSV for tabular data
- JSONL for streaming

### 4. Add Progress Reporting

For large datasets, add progress indicators:
```python
from tqdm import tqdm

for example in tqdm(dataset, desc="Generating synthetic data"):
    # ... generate synthetic examples
```

## Key Concepts

### Dataset Mounting

ASQI automatically handles dataset mounting:
- Input datasets → `INPUT_MOUNT_PATH/{dataset_name}/`
- Output datasets → `OUTPUT_MOUNT_PATH/datasets/{dataset_name}/`

### Metadata Tracking

Always include provenance metadata:
- `is_synthetic`: Boolean flag
- `source`: Generation method/model
- `original_id`: Reference to source example (for advanced use)

### Error Handling

Return proper error information:
```python
{
  "test_results": {"error": "Description of error"},
  "generated_datasets": []
}
```

## Design Philosophy

This container prioritizes:

1. **Simplicity** - Clear, easy-to-understand code
2. **Educational Value** - Demonstrates workflow patterns
3. **Independence** - Works without external dependencies
4. **Extensibility** - Easy to adapt for real use cases

## Related Documentation

- [ASQI Manifest Schema](../../src/asqi/schemas/asqi_manifest.schema.json)
- [Dataset Configuration](../../docs/datasets.md) _(if exists)_
- Main README: [../../README.md](../../README.md)

## Questions?

For issues or questions:
- Check the main ASQI documentation
- Review other test containers for patterns
- File an issue on the ASQI repository

---

**Remember**: This is a teaching example. Production SDG containers should use proper LLM APIs, implement quality controls, and handle edge cases robustly.
