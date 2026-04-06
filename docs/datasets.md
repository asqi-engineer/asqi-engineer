# Dataset Support and Data Generation

ASQI Engineer provides comprehensive dataset support for both testing and synthetic data generation workflows. This enables test containers to consume structured datasets and generate new datasets as outputs.

## Overview

Dataset support in ASQI includes:

- **Input Datasets**: Feed evaluation data, source documents, or training examples to test containers
- **Output Datasets**: Generate synthetic data, augmented datasets, or processed results
- **Dataset Registry**: Centralize dataset definitions for reuse across test suites and generation jobs
- **Multiple Formats**: Support for HuggingFace datasets, PDF documents, and text files
- **Data Generation Pipeline**: Create synthetic data using dedicated data generation containers

## Input Datasets

Input datasets allow test containers to consume structured data for evaluation, testing, or data generation.

### Supported Dataset Types

ASQI supports three dataset types:

1. **HuggingFace Datasets** (`huggingface`): Structured tabular data loaded using HuggingFace datasets library
2. **PDF Documents** (`pdf`): PDF files for document-based testing or RAG data generation
3. **Text Files** (`txt`): Plain text files for simple text-based inputs


### Dataset Registry Configuration

Create a centralized dataset registry using the `--datasets-config` flag. This promotes reusability across multiple test suites and generation jobs.

**Example: `datasets_registry.yaml`**

```yaml
# yaml-language-server: $schema=../src/asqi/schemas/asqi_datasets_config.schema.json

datasets:
  # HuggingFace dataset with column mapping
  eval_questions:
    type: "huggingface"
    description: "Evaluation questions for testing"
    loader_params:
      builder_name: "json"
      data_files: "eval_questions.json"
    mapping:
      question: "prompt"      # Maps 'question' in dataset to 'prompt' expected by container
      answer: "response"      # Maps 'answer' to 'response'
    tags: ["evaluation", "en"]

  # PDF document dataset
  company_handbook:
    type: "pdf"
    description: "Company policy handbook for RAG testing"
    file_path: "handbook.pdf"
    tags: ["rag", "documents"]
  
  # Text file dataset
  product_descriptions:
    type: "txt"
    description: "Product description corpus"
    file_path: "products.txt"
    tags: ["generation", "source"]
```

### HuggingFace Dataset Configuration

HuggingFace datasets require `loader_params` specifying how to load the dataset:

```yaml
datasets:
  custom_dataset:
    type: "huggingface"
    description: "Custom evaluation dataset"
    loader_params:
      builder_name: "json"              # Dataset format: json, csv, parquet, etc.
      data_files: "my_data.json"        # File path (relative to input mount)
      # OR use data_dir for multiple files:
      # data_dir: "dataset_folder/"
    mapping:
      # Map dataset columns to expected feature names
      input_text: "prompt"
      output_text: "response"
    tags: ["custom", "evaluation"]
```

**Supported `builder_name` values:**
- `json` - JSON files
- `csv` - CSV files
- `parquet` - Parquet files
- `arrow` - Apache Arrow files
- `text` - Plain text files (line-by-line)
- `xml` - XML files
- `imagefolder` - Image datasets
- `audiofolder` - Audio datasets
- `videofolder` - Video datasets

### Column Mapping

Column mapping aligns dataset fields with container expectations. The container manifest defines required features, and the mapping translates actual dataset column names to these required names.

**Container Manifest (`manifest.yaml`):**
```yaml
input_datasets:
  - name: "evaluation_data"
    type: "huggingface"
    required: true
    features:
      - name: "prompt"
        dtype: "string"
      - name: "response"
        dtype: "string"
```

**Dataset Registry (`datasets_registry.yaml`):**
```yaml
datasets:
  my_eval_data:
    type: "huggingface"
    loader_params:
      builder_name: "json"
      data_files: "questions.json"
    mapping:
      question: "prompt"      # Dataset has 'question', container expects 'prompt'
      answer: "response"      # Dataset has 'answer', container expects 'response'
```

**Test Suite (`test_suite.yaml`):**
```yaml
test_suite:
  - id: "eval_test"
    image: "my-test-container:latest"
    input_datasets:
      evaluation_data: "my_eval_data"  # References dataset registry
    volumes:
      input: "path/to/datasets/"
      output: "path/to/output/"
```

### Using Input Datasets in Test Suites

Reference datasets from your registry in test suite configurations:

```yaml
# yaml-language-server: $schema=../src/asqi/schemas/asqi_suite_config.schema.json

suite_name: "Dataset-based Evaluation"
test_suite:
  - id: "quality_test"
    name: "Quality Evaluation Test"
    image: "my-registry/evaluator:latest"
    systems_under_test: ["my_llm"]
    input_datasets:
      evaluation_data: "eval_questions"     # Reference from registry
      source_docs: "company_handbook"       # Another dataset reference
    volumes:
      input: "input/datasets/"
      output: "output/results/"
```

**Running with datasets:**

```bash
asqi execute-tests \
  --test-suite-config test_suite.yaml \
  --systems-config systems.yaml \
  --datasets-config datasets_registry.yaml \
  --output-file results.json
```

### Volume Mounting

When using input datasets, you must provide volume mounts:

- **`input`**: Directory containing dataset files (relative to current directory or absolute path)
- **`output`**: Directory for test container outputs

Dataset file paths in the registry are relative to the `input` mount point.

## Output Datasets

Containers can generate and return datasets as outputs. This is useful for synthetic data generation, dataset augmentation, or data preprocessing workflows.

### Declaring Output Datasets in Manifests

Test containers declare output datasets in their `manifest.yaml`:

```yaml
# Container manifest.yaml
name: "rag_data_generator"
version: "1.0.0"
description: "Generate RAG training data from documents"

input_datasets:
  - name: "source_documents_pdf"
    type: "pdf"
    required: true
    description: "Source PDF documents for data generation"

output_datasets:
  - name: "augmented_dataset"
    type: "huggingface"
    description: "Generated question-answer pairs from documents"
    features:
      - name: "prompt"
        dtype: "string"
        description: "Generated question"
      - name: "response"
        dtype: "string"
        description: "Expected answer from document"
      - name: "context"
        dtype: "string"
        description: "Source document context"
```

### HuggingFace Feature Types

ASQI supports the full range of HuggingFace dataset feature types for declaring dataset schemas. Features are schema declarations that describe the structure of your data without specifying processing details.

#### Value Features (Scalars)

For scalar data types like strings, numbers, and booleans:

```yaml
output_datasets:
  - name: "text_dataset"
    type: "huggingface"
    features:
      - name: "id"
        dtype: "string"
      - name: "score"
        dtype: "float32"
      - name: "is_valid"
        dtype: "bool"
```

**Supported scalar dtypes:**
- **Numeric**: `int8`, `int16`, `int32`, `int64`, `uint8-64`, `float16`, `float32`, `float64`
- **String**: `string`, `large_string`, `binary`, `large_binary`
- **Boolean**: `bool`
- **Temporal**: `timestamp[s|ms|us|ns]`, `date32`, `date64`, `duration[s|ms|us|ns]`, `time32/64[s|ms|us|ns]`

#### List Features (Sequences)

For variable-length or fixed-length sequences:

```yaml
output_datasets:
  - name: "conversation_data"
    type: "huggingface"
    features:
      - name: "conversation_id"
        dtype: "string"
      - name: "turns"
        feature_type: "List"
        feature: "string"  # List of strings
        description: "List of conversation turns"
      - name: "timestamps"
        feature_type: "List"
        feature: "int64"
        length: 10  # Fixed-length list (optional, -1 for variable)
```

#### ClassLabel Features (Categorical)

For categorical data with named categories:

```yaml
output_datasets:
  - name: "sentiment_data"
    type: "huggingface"
    features:
      - name: "text"
        dtype: "string"
      - name: "sentiment"
        feature_type: "ClassLabel"
        names: ["positive", "negative", "neutral"]
        description: "Sentiment classification"
```

#### Image Features

For image datasets:

```yaml
output_datasets:
  - name: "image_dataset"
    type: "huggingface"
    features:
      - name: "image"
        feature_type: "Image"
        description: "Product image"
      - name: "caption"
        dtype: "string"
```

#### Audio Features

For audio datasets:

```yaml
output_datasets:
  - name: "audio_dataset"
    type: "huggingface"
    features:
      - name: "audio"
        feature_type: "Audio"
        description: "Audio recording"
      - name: "transcript"
        dtype: "string"
```

#### Video Features

For video datasets:

```yaml
output_datasets:
  - name: "video_dataset"
    type: "huggingface"
    features:
      - name: "video"
        feature_type: "Video"
        description: "Video clip"
      - name: "caption"
        dtype: "string"
```

**Note:** Image, Audio, and Video features are schema declarations only. The container is responsible for handling processing details like resampling, color mode conversion, resolution, framerate, and decoding.

#### Nested Structures (Dict Features)

For complex nested structures like question-answering datasets:

```yaml
output_datasets:
  - name: "qa_dataset"
    type: "huggingface"
    description: "Question answering dataset with nested answer structure"
    features:
      - name: "id"
        dtype: "string"
      - name: "question"
        dtype: "string"
      - name: "answers"
        feature_type: "Dict"
        description: "Answer annotations with multiple fields"
        fields:
          - name: "text"
            feature_type: "List"
            feature: "string"
            description: "Answer text strings"
          - name: "answer_start"
            feature_type: "List"
            feature: "int32"
            description: "Character positions where answers start"
```

This corresponds to the SQuAD dataset structure where `answers` is a dict containing lists.

#### Multimodal Datasets

Combine different feature types for multimodal datasets:

```yaml
output_datasets:
  - name: "multimodal_data"
    type: "huggingface"
    features:
      - name: "image"
        feature_type: "Image"
      - name: "audio"
        feature_type: "Audio"
      - name: "video"
        feature_type: "Video"
      - name: "caption"
        dtype: "string"
      - name: "tags"
        feature_type: "List"
        feature: "string"
      - name: "category"
        feature_type: "ClassLabel"
        names: ["tutorial", "demo", "advertisement"]
```

#### Optional Features (Sparse Data)

Features can be marked as optional for datasets where some columns may be absent or contain null values:

```yaml
output_datasets:
  - name: "product_catalog"
    type: "huggingface"
    features:
      - name: "product_id"
        dtype: "string"
        required: true  # Must be present in every row
      - name: "name"
        dtype: "string"
        required: true
      - name: "description"
        dtype: "string"
        required: false  # Optional - not all products have descriptions
      - name: "thumbnail"
        feature_type: "Image"
        required: false  # Optional - not all products have images
      - name: "price"
        dtype: "float32"
        required: true
```

The `required` field (defaults to `false`) indicates whether a feature must be present in every row of the dataset.

### Multiple Input Dataset Types

Test containers can declare support for multiple dataset formats, allowing users flexibility in providing data while ensuring the container can handle different input types.

**Single Type:**
```yaml
# Container accepts only PDF documents
input_datasets:
  - name: "knowledge_base"
    type: "pdf"
    required: true
    description: "Source documents for RAG testing"
```

**Multiple Types:**
```yaml
# Container accepts either PDF or TXT documents
input_datasets:
  - name: "knowledge_base"
    type: ["pdf", "txt"]
    required: true
    description: "Source documents - accepts PDF or plain text files"
```

**Multiple Types Including HuggingFace Datasets:**
```yaml
# Container accepts HuggingFace datasets, PDFs, or text files
input_datasets:
  - name: "knowledge_base"
    type: ["huggingface", "pdf", "txt"]
    required: true
    description: "Training data in any supported format"
    features:  # Required when huggingface is an accepted type
      - name: "text"
        dtype: "string"
        description: "Text content"
```


### Returning Generated Datasets

Containers return generated dataset information in the JSON output:

```python
# In container entrypoint.py
from asqi.response_schemas import ContainerOutput, GeneratedDataset

# Generate your dataset
# ...

# Return dataset information using response schemas for type safety
generated_dataset = GeneratedDataset(
    dataset_name="augmented_dataset",
    dataset_type="huggingface",
    dataset_path="/output/generated_rag_data",  # Container path
    format="parquet",
    metadata={
        "num_rows": 100,
        "num_columns": 3,
        "generation_params": {
            "chunk_size": 600,
            "questions_per_chunk": 2
        }
    }
)

output = ContainerOutput(
    results={
        "success": True,
        "rows_generated": 100
    },
    generated_datasets=[generated_dataset]
)

print(output.model_dump_json(indent=2))
```

### Path Translation

ASQI automatically translates container paths to host paths. The container writes to `/output/generated_rag_data`, and ASQI maps this to the actual host output directory specified in the test configuration.

## Data Generation Workflow

ASQI provides a dedicated workflow for synthetic data generation using data generation containers (SDG - Synthetic Data Generation).

### Data Generation vs. Testing

Data generation jobs differ from test jobs:

- **Purpose**: Generate new datasets rather than evaluate systems
- **Systems**: Systems are optional (used for generation, not testing)
- **Output Focus**: Primary output is generated datasets
- **Configuration**: Uses `GenerationConfig` instead of `SuiteConfig`

### Generation Configuration

Create a generation configuration file:

**Example: `generation_config.yaml`**

```yaml
# yaml-language-server: $schema=../../src/asqi/schemas/asqi_generation_config.schema.json

job_name: "RAG Data Generation"
generation_jobs:
  - id: "rag_qa_generation"
    name: "Generate RAG Q&A Pairs"
    image: "my-registry/rag-data-generator:latest"
    systems:
      generation_system: "openai_gpt4o_mini"    # LLM for question generation
    input_datasets:
      source_documents_pdf:
        file_path: "sample.pdf"
    volumes:
      input: "input/"
      output: "output/"
    params:
      output_dataset_path: "rag_qa_dataset"
      chunk_size: 600
      chunk_overlap: 50
      num_questions: 2
      persona_name: "Customer"
      persona_description: "Customer of the Company"
```

### The `generate-dataset` Command

Generate synthetic data using the `generate-dataset` CLI command:

```bash
asqi generate-dataset \
  --generation-config generation_config.yaml \
  --systems-config systems.yaml \
  --datasets-config datasets_registry.yaml \
  --output-file generation_results.json
```

**Command Options:**

- `--generation-config`, `-t`: Path to the generation configuration YAML file (required)
- `--systems-config`, `-s`: Path to systems YAML file (optional, only if generation uses systems)
- `--datasets-config`, `-d`: Path to datasets registry YAML file (optional)
- `--output-file`, `-o`: Path to save execution results JSON (default: `output.json`)
- `--concurrent-tests`, `-c`: Number of generation jobs to run concurrently (1-20, default: 5)
- `--max-failures`, `-m`: Maximum number of failures to display (1-10, default: 3)
- `--progress-interval`, `-p`: Progress update interval in seconds (1-10, default: 2)
- `--container-config`: Optional path to container configuration YAML

### Systems for Data Generation

Systems in data generation jobs serve as tools for generation (e.g., LLMs for text generation, question generation, evaluation):

**Example: `systems.yaml`**

```yaml
systems:
  openai_gpt4o_mini:
    type: "llm_api"
    params:
      base_url: "http://localhost:4000/v1"
      model: "openai/gpt-4o-mini"
      api_key: "sk-1234"
```

## Dataset Loading Utilities

### `load_hf_dataset()`

Load and validate HuggingFace datasets with automatic column mapping.

**Function Signature:**
```python
def load_hf_dataset(
    dataset_config: Union[dict, HFDatasetDefinition],
    input_mount_path: Path | None = None,
    expected_features: Sequence[DatasetFeature | HFFeature] | None = None,
    dataset_name: str = "dataset",
) -> Dataset
```

**Parameters:**
- `dataset_config`: Dataset configuration (passed by ASQI via `--input-datasets`)
- `input_mount_path`: Container input mount point (typically `/input`)
- `expected_features`: Feature definitions from manifest for validation (optional)
- `dataset_name`: Dataset name for error messages

**Returns:** Loaded HuggingFace `Dataset` with columns mapped and validated

**Basic Usage (without validation):**
```python
from pathlib import Path
from asqi.datasets import load_hf_dataset

# dataset_config passed by ASQI
dataset = load_hf_dataset(
    dataset_config,
    input_mount_path=Path("/input")
)
```

**With Validation (recommended):**
```python
import sys
from pathlib import Path
import yaml
from asqi.datasets import load_hf_dataset
from asqi.schemas import Manifest

# Load manifest
with open("/app/manifest.yaml") as f:
    manifest = Manifest(**yaml.safe_load(f))

# Get expected features from manifest
input_spec = manifest.input_datasets[0]

# Load and validate
try:
    dataset = load_hf_dataset(
        dataset_config,
        input_mount_path=Path("/input"),
        expected_features=input_spec.features,  # Validates schema
        dataset_name=input_spec.name
    )
    print(f"✓ Validated {len(dataset)} rows")
except ValueError as e:
    print(f"Validation failed: {e}", file=sys.stderr)
    sys.exit(1)
```

**What it does:**
1. Loads dataset using HuggingFace `load_dataset()` with `loader_params`
2. Applies column mapping (renames columns per `mapping` config)
3. Validates features if `expected_features` provided (checks required columns and types)
4. Returns validated dataset ready for use

**Error Example:**
```
Dataset 'source_data' validation failed:
Missing required features: text
Available columns: label, text2

Hint: Check your dataset mapping configuration and feature types.
```

For advanced validation scenarios beyond basic single-dataset usage:

**Validating Multiple Input Datasets:**

```python
import sys
from pathlib import Path
from asqi.datasets import load_hf_dataset

# Load and validate each input dataset
input_mount_path = Path("/input")
datasets = {}
for input_spec in manifest.input_datasets:
    dataset_name = input_spec.name
    dataset_config = input_datasets_config.get(dataset_name)

    if dataset_config is None:
        if input_spec.required:
            print(f"Error: Missing required dataset '{dataset_name}'", file=sys.stderr)
            sys.exit(1)
        continue  # Skip optional datasets

    datasets[dataset_name] = load_hf_dataset(
        dataset_config,
        input_mount_path=input_mount_path,
        expected_features=input_spec.features,
        dataset_name=dataset_name
    )

# Use validated datasets
source_data = datasets["source_data"]
reference_data = datasets.get("reference_data")  # Optional
```

**Validating Output Datasets:**

```python
from asqi.datasets import validate_dataset_features

# After generating synthetic data
output_dataset = Dataset.from_list(generated_rows)

# Validate against manifest output schema
output_spec = manifest.output_datasets[0]
validate_dataset_features(
    output_dataset,
    expected_features=output_spec.features,
    dataset_name=output_spec.name
)

### File Validation Functions

Validate dataset file types:

```python
from asqi.datasets import verify_pdf_file, verify_txt_file

# Validate PDF files
pdf_path = verify_pdf_file("document.pdf")  # Raises ValueError if not .pdf

# Validate text files
txt_path = verify_txt_file("corpus.txt")    # Raises ValueError if not .txt
```

## Complete Examples

### Example 1: RAG Data Generation

Generate question-answer pairs from PDF documents for RAG training.

**1. Create dataset registry (`datasets/source_datasets.yaml`):**

```yaml
datasets:
  company_docs:
    type: "pdf"
    description: "Company documentation for RAG"
    file_path: "company_handbook.pdf"
    tags: ["rag", "source"]
```

**2. Create generation config (`generation_jobs/rag_generation.yaml`):**

```yaml
job_name: "RAG Training Data Generation"
generation_jobs:
  - id: "generate_rag_qa"
    name: "Generate Q&A from Docs"
    image: "my-registry/rag-generator:latest"
    systems:
      generation_system: "gpt4o_mini"
    input_datasets:
      source_documents_pdf: "company_docs"
    volumes:
      input: "data/pdfs/"
      output: "data/generated/"
    params:
      chunk_size: 500
      questions_per_chunk: 3
      output_dataset_path: "rag_qa_pairs"
```

**3. Run generation:**

```bash
asqi generate-dataset \
  --generation-config generation_jobs/rag_generation.yaml \
  --systems-config config/systems.yaml \
  --datasets-config datasets/source_datasets.yaml \
  --output-file rag_generation_results.json
```

### Example 2: Evaluation with Input Datasets

Run evaluation tests using pre-defined evaluation datasets.

**1. Create dataset registry (`datasets/eval_datasets.yaml`):**

```yaml
datasets:
  benchmark_questions:
    type: "huggingface"
    description: "Standard benchmark questions"
    loader_params:
      builder_name: "json"
      data_files: "benchmark.json"
    mapping:
      input: "prompt"
      expected_output: "response"
    tags: ["evaluation", "benchmark"]
```

**2. Create test suite (`suites/eval_suite.yaml`):**

```yaml
suite_name: "Benchmark Evaluation"
test_suite:
  - id: "benchmark_test"
    name: "Standard Benchmark Test"
    image: "my-registry/evaluator:latest"
    systems_under_test: ["my_chatbot"]
    input_datasets:
      evaluation_data: "benchmark_questions"
    volumes:
      input: "data/benchmarks/"
      output: "results/"
```

**3. Run tests:**

```bash
asqi execute-tests \
  --test-suite-config suites/eval_suite.yaml \
  --systems-config config/systems.yaml \
  --datasets-config datasets/eval_datasets.yaml \
  --output-file benchmark_results.json
```

### Example 3: Dataset Augmentation Pipeline

Generate augmented versions of existing datasets.

**1. Input dataset (`datasets/base_data.yaml`):**

```yaml
datasets:
  base_training_data:
    type: "huggingface"
    description: "Base training examples"
    loader_params:
      builder_name: "parquet"
      data_dir: "training_data/"
    mapping: {}  # No mapping needed
    tags: ["training", "base"]
```

**2. Generation config (`generation_jobs/augment.yaml`):**

```yaml
job_name: "Dataset Augmentation"
generation_jobs:
  - id: "augment_training_data"
    name: "Augment Training Examples"
    image: "my-registry/data-augmenter:latest"
    systems:
      generation_system: "gpt4o_mini"
    input_datasets:
      base_data: "base_training_data"
    volumes:
      input: "data/base/"
      output: "data/augmented/"
    params:
      augmentation_factor: 3
      variation_types: ["paraphrase", "style_transfer"]
```

**3. Run augmentation:**

```bash
asqi generate-dataset \
  --generation-config generation_jobs/augment.yaml \
  --systems-config config/systems.yaml \
  --datasets-config datasets/base_data.yaml \
  --output-file augmentation_results.json
```

## Related Documentation

- [Configuration](configuration.md) - Detailed schema documentation for all configuration types
- [Custom Test Containers](custom-test-containers.md) - Guide to creating containers with dataset support
- [Examples](examples.md) - More practical examples and workflows
- [CLI Reference](cli.rst) - Complete CLI command documentation
