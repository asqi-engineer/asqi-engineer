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

### Returning Generated Datasets

Containers return generated dataset information in the JSON output:

```python
# In container entrypoint.py
import json

# Generate your dataset
# ...

# Return dataset information in output JSON
output = {
    "results": {  # Use 'results' (recommended) or 'test_results' (legacy)
        "success": True,
        "rows_generated": 100
    },
    "generated_datasets": [
        {
            "dataset_name": "augmented_dataset",
            "dataset_type": "huggingface",
            "dataset_path": "/output/generated_rag_data",  # Container path
            "format": "parquet",
            "metadata": {
                "num_rows": 100,
                "num_columns": 3,
                "generation_params": {
                    "chunk_size": 600,
                    "questions_per_chunk": 2
                }
            }
        }
    ]
}

print(json.dumps(output))
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
      simulator_system: "openai_embedding"      # Embedding model for chunking
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

Systems in data generation jobs serve as tools for generation (e.g., LLMs for text generation, embedding models for processing):

**Example: `systems.yaml`**

```yaml
systems:
  openai_gpt4o_mini:
    type: "llm_api"
    params:
      base_url: "http://localhost:4000/v1"
      model: "openai/gpt-4o-mini"
      api_key: "sk-1234"
  
  openai_embedding:
    type: "llm_api"
    params:
      base_url: "http://localhost:4000/v1"
      model: "openai/text-embedding-3-small"
      api_key: "sk-1234"
```

## Dataset Loading Utilities

ASQI provides utility functions for loading datasets within containers:

### `load_hf_dataset()`

Load a HuggingFace dataset with automatic column mapping:

```python
from asqi.datasets import load_hf_dataset

# Load dataset with configuration
dataset_config = {
    "type": "huggingface",
    "loader_params": {
        "builder_name": "json",
        "data_files": "data.json"
    },
    "mapping": {
        "question": "prompt",
        "answer": "response"
    }
}

dataset = load_hf_dataset(dataset_config)

# Access mapped columns
for row in dataset:
    print(row["prompt"])    # Originally 'question' in dataset
    print(row["response"])  # Originally 'answer' in dataset
```

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
      embedding_system: "text_embedding"
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

## Best Practices

### Dataset Registry Organization

- **Use descriptive names**: `eval_questions_v1`, `company_handbook_2024`
- **Add tags**: Categorize datasets with tags for easy filtering
- **Include descriptions**: Document dataset purpose and contents
- **Version datasets**: Keep track of dataset versions in names or tags

### Volume Management

- **Separate input/output**: Use distinct directories for inputs and outputs
- **Use relative paths**: Prefer relative paths for portability
- **Organize by type**: Group datasets by type or purpose in subdirectories

### Column Mapping

- **Document mappings**: Add comments explaining column transformations
- **Validate schemas**: Ensure mapped columns match container expectations
- **Handle missing columns**: Containers should validate required columns exist

### Data Generation

- **Test locally first**: Validate generation logic with small samples
- **Monitor resources**: Large-scale generation can be resource-intensive
- **Version outputs**: Include version/timestamp in output dataset names
- **Validate outputs**: Verify generated datasets meet quality standards

## Schema References

Dataset-related JSON schemas for IDE integration:

```yaml
# For datasets configuration files
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_datasets_config.schema.json

# For generation configuration files  
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_generation_config.schema.json

# For test suite files (with datasets support)
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_suite_config.schema.json

# For manifest files (with input/output datasets)
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_manifest.schema.json
```

## Related Documentation

- [Configuration](configuration.md) - Detailed schema documentation for all configuration types
- [Custom Test Containers](custom-test-containers.md) - Guide to creating containers with dataset support
- [Examples](examples.md) - More practical examples and workflows
- [CLI Reference](cli.rst) - Complete CLI command documentation
