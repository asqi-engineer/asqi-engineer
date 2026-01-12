# Multi-Format Dataset Support Example

This document demonstrates the new multi-format dataset support feature that allows manifests to accept datasets in multiple formats (either/or relationship).

## Example Use Cases

### 1. RAG Container - Accept PDF or TXT Documents

```yaml
# test_containers/rag_evaluator/manifest.yaml
name: "rag_evaluator"
version: "1.0"
description: "RAG evaluation container that can work with various document formats"

input_datasets:
  - name: "knowledge_base"
    type: ["pdf", "txt"]  # Accept EITHER PDF or TXT
    required: true
    description: "Source documents for RAG - can be PDF files or plain text files"
```

**User provides PDF dataset:**
```yaml
# config/datasets/datasets.yaml
datasets:
  my_docs:
    type: "pdf"
    file_path: "documents.pdf"
    description: "Company documentation in PDF format"
```

```yaml
# config/suites/rag_test.yaml
test_suite:
  - name: "RAG Evaluation"
    image: "rag_evaluator:latest"
    input_datasets:
      knowledge_base: my_docs  # PDF dataset - validated as acceptable
```

**Or user provides TXT dataset:**
```yaml
# config/datasets/datasets.yaml
datasets:
  my_text_docs:
    type: "txt"
    file_path: "documents.txt"
    description: "Company documentation in text format"
```

```yaml
# config/suites/rag_test.yaml
test_suite:
  - name: "RAG Evaluation"
    image: "rag_evaluator:latest"
    input_datasets:
      knowledge_base: my_text_docs  # TXT dataset - also validated as acceptable
```

### 2. Data Analysis Container - Accept Any Format

```yaml
# test_containers/data_analyzer/manifest.yaml
name: "data_analyzer"
version: "1.0"
description: "Flexible data analysis container"

input_datasets:
  - name: "analysis_data"
    type: ["huggingface", "pdf", "txt"]  # Accept ANY of these formats
    required: true
    description: "Input data for analysis - structured dataset or documents"
    features:  # Optional when multiple types are accepted
      - name: "text"
        dtype: "string"
```

Users can provide:
- A HuggingFace dataset with structured data
- A PDF document
- A plain text file

The container will detect which type was provided and process accordingly.

### 3. Backward Compatibility - Single Type Still Works

```yaml
# Existing manifests continue to work exactly as before
input_datasets:
  - name: "evaluation_data"
    type: "huggingface"  # Single type (existing behavior)
    required: true
    features:  # Features still required for single HuggingFace type
      - name: "prompt"
        dtype: "string"
      - name: "response"
        dtype: "string"
```

## Validation Behavior

### Error Messages

**Missing required dataset:**
```
Job 'my_job': Missing required dataset 'knowledge_base'
(accepted types: pdf or txt, description: Source documents for RAG)
```

**Wrong type provided:**
```
Job 'my_job': Dataset 'knowledge_base' has type 'huggingface'
but container accepts: [pdf, txt]
```

### Features Requirement

- **Single HuggingFace type**: Features REQUIRED (backward compatible)
  ```yaml
  type: "huggingface"
  features:  # Must be specified
    - name: "text"
      dtype: "string"
  ```

- **Multiple types including HuggingFace**: Features OPTIONAL
  ```yaml
  type: ["huggingface", "pdf", "txt"]
  features:  # Optional - recommended but not required
    - name: "text"
      dtype: "string"
  ```

- **Non-HuggingFace types**: Features always optional
  ```yaml
  type: ["pdf", "txt"]
  features: null  # Not used
  ```

## Implementation Details

### Schema Changes

```python
class InputDataset(BaseModel):
    type: Union[DatasetType, List[DatasetType]]  # New: supports list of types
```

### Validation Logic

```python
def validate_dataset_configs(item, manifest):
    for schema_dataset in manifest.input_datasets:
        expected_types = normalize_dataset_types(schema_dataset.type)

        if dataset_def.type not in expected_types:
            # Error: type mismatch
```

### No Field Mixing

The multi-type support is **manifest-only**. User-provided dataset configs remain separate:
- Manifest declares accepted types (InputDataset)
- User provides dataset reference (string)
- Dataset registry contains actual definitions (HFDatasetDefinition | PDFDatasetDefinition | TXTDatasetDefinition)

Each dataset type keeps its own fields - no mixing!
