# Creating Custom Test Containers

This guide walks you through creating your own test containers for ASQI Engineer, enabling you to implement domain-specific testing frameworks and evaluation logic.

## Container Structure Requirements

Every test container must follow this standardized structure:

```
test_containers/my_custom_tester/
├── Dockerfile                    # Container build instructions
├── entrypoint.py                # Main test execution script
├── manifest.yaml               # Container capabilities declaration
└── requirements.txt            # Python dependencies (optional)
```

## Step-by-Step Development Guide

### 1. Create Container Directory

```bash
mkdir test_containers/my_custom_tester
cd test_containers/my_custom_tester
```

### 2. Define Container Capabilities

Create `manifest.yaml` to declare what your container can do:

```yaml
name: "my_custom_tester"
version: "1.0.0"
description: "Custom testing framework for specific AI system evaluation"

input_systems:
  # Single system type
  - name: "system_under_test"
    type: "llm_api"
    required: true
    description: "The primary system being tested"

  # Multiple system types (supports both LLM and VLM)
  # Uncomment to allow container to accept multiple system types:
  # - name: "system_under_test"
  #   type: ["llm_api", "vlm_api"]
  #   required: true
  #   description: "System that accepts both text and vision models"

  - name: "evaluator_system"
    type: "llm_api"
    required: false
    description: "Optional system for result evaluation"

input_schema:
  - name: "test_iterations"
    type: "integer"
    required: false
    description: "Number of test iterations to run"
  - name: "evaluation_criteria"
    type: "list"
    required: true
    description: "List of evaluation criteria to assess"
  - name: "difficulty_level"
    type: "string"
    required: false
    description: "Test difficulty: easy, medium, hard"

output_metrics:
  - name: "success"
    type: "boolean"
    description: "Whether the test completed successfully"
  - name: "score"
    type: "float"
    description: "Overall test score (0.0 to 1.0)"
  - name: "test_count"
    type: "integer"
    description: "Number of individual tests executed"
  - name: "detailed_results"
    type: "object"
    description: "Comprehensive test results and analysis"
```

### 3. Implement Test Logic

Create `entrypoint.py` with standardized argument handling:

```python
#!/usr/bin/env python3
import argparse
import json
import sys
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_llm_client(system_params: Dict[str, Any]):
    """Create LLM client from system parameters."""
    try:
        import openai
        return openai.OpenAI(
            base_url=system_params.get("base_url"),
            api_key=system_params.get("api_key")
        )
    except ImportError:
        logger.error("OpenAI package not available")
        raise

def run_custom_tests(sut_client, test_params: Dict[str, Any], evaluator_client=None) -> Dict[str, Any]:
    """
    Implement your custom test logic here.
    
    Args:
        sut_client: OpenAI client for system under test
        test_params: Test parameters from YAML configuration
        evaluator_client: Optional OpenAI client for evaluation
        
    Returns:
        Dictionary with test results
    """
    test_iterations = test_params.get("test_iterations", 10)
    evaluation_criteria = test_params.get("evaluation_criteria", [])
    difficulty_level = test_params.get("difficulty_level", "medium")
    
    results = {
        "success": True,
        "score": 0.0,
        "test_count": 0,
        "detailed_results": {
            "individual_tests": [],
            "criteria_scores": {},
            "difficulty_used": difficulty_level
        }
    }
    
    try:
        # Example test logic
        passed_tests = 0
        
        for i in range(test_iterations):
            # Generate test prompt based on difficulty and criteria
            test_prompt = generate_test_prompt(i, difficulty_level, evaluation_criteria)
            
            # Get response from system under test
            response = sut_client.chat.completions.create(
                model=sut_client.model,  # This should be set from system_params
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=512
            )
            
            # Evaluate response
            if evaluator_client:
                evaluation = evaluate_with_llm(response.choices[0].message.content, 
                                             test_prompt, evaluator_client)
            else:
                evaluation = evaluate_response(response.choices[0].message.content, 
                                             evaluation_criteria)
            
            # Record individual test result
            test_result = {
                "test_id": i,
                "prompt": test_prompt,
                "response": response.choices[0].message.content,
                "evaluation": evaluation,
                "passed": evaluation["score"] >= 0.7
            }
            
            results["detailed_results"]["individual_tests"].append(test_result)
            
            if test_result["passed"]:
                passed_tests += 1
                
        # Calculate final metrics
        results["test_count"] = test_iterations
        results["score"] = passed_tests / test_iterations if test_iterations > 0 else 0.0
        
        # Calculate scores by criteria
        for criterion in evaluation_criteria:
            criterion_scores = [t["evaluation"].get(criterion, 0.0) 
                              for t in results["detailed_results"]["individual_tests"]]
            results["detailed_results"]["criteria_scores"][criterion] = {
                "average": sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0.0,
                "min": min(criterion_scores) if criterion_scores else 0.0,
                "max": max(criterion_scores) if criterion_scores else 0.0
            }
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        results["success"] = False
        results["error"] = str(e)
        
    return results

def generate_test_prompt(test_id: int, difficulty: str, criteria: List[str]) -> str:
    """Generate test prompts based on difficulty and criteria."""
    base_prompts = {
        "easy": f"Simple test question {test_id}: What is 2+2?",
        "medium": f"Medium test {test_id}: Explain the concept of machine learning in simple terms.",
        "hard": f"Complex test {test_id}: Analyze the ethical implications of AI decision-making in healthcare."
    }
    
    prompt = base_prompts.get(difficulty, base_prompts["medium"])
    
    if criteria:
        prompt += f" Please address these aspects: {', '.join(criteria)}"
        
    return prompt

def evaluate_response(response: str, criteria: List[str]) -> Dict[str, Any]:
    """Implement your custom evaluation logic."""
    # Simple rule-based evaluation example
    evaluation = {
        "score": 0.8,  # Base score
        "length_appropriate": len(response.split()) > 10,
        "contains_keywords": any(criterion.lower() in response.lower() for criterion in criteria)
    }
    
    # Adjust score based on evaluation
    if not evaluation["length_appropriate"]:
        evaluation["score"] -= 0.2
    if not evaluation["contains_keywords"] and criteria:
        evaluation["score"] -= 0.3
        
    evaluation["score"] = max(0.0, min(1.0, evaluation["score"]))
    return evaluation

def evaluate_with_llm(response: str, prompt: str, evaluator_client) -> Dict[str, Any]:
    """Use LLM for evaluation (if evaluator system provided)."""
    evaluation_prompt = f"""
    Evaluate this response on a scale of 0.0 to 1.0:
    
    Original Prompt: {prompt}
    Response: {response}
    
    Consider accuracy, helpfulness, and appropriateness.
    Return only a JSON object with 'score' and 'reasoning' fields.
    """
    
    try:
        eval_response = evaluator_client.chat.completions.create(
            model=evaluator_client.model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=256
        )
        
        # Parse JSON response
        eval_result = json.loads(eval_response.choices[0].message.content)
        return {
            "score": eval_result.get("score", 0.5),
            "reasoning": eval_result.get("reasoning", "No reasoning provided"),
            "evaluated_by_llm": True
        }
    except Exception as e:
        logger.warning(f"LLM evaluation failed: {e}")
        return {"score": 0.5, "error": str(e), "evaluated_by_llm": False}

def main():
    """Main entrypoint following ASQI container interface."""
    parser = argparse.ArgumentParser(description="Custom ASQI test container")
    parser.add_argument("--systems-params", required=True, 
                       help="JSON string with system configurations")
    parser.add_argument("--test-params", required=True,
                       help="JSON string with test parameters")
    args = parser.parse_args()
    
    try:
        # Parse input parameters
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        
        # Extract systems
        sut_params = systems_params.get("system_under_test", {})
        evaluator_params = systems_params.get("evaluator_system")
        
        # Create clients
        sut_client = create_llm_client(sut_params)
        sut_client.model = sut_params.get("model")  # Store model for use in requests
        
        evaluator_client = None
        if evaluator_params:
            evaluator_client = create_llm_client(evaluator_params)
            evaluator_client.model = evaluator_params.get("model")
        
        # Run tests
        results = run_custom_tests(sut_client, test_params, evaluator_client)
        
        # Output results as JSON to stdout
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        # Always output JSON, even on error
        error_result = {
            "success": False,
            "error": str(e),
            "score": 0.0,
            "test_count": 0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 4. Create Container Dependencies

Create `requirements.txt` with your dependencies:

```txt
openai>=1.0.0
pydantic>=2.0.0
requests>=2.28.0
# Add other dependencies your test logic needs
```

### 5. Build Container

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY entrypoint.py .
COPY manifest.yaml .

# Make entrypoint executable
RUN chmod +x entrypoint.py

# Set entrypoint
ENTRYPOINT ["python", "entrypoint.py"]
```

### 6. Build and Test

```bash
# Build container
docker build -t my-registry/my_custom_tester:latest .

# Test manually
docker run --rm my-registry/my_custom_tester:latest \
  --systems-params '{
    "system_under_test": {
      "type": "llm_api",
      "base_url": "http://localhost:4000/v1",
      "model": "gpt-4o-mini",
      "api_key": "sk-1234"
    }
  }' \
  --test-params '{
    "test_iterations": 3,
    "evaluation_criteria": ["accuracy", "helpfulness"],
    "difficulty_level": "medium"
  }'
```

## Adding Technical Reports in Custom Test Containers

This section describes how to generate technical reports in your custom ASQI test containers. Follow these guidelines to ensure your container outputs are compatible with the ASQI workflow and score card evaluation system.

### 1. Output Structure

Your container **MUST** output JSON to stdout. The output should include the following fields:

- `results` (or `test_results` for legacy compatibility): (dict) The main results of the test (output_metrics).
- `generated_reports`: (list, optional) List that describes report files generated by the test.
- `generated_datasets`: (list, optional) List that describes datasets generated by the test.

**Note:** Use `results` for new containers. The `test_results` field name is supported for backward compatibility but `results` is the recommended field name.

Example **output**:

```python
output = {
  "results": {  # Use 'results' (recommended) or 'test_results' (legacy)
    "success": True,
    "metrics": { ... }
  },
  "generated_reports": [
    {
      "report_name": "detailed_metrics",
      "report_type": "html",
      "report_path": write_detailed_metrics_report(...)
    },
    {
      "report_name": "performance_summary",
      "report_type": "html",
      "report_path": write_performance_summary_report(...)
    }
  ]
}
print(json.dumps(output))
```

#### Required Fields for Each Report

Each report generated in `generated_reports` **MUST** return:

- `report_name`: (string) The name of the report, matching the manifest.
- `report_type`: (string) The type of the report (`html` or `pdf`).
- `report_path`: (string) The container path to the report file.

### 2. Report Generation

You are free to use any framework or library you prefer to generate your reports. We don't enforce a specific one, giving you the flexibility to choose what fits your needs.

The only requirement is that the final files (HTML or PDF) must be persisted to the container volume.

### 3. Declaring Reports in the Manifest

For every report your container generates, you must add an entry to the `output_reports` list in your manifest. This tells the system exactly what files to expect. There must be a one-to-one match between the reports your container produces and those declared in the manifest

- `name`: (string) Unique identifier (must match the name in the `entrypoint`).
- `type`: (string) The type of the report (`html` or `pdf`).
- `description`: (string) A brief summary of the report's content.

Example **manifest**:

```yaml
output_reports:
  - name: "quick_summary"
    type: "html"
    description: "A quick HTML summary report of the mock test execution"
```

### 4. Manifest Validation

When running a test in `ASQI Engineer` and once the container has finished executing, the system automatically validates the generated reports against the manifest.

- The name and type of every report returned by your container match the `output_reports` section of the manifest.
- If the container output does not match the manifest by returning extra, missing or mislabeled reports, the entire test will **fail**.

### 5. Backward Compatibility

- If your container does not output `generated_reports` and `results`/`test_results`, ASQI will treat it as legacy output and skip report validation.
- For new containers, it is recommended to include the `results` field (preferred) or `test_results` field (legacy), along with `generated_reports` if generating reports.

### 6. Reference Implementation 

For a complete reference implementation, check `test_containers/mock_tester`. It provides a working example with the correct output structure and an example report (`quick_summary.html`).

  #### 1. Build the `mock_tester` image
  ```bash
  docker build -t asqiengineer/test-container:mock_tester-latest ./test_containers/mock_tester/.
  ```
  #### 2. Execute and get the generated report
  ```bash
  asqi execute -s config/systems/demo_systems.yaml -t config/suites/demo_test.yaml -r config/score_cards/example_score_card.yaml -o output.json
  ```

## Working with Input Datasets

Test containers can receive and process input datasets for evaluation, testing, or data generation.

### 1. Declaring Input Datasets in Manifest

Define required datasets in your container's `manifest.yaml`:

```yaml
# manifest.yaml
name: "dataset_evaluator"
version: "1.0.0"
description: "Evaluates systems using benchmark datasets"

input_datasets:
  # HuggingFace dataset with required features
  - name: "evaluation_data"
    type: "huggingface"
    required: true
    description: "Benchmark evaluation dataset"
    features:
      - name: "prompt"
        dtype: "string"
        description: "Input prompt"
      - name: "response"
        dtype: "string"
        description: "Expected response"
  
  # PDF document input
  - name: "source_documents_pdf"
    type: "pdf"
    required: false
    description: "Optional source documents"
```

**Dataset Types:**
- `huggingface`: Structured datasets (requires `features` definition)
- `pdf`: PDF document files
- `txt`: Plain text files

### 2. Receiving Dataset Paths in Container

ASQI passes dataset information via the `--test-params` argument:

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=True)
    parser.add_argument("--test-params", required=True)
    args = parser.parse_args()
    
    test_params = json.loads(args.test_params)
    
    # Access dataset paths
    datasets = test_params.get("datasets", {})
    eval_data_info = datasets.get("evaluation_data")
    pdf_info = datasets.get("source_documents_pdf")
    
    # Dataset info structure:
    # {
    #   "type": "huggingface",
    #   "path": "/input/dataset_path",
    #   "config": {...}  # Original dataset config
    # }
```

### 3. Loading and Processing Datasets

**HuggingFace Datasets:**

```python
from datasets import load_from_disk
import json

# Get dataset path from test params
test_params = json.loads(args.test_params)
dataset_info = test_params["datasets"]["evaluation_data"]

# Load the dataset
dataset = load_from_disk(dataset_info["path"])

# Process dataset
for row in dataset:
    prompt = row["prompt"]
    expected_response = row["response"]
    # Run evaluation...
```

**PDF Files:**

```python
from pathlib import Path

# Get PDF path from test params
pdf_info = test_params["datasets"]["source_documents_pdf"]
pdf_path = Path(pdf_info["path"])

# Process PDF
with open(pdf_path, "rb") as f:
    # Use PDF processing library
    pass
```

**Text Files:**

```python
# Get text file path
txt_info = test_params["datasets"]["corpus_txt"]
txt_path = Path(txt_info["path"])

# Read text file
with open(txt_path, "r") as f:
    content = f.read()
```

### 4. Column Mapping Awareness

When users configure column mapping in the dataset registry, ASQI handles the mapping before passing data to your container. Your container always receives data with column names matching the `features` defined in your manifest.

**Example:**
- Your manifest expects features: `prompt`, `response`
- User's dataset has columns: `question`, `answer`
- User configures mapping: `question -> prompt`, `answer -> response`
- Your container receives dataset with columns already mapped to `prompt`, `response`

### 5. Complete Example: Dataset Evaluation Container

```python
#!/usr/bin/env python3
import argparse
import json
import sys
from datasets import load_from_disk
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=True)
    parser.add_argument("--test-params", required=True)
    args = parser.parse_args()
    
    systems_params = json.loads(args.systems_params)
    test_params = json.loads(args.test_params)
    
    # Get system under test
    sut = systems_params["system_under_test"]
    client = OpenAI(
        base_url=sut["base_url"],
        api_key=sut["api_key"]
    )
    
    # Load evaluation dataset
    dataset_info = test_params["datasets"]["evaluation_data"]
    dataset = load_from_disk(dataset_info["path"])
    
    # Run evaluation
    correct = 0
    total = len(dataset)
    
    for row in dataset:
        response = client.chat.completions.create(
            model=sut["model"],
            messages=[{"role": "user", "content": row["prompt"]}]
        )
        
        actual = response.choices[0].message.content
        if actual.strip() == row["response"].strip():
            correct += 1
    
    # Output results
    results = {
        "results": {  # Use 'results' (recommended) or 'test_results' (legacy)
            "success": True,
            "accuracy": correct / total,
            "correct_count": correct,
            "total_count": total
        }
    }
    
    print(json.dumps(results))

if __name__ == "__main__":
    main()
```

## Generating Output Datasets

Containers can generate datasets as outputs, useful for synthetic data generation, augmentation, or preprocessing.

### 1. Declaring Output Datasets in Manifest

```yaml
# manifest.yaml
name: "rag_data_generator"
version: "1.0.0"
description: "Generate RAG training data from documents"

input_datasets:
  - name: "source_documents_pdf"
    type: "pdf"
    required: true
    description: "Source PDF documents"

output_datasets:
  - name: "generated_qa_pairs"
    type: "huggingface"
    description: "Generated question-answer pairs"
    features:
      - name: "prompt"
        dtype: "string"
        description: "Generated question"
      - name: "response"
        dtype: "string"
        description: "Answer extracted from document"
      - name: "context"
        dtype: "string"
        description: "Source document context"
```

### 2. Generating Datasets in Container

```python
#!/usr/bin/env python3
import json
from datasets import Dataset
from pathlib import Path

def generate_qa_pairs(pdf_path, llm_client, params):
    """Generate Q&A pairs from PDF document."""
    # Your generation logic here
    qa_pairs = []
    
    # Example generation
    for chunk in process_pdf(pdf_path, params["chunk_size"]):
        questions = generate_questions(chunk, llm_client, params["num_questions"])
        for question in questions:
            qa_pairs.append({
                "prompt": question,
                "response": extract_answer(chunk, question),
                "context": chunk
            })
    
    return qa_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=True)
    parser.add_argument("--test-params", required=True)
    args = parser.parse_args()
    
    systems_params = json.loads(args.systems_params)
    test_params = json.loads(args.test_params)
    
    # Get input PDF
    pdf_info = test_params["datasets"]["source_documents_pdf"]
    pdf_path = Path(pdf_info["path"])
    
    # Generate Q&A pairs
    llm_client = create_llm_client(systems_params["generation_system"])
    qa_pairs = generate_qa_pairs(pdf_path, llm_client, test_params)
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(qa_pairs)
    
    # Save dataset to output directory
    output_path = Path("/output") / test_params["output_dataset_path"]
    dataset.save_to_disk(str(output_path))
    
    # Return dataset information
    results = {
        "results": {  # Use 'results' (recommended) or 'test_results' (legacy)
            "success": True,
            "rows_generated": len(qa_pairs)
        },
        "generated_datasets": [
            {
                "dataset_name": "generated_qa_pairs",
                "dataset_type": "huggingface",
                "dataset_path": str(output_path),  # Container path
                "format": "arrow",
                "metadata": {
                    "num_rows": len(qa_pairs),
                    "num_columns": 3,
                    "generation_params": {
                        "chunk_size": test_params["chunk_size"],
                        "questions_per_chunk": test_params["num_questions"]
                    }
                }
            }
        ]
    }
    
    print(json.dumps(results))
```

### 3. Output Dataset Structure

Each generated dataset entry in the JSON output must include:

```python
{
    "dataset_name": "generated_qa_pairs",    # Matches manifest declaration
    "dataset_type": "huggingface",           # Type: huggingface, pdf, txt
    "dataset_path": "/output/dataset_path",  # Container path (ASQI translates to host)
    "format": "arrow",                       # File format (arrow, parquet, json, etc.)
    "metadata": {                            # Optional metadata
        "num_rows": 1000,
        "num_columns": 3,
        # Any other relevant metadata
    }
}
```

### 4. Path Translation

ASQI automatically translates container output paths to host paths:
- Container writes to: `/output/generated_data`
- ASQI maps to: Actual host output directory specified in test configuration
- Users can access generated datasets in the configured output directory

### 5. Validation

ASQI validates that:
- Returned `dataset_name` matches manifest `output_datasets` declaration
- Dataset type is consistent with manifest
- Dataset files exist at the specified path
- Required fields are present in the output JSON

## Data Generation Container Guidelines

Data generation containers differ from test containers in their purpose and configuration.

### Key Differences

| Aspect | Test Containers | Data Generation Containers |
|--------|----------------|---------------------------|
| **Purpose** | Evaluate systems | Generate synthetic data |
| **Systems** | Systems under test (required) | Generation tools (optional) |
| **Primary Output** | Test metrics | Generated datasets |
| **Configuration** | `SuiteConfig` | `GenerationConfig` |
| **CLI Command** | `execute-tests` | `generate-dataset` |

### Data Generation Manifest Example

```yaml
name: "synthetic_data_generator"
version: "1.0.0"
description: "Generate synthetic training data"

# Systems are optional for data generation
input_systems:
  - name: "generation_system"
    type: "llm_api"
    required: false  # Optional - pure data transformation might not need LLM
    description: "LLM for content generation"

input_datasets:
  - name: "seed_data"
    type: "huggingface"
    required: true
    features:
      - name: "prompt"
        dtype: "string"

output_datasets:
  - name: "augmented_data"
    type: "huggingface"
    description: "Augmented training dataset"
    features:
      - name: "prompt"
        dtype: "string"
      - name: "response"
        dtype: "string"
      - name: "variation_type"
        dtype: "string"

input_schema:
  - name: "augmentation_factor"
    type: "integer"
    required: true
    description: "Number of variations to generate per example"
  - name: "variation_types"
    type: "list"
    required: false
    description: "Types of variations (paraphrase, style_transfer, etc.)"

output_metrics:
  - name: "success"
    type: "boolean"
  - name: "examples_generated"
    type: "integer"
```

### Volume Mounting Requirements

Data generation containers require volume mounts for input and output:

```yaml
generation_jobs:
  - id: "generate_synthetic_data"
    image: "my-registry/data-generator:latest"
    volumes:
      input: "data/inputs/"   # Directory with input datasets
      output: "data/outputs/" # Directory for generated datasets
```

### Best Practices for Data Generation Containers

1. **Validate Inputs**: Check that required datasets and parameters are provided
2. **Progress Logging**: Log generation progress (but keep JSON output clean)
3. **Error Handling**: Handle failures gracefully and report errors clearly
4. **Metadata**: Include useful metadata about generation process
5. **Quality Checks**: Validate generated data quality before outputting
6. **Resource Management**: Handle large datasets efficiently (streaming, batching)
7. **Reproducibility**: Support random seeds and deterministic generation

### Example: Minimal Data Generation Container

```python
#!/usr/bin/env python3
import argparse
import json
from datasets import Dataset
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=False)  # Optional for generation
    parser.add_argument("--test-params", required=True)
    args = parser.parse_args()
    
    test_params = json.loads(args.test_params)
    
    # Load seed data
    seed_info = test_params["datasets"]["seed_data"]
    seed_dataset = load_from_disk(seed_info["path"])
    
    # Generate augmented examples
    augmented_examples = []
    augmentation_factor = test_params["augmentation_factor"]
    
    for row in seed_dataset:
        for i in range(augmentation_factor):
            augmented_examples.append({
                "prompt": augment_text(row["prompt"], i),
                "response": row.get("response", ""),
                "variation_type": f"augmentation_{i}"
            })
    
    # Save dataset
    output_dataset = Dataset.from_list(augmented_examples)
    output_path = Path("/output/augmented_data")
    output_dataset.save_to_disk(str(output_path))
    
    # Return results
    results = {
        "results": {  # Use 'results' (recommended) or 'test_results' (legacy)
            "success": True,
            "examples_generated": len(augmented_examples)
        },
        "generated_datasets": [
            {
                "dataset_name": "augmented_data",
                "dataset_type": "huggingface",
                "dataset_path": str(output_path),
                "format": "arrow",
                "metadata": {
                    "num_rows": len(augmented_examples),
                    "augmentation_factor": augmentation_factor,
                    "seed_size": len(seed_dataset)
                }
            }
        ]
    }
    
    print(json.dumps(results))

if __name__ == "__main__":
    main()
```

## Related Documentation

- [Dataset Support](datasets.md) - Complete dataset documentation
- [Configuration](configuration.md) - Dataset and generation configuration schemas