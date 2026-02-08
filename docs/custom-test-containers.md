# Creating Custom Test Containers

This guide shows you how to create custom test containers for ASQI, enabling domain-specific testing frameworks and evaluation logic.

## Quick Start

Every test container needs three files:

```
test_containers/my_tester/
├── Dockerfile          # Container build
├── entrypoint.py      # Test execution script
└── manifest.yaml      # Capabilities declaration
```

**Minimal Example:**

```yaml
# manifest.yaml
name: "my_tester"
version: "1.0.0"

input_systems:
  - name: "system_under_test"
    type: "llm_api"
    required: true

input_schema:
  - name: "num_tests"
    type: "integer"
    default: 10

output_metrics:
  - name: "success"
    type: "boolean"
  - name: "score"
    type: "float"
```

```python
# entrypoint.py
import argparse, json, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=True)
    parser.add_argument("--test-params", required=True)
    args = parser.parse_args()

    systems = json.loads(args.systems_params)
    params = json.loads(args.test_params)

    # Your test logic here
    results = run_tests(systems["system_under_test"], params)

    print(json.dumps(results))

if __name__ == "__main__":
    main()
```

See `test_containers/mock_tester/` for a complete working example.

---

## Container Structure

### 1. Directory Setup

```bash
mkdir -p test_containers/my_tester
cd test_containers/my_tester
```

### 2. Manifest Declaration

Create `manifest.yaml` to define container capabilities:

```yaml
name: "my_tester"
version: "1.0.0"
description: "Brief description of what this container tests"

# Systems the container can test
input_systems:
  - name: "system_under_test"
    type: "llm_api"              # or ["llm_api", "vlm_api"] for multiple
    required: true

  - name: "evaluator_system"     # Optional additional systems
    type: "llm_api"
    required: false

# Test configuration parameters
input_schema:
  - name: "test_iterations"
    type: "integer"
    default: 10
    description: "Number of test iterations"

# Results the container returns
output_metrics:
  - name: "success"
    type: "boolean"
    description: "Test completion status"
  - name: "score"
    type: "float"
    description: "Overall score (0.0 to 1.0)"
```

#### Parameter Types

ASQI supports rich parameter types for sophisticated test configurations:

**Basic Types:** `string`, `integer`, `float`, `boolean`

**Enum** - Restrict to specific values:
```yaml
- name: "mode"
  type: "enum"
  choices: ["fast", "thorough"]
  default: "fast"
```

**Typed Lists** - Specify element types:
```yaml
- name: "tags"
  type: "list"
  items: "string"
  default: ["test", "security"]
```

**Nested Objects** - Complex configurations:
```yaml
- name: "api_config"
  type: "object"
  properties:
    - name: "retries"
      type: "integer"
      default: 3
    - name: "timeout"
      type: "integer"
      default: 30
```

**List of Objects** - Advanced scenarios:
```yaml
- name: "test_scenarios"
  type: "list"
  items:
    name: "scenario"
    type: "object"
    properties:
      - name: "name"
        type: "string"
        required: true
      - name: "severity"
        type: "enum"
        choices: ["low", "medium", "high"]
```

**Complete Example:**
```yaml
input_schema:
  - name: "test_type"
    type: "enum"
    choices: ["security", "performance", "accuracy"]
    required: true

  - name: "config"
    type: "object"
    properties:
      - name: "max_iterations"
        type: "integer"
        default: 10
      - name: "attack_types"
        type: "list"
        items: "string"
      - name: "threshold"
        type: "enum"
        choices: ["low", "medium", "high"]
        default: "medium"
```

### 3. Test Implementation

Create `entrypoint.py` following the standardized interface:

```python
#!/usr/bin/env python3
import argparse
import json
import sys

def evaluate_response(response):
    """Evaluate if a response passes. Implement your logic here."""
    content = response.choices[0].message.content
    return len(content) > 10  # Simple check

def run_tests(sut_params, test_params):
    """
    Your custom test logic.

    Args:
        sut_params: System under test configuration
        test_params: Test parameters from YAML

    Returns:
        Dict matching output_metrics in manifest
    """
    # Create LLM client
    from openai import OpenAI
    client = OpenAI(
        base_url=sut_params.get("base_url"),
        api_key=sut_params.get("api_key")
    )

    # Run your tests
    num_tests = test_params.get("num_tests", 10)
    passed = 0

    for i in range(num_tests):
        response = client.chat.completions.create(
            model=sut_params.get("model"),
            messages=[{"role": "user", "content": f"Test {i}"}]
        )
        # Your evaluation logic
        if evaluate_response(response):
            passed += 1

    return {
        "success": True,
        "score": (passed / num_tests) if num_tests > 0 else 0.0,
        "tests_run": num_tests
    }

def main():
    """Standard ASQI container interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=True,
                       help="JSON with system configurations")
    parser.add_argument("--test-params", required=True,
                       help="JSON with test parameters")
    args = parser.parse_args()

    try:
        systems = json.loads(args.systems_params)
        params = json.loads(args.test_params)

        # Extract system under test
        sut = systems.get("system_under_test", {})

        # Optional: Extract additional systems
        evaluator = systems.get("evaluator_system", sut)

        # Run tests
        results = run_tests(sut, params)

        # Output JSON to stdout
        print(json.dumps(results, indent=2))

    except Exception as e:
        # Always output JSON, even on error
        error_result = {
            "success": False,
            "error": str(e),
            "score": 0.0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Key Requirements:**
- Accept `--systems-params` and `--test-params` as JSON strings
- Extract `system_under_test` from systems_params
- Output JSON to stdout matching your `output_metrics`
- Handle errors gracefully with JSON output

### 4. Container Build

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY entrypoint.py manifest.yaml ./
RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "entrypoint.py"]
```

Create `requirements.txt`:

```txt
openai>=1.0.0
# Add your dependencies
```

Build and test:

```bash
# Build
docker build -t my-registry/my_tester:latest .

# Test locally
docker run --rm my-registry/my_tester:latest \
  --systems-params '{
    "system_under_test": {
      "base_url": "http://localhost:4000/v1",
      "model": "gpt-4o-mini",
      "api_key": "sk-1234"
    }
  }' \
  --test-params '{"num_tests": 3}'
```

---

## Using the ASQI Package

If your container needs to import from the `asqi` package (e.g. `asqi.datasets.load_hf_dataset`), you can include it as a build-time dependency.

### How It Works

The script `scripts/prepare_asqi_pkg.sh` copies `src/asqi` from the repository root into your container's build context as `asqi_pkg/`. Your Dockerfile then installs it as a local package.

### Setup

**1. Reference `asqi_pkg/` in your Dockerfile:**

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install asqi package
COPY asqi_pkg/ /tmp/asqi_pkg/
RUN uv pip install --system --no-cache /tmp/asqi_pkg/ && rm -rf /tmp/asqi_pkg/

# Install container dependencies
COPY pyproject.toml .
RUN uv pip install --system --no-cache -e .

COPY entrypoint.py manifest.yaml ./

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

ENTRYPOINT ["python", "./entrypoint.py"]
```

**2. Run the prep script before building:**

```bash
scripts/prepare_asqi_pkg.sh test_containers/my_tester
docker build -t my-registry/my_tester:latest test_containers/my_tester
```

The CI pipeline runs this automatically for any container whose Dockerfile references `asqi_pkg/`. The generated `asqi_pkg/` directory is git-ignored.

**3. Import in your entrypoint:**

```python
from asqi.datasets import load_hf_dataset
```

See `test_containers/hf_vision_tester/` for a working example.

---

## Advanced Features

### Technical Reports (Optional)

Generate HTML/PDF reports alongside test results.

**Output Format:**
```python
output = {
  "test_results": {
    "success": True,
    "score": 0.95
  },
  "generated_reports": [
    {
      "report_name": "summary",
      "report_type": "html",
      "report_path": "/output/summary.html"
    }
  ]
}
```

**Manifest Declaration:**
```yaml
output_reports:
  - name: "summary"
    type: "html"
    description: "Test execution summary"
```

**Requirements:**
- Report names must match between manifest and output
- Files must be written to mounted output volume
- One-to-one correspondence between declared and generated reports

See `test_containers/mock_tester/` for a complete example with report generation.

### Multi-System Testing

Test containers can coordinate multiple systems:

```yaml
input_systems:
  - name: "system_under_test"
    type: "llm_api"
    required: true
  - name: "simulator_system"
    type: "llm_api"
    required: false
  - name: "evaluator_system"
    type: "llm_api"
    required: false
```

```python
# In entrypoint.py
systems = json.loads(args.systems_params)
sut = systems.get("system_under_test")
simulator = systems.get("simulator_system", sut)  # Fallback to SUT
evaluator = systems.get("evaluator_system", sut)  # Fallback to SUT
```

### Input Datasets

Declare dataset requirements in manifest:

```yaml
input_datasets:
  - name: "test_data"
    type: "huggingface"
    required: true
    features:
      - name: "prompt"
        dtype: "string"
      - name: "expected"
        dtype: "string"
```

Access datasets via mounted volumes in your container.

---

## Reference

**Working Examples:**
- `test_containers/mock_tester/` - Basic test container with reports
- `test_containers/garak/` - Security testing framework
- `test_containers/chatbot_simulator/` - Multi-system testing

**System Types:**
- `llm_api` - Language models (OpenAI-compatible)
- `vlm_api` - Vision-language models
- `rag_api` - RAG systems with retrieval
- `image_generation_api` - Image generation models
- `rest_api` - Generic REST APIs

**Validation:**
- ASQI validates parameter presence (required fields exist)
- Containers validate actual parameter values
- Rich type metadata provides IDE support and documentation

For complete API details, see [Configuration](configuration.md).
