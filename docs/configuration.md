# Configuration

ASQI uses YAML configuration files to define systems, test suites, and score cards. All configurations use Pydantic schemas for type safety and include JSON Schema files for IDE integration.

## IDE Integration

For the best development experience, add schema references to your YAML files:

```yaml
# For systems configuration files
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_systems_config.schema.json

# For test suite files  
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_suite_config.schema.json

# For score card files
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_score_card.schema.json

# For test container manifest files
# yaml-language-server: $schema=https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/refs/heads/main/src/asqi/schemas/asqi_manifest.schema.json
```

This enables real-time validation, autocompletion, and documentation in VS Code, PyCharm, and other editors using the published schemas from GitHub.

## Systems Configuration

Systems represent the AI services, models, and APIs that participate in testing.

### LLM API Systems

The primary system type for language models using OpenAI-compatible APIs:

```yaml
systems:
  # Direct provider configuration
  openai_gpt4o_mini:
    type: "llm_api"
    description: "Fast and General Purpose Model from OpenAI"
    provider: "openai"
    params:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o-mini"
      api_key: "sk-your-openai-key"

  # LiteLLM proxy configuration
  proxy_llm:
    type: "llm_api"
    description: "Fast and General Purpose Model from OpenAI"
    provider: "openai"
    params:
      base_url: "http://localhost:4000/v1"
      model: "gpt-4o-mini"
      api_key: "sk-1234"

  # Using environment variable fallbacks
  fallback_llm:
    type: "llm_api"
    description: "Custom Model"
    provider: "custom"
    params:
      model: "my-model"
      # base_url and api_key will use fallbacks from .env
```

### RAG API Systems

`rag_api` systems extend the OpenAI chat response format with a specified response interface - see expected response schema below. Assuming an API has been configured with to support RAG functionality, you can define RAG systems as follows:

#### System Configuration

Configure RAG systems in your `litellm_config.yaml`:

```yaml
model_list:
  # ... existing models ...

  # RAG API Systems - Retrieval-Augmented Generation with contextual retrieval
  - model_name: custom_rag_chatbot
    litellm_params:
      model: custom_rag
      api_key: os.environ/RAG_API_KEY  # Replace with your actual RAG endpoint authentication
```

Then reference it in your ASQI systems configuration:

```yaml
systems:
  # LiteLLM proxy configuration
  rag_proxy_system:
    type: "rag_api"
    description: "Custom RAG chatbot"
    provider: "openai"
    params:
      base_url: "http://localhost:4000/v1"
      model: "custom_rag"
      api_key: "sk-1234"
```

#### Expected Request Format

ASQI sends OpenAI-compatible chat completion requests to RAG systems. The request format is identical to `llm_api` systems, using standard chat completion parameters:

```json
{
  "model": "my-rag-model",
  "messages": [
    {"role": "user", "content": "What is the company's refund policy?"}
  ],
  "temperature": 0.0
}
```

**Optional Parameters:**
- `user_group` (string): When specified as a test input parameter, it may be passed to the RAG system for access control tests in the request payload.

```json
{
  "model": "my-rag-model",
  "messages": [
    {"role": "user", "content": "What is the company's refund policy?"}
  ],
  "temperature": 0.0,
  "user_group": "admin"
}
```

#### Expected Response Schema

RAG API systems must return responses in OpenAI-compatible chat completions format with an additional `context` field in each message containing retrieval citations.

**Context Field Requirements:**
- `context`: Object containing retrieval information (required)
- `context.citations`: Array of citation objects (required)
- Each citation object contains:
  - `retrieved_context` (string): The retrieved information text
  - `document_id` (string): A stable identifier for the originating document
  - `score` (float, optional): Retrieval ranking or confidence score, normalized to range [0.0, 1.0] where 1.0 indicates highest confidence/relevance
  - `source_id` (string, optional): Collection / index / knowledge-base identifier

**Example Response:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "my-rag-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "We offer 30-day returns at no additional cost for all customers",
        "context": {
          "citations": [
            {
              "retrieved_context": "All customers are eligible for a 30-day full refund at no extra cost.",
              "document_id": "return_policy.pdf",
              "score": 0.96,
              "source_id": "company_policy"
            },
            {
              "retrieved_context": "We need receipt for 30-day refund",
              "document_id": "return_policy.pdf",
              "score": 0.7,
              "source_id": "company_policy"
            }
          ]
        }
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Image Generation API Systems

`image_generation_api` systems support text-to-image generation using OpenAI-compatible APIs.

#### System Configuration

Configure image generation systems in your `litellm_config.yaml`:

```yaml
model_list:
  # ... existing models ...

  # Image Generation Systems (from OpenAI)
  - model_name: "openai/*"
    litellm_params:
      model: "openai/*"
      api_key: os.environ/OPENAI_API_KEY
```

Then reference it in your ASQI systems configuration:

```yaml
systems:
  # LiteLLM proxy configuration
  dalle3_generator:
    type: "image_generation_api"
    description: "OpenAI DALL-E 3 Image Generator"
    params:
      base_url: "http://localhost:4000/v1"
      model: "openai/dall-e-3"
      api_key: "sk-1234"
```

#### Expected Request Format

ASQI sends OpenAI-compatible image generation requests:

```json
{
  "model": "openai/dall-e-3",
  "prompt": "A cute baby sea otter, in an animated style",
  "n": 1,
  "size": "1024x1024",
  "response_format": "url"
}
```

#### Expected Response Schema

Image generation systems return OpenAI-compatible image generation responses:

```json
{
  "created": 1703658209,
  "data": [
    {
      "url": "https://example.com/generated_image.png",
      "revised_prompt": "A cute baby sea otter..."
    }
  ]
}
```

### Image Editing API Systems

`image_editing_api` systems support image-to-image editing using OpenAI-compatible APIs.

#### System Configuration

Configure image editing systems in your `litellm_config.yaml`:

```yaml
model_list:
  # ... existing models ...

  # Image Editing Systems
  - model_name: "openai/*"
    litellm_params:
      model: "openai/*"
      api_key: os.environ/OPENAI_API_KEY
```

Then reference it in your ASQI systems configuration:

```yaml
systems:
  # LiteLLM proxy configuration
  dalle3_editor:
    type: "image_editing_api"
    description: "OpenAI DALL-E 3 Image Editor"
    params:
      base_url: "http://localhost:4000/v1"
      model: "openai/dall-e-3"
      api_key: "sk-1234"
```

#### Expected Request Format

ASQI sends OpenAI-compatible image editing requests (multipart/form-data):

```
POST /v1/images/edits
Content-Type: multipart/form-data

image: <uploaded_image.png>
prompt: "Change the background to a beach scene"
model: "dall-e-3"
n: 1
size: "1024x1024"
```

#### Expected Response Schema

Image editing systems return the same format as image generation APIs.

### VLM API Systems

`vlm_api` systems support vision language models that can process both text and images.

#### System Configuration

Configure VLM systems in your `litellm_config.yaml`:

```yaml
model_list:
  # ... existing models ...

  # Vision Language Models
  - model_name: "openai/*"
    litellm_params:
      model: "openai/*"
      api_key: os.environ/OPENAI_API_KEY
```

Then reference it in your ASQI systems configuration:

```yaml
systems:
  # LiteLLM proxy configuration
  gpt4_1_mini_vlm:
    type: "vlm_api"
    description: "OpenAI GPT-4.1-Mini VLM Evaluator"
    params:
      base_url: "http://localhost:4000/v1"
      model: "openai/gpt-4.1-mini"
      api_key: "sk-1234"
```

#### Expected Request Format

ASQI sends multimodal chat completion requests:

```json
{
  "model": "openai/gpt-4.1-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Evaluate this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
      ]
    }
  ],
  "max_tokens": 150
}
```

#### Expected Response Schema

VLMs return standard chat completion format with text responses.

### Environment Variable Handling

ASQI supports a three-level configuration hierarchy:

1. **Explicit Parameters** (highest priority): Directly specified in system configuration
2. **Environment File Fallbacks**: Values from `.env` file or custom `env_file`
   
    Configure your environment file with the following variables

    - Required

      - API Configuration
      ```yaml
      # Environment variables to pass into test containers that specifies an env_file
      BASE_URL=http://localhost:4000
      API_KEY=sk-1234 
      ```
    
      - Database
      ```yaml
      # Database connection string
      DBOS_DATABASE_URL=postgres://postgres:asqi@db:5432/asqi_starter
      ```
      
      - Observability
      ```yaml
      # Otel
      OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318/v1/traces
      ```

      - LiteLLM Configuration
      ```yaml
      # Master key for LiteLLM 
      LITELLM_MASTER_KEY="sk-1234"
      ```

    - Optional

      - HuggingFace Token
      ```yaml
      # HuggingFace Token - Required for some gated datasets
      HF_TOKEN=hf_api_V9oSu3L1onGE0Yz2s2swlT8ZtJ
      ``` 

      - Container logs
      ```yaml
      # Path for the container logs (default: logs)
      LOGS_FOLDER=asqi/logs
      ``` 
     
      - API Keys
      ```yaml
      # OpenAI
      OPENAI_API_KEY=api-key-openai
      # Anthropic
      ANTHROPIC_API_KEY=api-key-anthropic
      # Amazon Bedrock
      AWS_BEARER_TOKEN_BEDROCK=api-key-bedrock
      ```       

3. **Validation Error**: If required fields are missing

#### Environment File Reference

Systems can specify a custom environment file:

```yaml
systems:
  production_system:
    type: "llm_api"
    description: "High Performance Model from OpenAI with Reasoning Capabilities"
    provider: "openai"
    params:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o"
      env_file: "production.env"  # Custom environment file
```

### String Interpolation

ASQI supports environment variable interpolation directly in YAML configuration files using shell-style syntax:

```yaml
# Direct substitution - uses environment variable or empty string if not set
image: ${REGISTRY}/my-app:latest

# Default value - uses environment variable or default if not set/unset
image: ${REGISTRY:-docker.io}/my-app:latest

# Default if unset - uses environment variable (including empty) or default if unset
image: ${REGISTRY-docker.io}/my-app:latest
```

#### Examples

```yaml
suite_name: "Dynamic Testing Suite"
description: "Runs Security Tests"
test_suite:
  - id: "registry_test"
    name: "registry test"
    description: "Test for Security Vulnerabilities using Garak"
    image: ${REGISTRY:-my-registry}/garak:latest
    systems_under_test: ["${TARGET_SYSTEM:-openai_gpt4o}"]
    params:
      api_key: "${API_KEY}"
      model: "${MODEL:-gpt-4o-mini}"
```

## Test Suite Configuration

Test suites define collections of tests to execute against your systems.

- Tests ID field (id)
    
    This is the unique identifier for the tests across the project.
    This approach is a slight modification of the standard RFC 9562
    - Valid Characters: 0-9, a-z, _ 
    - Max Length: 32

### Basic Test Suite

```yaml
suite_name: "Basic Mock Testing"
description: "Simple Compatibility Checks"
test_suite:
  - id: "compatibility_check"
    name: "compatibility check"
    description: "Verifies Basic Compatibility"
    image: "my-registry/mock_tester:latest"
    systems_under_test: ["my_llm_service"]
    params:
      delay_seconds: 1
```

### Multi-System Tests

Tests can coordinate multiple AI systems for complex scenarios:

```yaml
suite_name: "Advanced Chatbot Testing"
description: "Evaluates Chatbot Performance, Safety..."
test_suite:
  - id: "chatbot_simulation"
    name: "chatbot simulation"
    description: "Simulates Realistic Conversations with the Chatbot"
    image: "my-registry/chatbot_simulator:latest"
    systems_under_test: ["my_chatbot"]
    systems:
      simulator_system: "gpt4o_simulator"
      evaluator_system: "claude_evaluator"
    params:
      chatbot_purpose: "customer service"
      num_scenarios: 5
      sycophancy_level: "medium"
```

### Multiple Tests in One Suite

```yaml
suite_name: "Comprehensive Security Testing"
description: "Spot Vulnerabilities in the Target Model"
test_suite:
  - id: "prompt injection test"
    name: "prompt_injection_test"
    description: "Checks if the Model Can be Tricked by Malicious Prompts"
    image: "my-registry/garak:latest"
    systems_under_test: ["target_model"]
    params:
      probes: ["promptinject"]
      generations: 10

  - id: "encoding_attack_test"
    name: "encoding attack test"
    description: "Tests the Model Against Attacks Using Encoded Inputs"
    image: "my-registry/garak:latest" 
    systems_under_test: ["target_model"]
    params:
      probes: ["encoding.InjectHex"]
      generations: 5

  - id: "red_team_assessment"
    name: "red team assessment"
    description: "Simulates Attacks to Find Jailbreaks or Injections"
    image: "my-registry/deepteam:latest"
    systems_under_test: ["target_model"]
    params:
      attack_types: ["jailbreak", "prompt_injection"]
      max_iterations: 20
```

### Test Suite with Input Datasets

Tests can reference datasets from a dataset registry:

```yaml
suite_name: "Dataset-based Evaluation"
test_suite:
  - id: "benchmark_eval"
    name: "benchmark evaluation"
    description: "Evaluate using standard benchmark dataset"
    image: "my-registry/evaluator:latest"
    systems_under_test: ["my_chatbot"]
    input_datasets:
      evaluation_data: "benchmark_questions"  # References dataset from registry
      source_docs: "company_handbook"
    volumes:
      input: "data/inputs/"
      output: "data/outputs/"
    params:
      batch_size: 10
```

See [Dataset Support](datasets.md) for detailed documentation on using datasets in test suites.

## Dataset Registry Configuration

The dataset registry allows you to define reusable datasets that can be referenced across multiple test suites and generation jobs.

### Basic Structure

```yaml
# yaml-language-server: $schema=../src/asqi/schemas/asqi_datasets_config.schema.json

datasets:
  dataset_name:
    type: "huggingface" | "pdf" | "txt"
    description: "Optional description"
    # Type-specific fields...
```

### HuggingFace Datasets

```yaml
datasets:
  eval_questions:
    type: "huggingface"
    description: "Evaluation questions for chatbot testing"
    loader_params:
      builder_name: "json"           # Format: json, csv, parquet, arrow, text, etc.
      data_files: "questions.json"   # File path relative to input mount
      # OR for directories:
      # data_dir: "dataset_folder/"
    mapping:
      # Map actual dataset columns to expected feature names
      question: "prompt"
      answer: "response"
    tags: ["evaluation", "en"]
```

**Loader Parameters:**
- `builder_name`: Dataset format (`json`, `csv`, `parquet`, `arrow`, `text`, `xml`, `imagefolder`, `audiofolder`, `videofolder`)
- `data_files`: Single file or list of files (relative to input mount)
- `data_dir`: Directory containing dataset files (alternative to `data_files`)
- `revision`: Git revision for HuggingFace Hub datasets (optional, for Hub datasets only)

**Column Mapping:**
The `mapping` field translates dataset column names to container-expected feature names:
- Keys: Actual column names in the dataset
- Values: Expected feature names from container manifest

### PDF Datasets

```yaml
datasets:
  company_handbook:
    type: "pdf"
    description: "Company policy handbook for RAG testing"
    file_path: "handbook.pdf"  # Path relative to input mount
    tags: ["rag", "documents"]
```

### Text File Datasets

```yaml
datasets:
  product_catalog:
    type: "txt"
    description: "Product descriptions corpus"
    file_path: "products.txt"  # Path relative to input mount
    tags: ["generation", "source"]
```

### Complete Example

```yaml
datasets:
  # HuggingFace dataset with mapping
  benchmark_v1:
    type: "huggingface"
    description: "Standard QA benchmark dataset"
    loader_params:
      builder_name: "json"
      data_files: "benchmark_qa.json"
    mapping:
      input_text: "prompt"
      expected_output: "response"
    tags: ["evaluation", "benchmark", "v1"]
  
  # PDF document
  rag_source_docs:
    type: "pdf"
    description: "Source documents for RAG data generation"
    file_path: "knowledge_base.pdf"
    tags: ["rag", "source"]
  
  # Text file
  training_corpus:
    type: "txt"
    description: "Training text corpus"
    file_path: "corpus.txt"
    tags: ["training", "text"]
```

## Data Generation Configuration

Data generation configuration defines synthetic data generation jobs using data generation containers.

### Basic Structure

```yaml
# yaml-language-server: $schema=../../src/asqi/schemas/asqi_generation_config.schema.json

job_name: "Generation Job Name"
generation_jobs:
  - id: "job_id"
    name: "Human-readable job name"
    image: "container-image:tag"
    systems:
      generation_system: "system_name"
    input_datasets:
      dataset_alias: "dataset_reference"
    volumes:
      input: "input/path/"
      output: "output/path/"
    params:
      # Container-specific parameters
```

### Field Descriptions

**Required Fields:**
- `job_name`: Name of the overall generation job
- `generation_jobs`: List of individual generation job configurations
  - `id`: Unique identifier for the job (a-z, 0-9, _, max 32 chars)
  - `name`: Human-readable job name
  - `image`: Docker image for the data generation container

**Optional Fields:**
- `systems`: Systems used for generation (not systems under test)
- `input_datasets`: Input datasets for data generation
  - Can reference datasets from registry (string value)
  - Can define inline datasets (object with file_path, etc.)
- `output_datasets`: Expected output datasets (usually declared in container manifest)
- `params`: Parameters passed to the container
- `volumes`: Input/output directory mounts
- `env_file`: Path to environment file
- `environment`: Environment variables for the container

### Dataset References

Reference datasets from the registry or define them inline:

```yaml
# Reference from registry
input_datasets:
  evaluation_data: "benchmark_questions"

# Inline definition
input_datasets:
  source_documents_pdf:
    file_path: "sample.pdf"
```

## Score Card Configuration

Score cards define automated assessment criteria for test results. They evaluate individual test executions (not aggregated results).

- Indicators ID field (id)
    
    This is the unique identifier for the indicator across the project.
    This approach is a slight modification of the standard RFC 9562
    - Valid Characters: 0-9, a-z, _ 
    - Max Length: 32

### Basic Score Card Structure

```yaml
score_card_name: "Production Readiness Assessment"
indicators:
  - id: "test_success_requirement"
    name: "Test Success Requirement"
    apply_to:
      test_id: "security_scan"
    metric: "success"
    assessment:
      - { outcome: "PASS", condition: "equal_to", threshold: true }
      - { outcome: "FAIL", condition: "equal_to", threshold: false }
```

### Assessment Conditions

Score cards support various comparison operators:

```yaml
indicators:
  - id: "performance_score_assessment"
    name: "Performance Score Assessment"
    apply_to:
      test_id: "benchmark_test"
    metric: "score"
    assessment:
      - { outcome: "EXCELLENT", condition: "greater_equal", threshold: 0.9 }
      - { outcome: "GOOD", condition: "greater_equal", threshold: 0.8 }
      - { outcome: "ACCEPTABLE", condition: "greater_equal", threshold: 0.7 }
      - { outcome: "NEEDS_IMPROVEMENT", condition: "less_than", threshold: 0.7 }

  - id: "security_threshold"
    name: "Security Threshold"
    apply_to:
      test_id: "vulnerability_scan"
    metric: "vulnerabilities_found"
    assessment:
      - { outcome: "SECURE", condition: "equal_to", threshold: 0 }
      - { outcome: "LOW_RISK", condition: "less_equal", threshold: 2 }
      - { outcome: "HIGH_RISK", condition: "greater_than", threshold: 2 }
```


### Displaying Technical Reports in Score Cards

To display technical reports in a score card, use the `display_reports` field in your indicator configuration. Each entry in `display_reports` must match the `name` of a report declared in the test container's manifest under `output_reports`.

#### Selecting Reports for Display

- In your indicator, you can reference one or more of the test reports by name using `display_reports`

```yaml
indicators:
  - id: "garak_security_check"
    name: "Garak Security Check"
    apply_to:
      test_id: "garak_prompt_injection"
    display_reports: ["quick_summary", "detailed_metrics"]
    metric: "attack_success_rate"
    assessment:
      - { outcome: "SECURE", condition: "equal_to", threshold: 0.0 }
      - { outcome: "VULNERABLE", condition: "greater_than", threshold: 0.0 }
```

#### Report Validations

- Every report listed in `display_reports` exists in the container manifest (`output_reports`).
- There are no duplicate report names in `display_reports`.

### Available Conditions

- `equal_to`: Exact value matching (supports boolean and numeric)
- `greater_than` / `less_than`: Strict numeric comparisons
- `greater_equal` / `less_equal`: Inclusive numeric comparisons

### Targeting Specific Tests

Use the `apply_to` field to target specific tests:

```yaml
indicators:
  - id: "garak_security_check"
    name: "Garak Security Check"
    apply_to:
      test_id: "garak_prompt_injection"  # Only applies to this test
    metric: "attack_success_rate"
    assessment:
      - { outcome: "SECURE", condition: "equal_to", threshold: 0.0 }
      - { outcome: "VULNERABLE", condition: "greater_than", threshold: 0.0 }
```

### Enforcing Certain System Type

Score card indicators can additional specify that the test results should be from certain system types using the `target_system_type` field. This is useful when a indicator only applies to one or multiple system types and should not be used for other types.

**Single System Type:**
```yaml
indicators:
  - id: "llm_accuracy_check"
    name: "LLM Accuracy Check"
    apply_to:
      test_id: "multi_modal_test"
      target_system_type: "llm_api"  # Only applies to LLM systems
    metric: "accuracy"
    assessment:
      - { outcome: "PASS", condition: "greater_equal", threshold: 0.85 }
      - { outcome: "FAIL", condition: "less_than", threshold: 0.85 }

  - id: "vlm_accuracy_check"
    name: "VLM Accuracy Check"
    apply_to:
      test_id: "multi_modal_test"
      target_system_type: "vlm_api"  # Only applies to VLM systems
    metric: "accuracy"
    assessment:
      - { outcome: "PASS", condition: "greater_equal", threshold: 0.75 }  # Different threshold
      - { outcome: "FAIL", condition: "less_than", threshold: 0.75 }
```

**Multiple System Types:**
```yaml
indicators:
  - id: "general_accuracy_check"
    name: "General Accuracy Check"
    apply_to:
      test_id: "multi_modal_test"
      target_system_type: ["llm_api", "vlm_api"]  # Applies to both types
    metric: "accuracy"
    assessment:
      - { outcome: "PASS", condition: "greater_equal", threshold: 0.70 }
      - { outcome: "FAIL", condition: "less_than", threshold: 0.70 }
```

**No System Type Filter (Default):**

If `target_system_type` is omitted, the indicator applies to all system types for the specified test:

```yaml
indicators:
  - id: "success_check"
    name: "Success Check"
    apply_to:
      test_id: "compatibility_test"
      # No target_system_type - applies to all system types
    metric: "success"
    assessment:
      - { outcome: "PASS", condition: "equal_to", threshold: true }
```

**System Types:**
- `llm_api` - Language models
- `vlm_api` - Vision-language models
- `rag_api` - RAG systems
- `rest_api` - REST API endpoints
- `image_generation_api` - Image generation models
- `image_editing_api` - Image editing models

### Metric Expressions

Combine multiple metrics using mathematical operations and functions for sophisticated composite scoring.

#### Basic Usage

Simple metric path (backward compatible):
```yaml
metric: "accuracy_score"
```

Expression format for combining metrics:
```yaml
metric:
  expression: "0.7 * accuracy + 0.3 * relevance"
  values:
    accuracy: "metrics.answer_accuracy"
    relevance: "metrics.answer_relevance"
```

**Key components:**
- `expression`: Mathematical formula using variable names
- `values`: Maps variable names to metric paths in test results

#### Supported Operations

**Arithmetic Operators:** `+`, `-`, `*`, `/`, `()`

**Comparison Operators:** `>`, `>=`, `<`, `<=`, `==`, `!=`

**Boolean Operators:** `and`, `or`, `not`

**Conditional:** `if-else` expressions for conditional logic

**Functions:**
- `min(...)`, `max(...)`, `avg(...)` - Aggregation
- `abs(x)` - Absolute value
- `round(x, n)` - Round to n decimals
- `pow(x, y)` - Power (x^y)

#### Common Patterns

Weighted average:
```yaml
expression: "0.5 * accuracy + 0.3 * speed + 0.2 * reliability"
values: { accuracy: "test_accuracy", speed: "response_time", reliability: "uptime" }
```

All metrics must pass:
```yaml
expression: "min(security, privacy, compliance)"
values: { security: "sec_score", privacy: "priv_score", compliance: "comp_score" }
```

Best performer:
```yaml
expression: "max(model_a, model_b, model_c)"
values: { model_a: "models.a.score", model_b: "models.b.score", model_c: "models.c.score" }
```

Capped composite:
```yaml
expression: "min((0.4 * speed + 0.6 * quality), 1.0)"
values: { speed: "perf.speed_score", quality: "perf.quality_score" }
```

Hard gates with AND conditions (returns score if all gates pass, else penalty):
```yaml
expression: "(0.45 * accuracy + 0.35 * relevance + 0.20 * helpfulness) if (faith >= 0.7 and retrieval >= 0.6) else -1"
values: 
  accuracy: "metrics.accuracy"
  relevance: "metrics.relevance"
  helpfulness: "metrics.helpfulness"
  faith: "metrics.faithfulness"
  retrieval: "metrics.retrieval"
```

Gate compliance counting (counts how many gates pass):
```yaml
expression: "(accuracy >= 0.8) + (relevance >= 0.75) + (helpfulness >= 0.7)"
values:
  accuracy: "metrics.accuracy"
  relevance: "metrics.relevance"
  helpfulness: "metrics.helpfulness"
```

Flexible OR gating (meets A or B requirement):
```yaml
expression: "1 if (performance >= 80 or cost <= 0.01) else 0"
values:
  performance: "metrics.performance_score"
  cost: "metrics.cost_per_request"
```

Nested conditional tiers (tiered scoring):
```yaml
expression: "0.95 if (risk < 0.1) else (0.75 if (risk < 0.3) else (0.5 if (risk < 0.5) else 0.2))"
values:
  risk: "metrics.risk_score"
```

### Audit Indicators

Audit indicators represent **human-reviewed assessment items** that do **not** reference test metrics.
They require a corresponding manual entry in `audit_responses.yaml` unless skipped using `--skip-audit`.

```yaml
# Example audit indicator
- id: "configuration_complexity"
  type: "audit"
  name: "Configuration Complexity"
  assessment:
    - outcome: "A"
      description: "Simple configuration with minimal technical effort"
    - outcome: "B"
      description: "Moderate configuration requiring some understanding"
    - outcome: "C"
      description: "Requires expert knowledge or prompt engineering skill"
```

### Audit Responses File

Audit responses need to be provided separately:

```yaml
responses:
  - indicator_id: configuration_complexity
    sut_name: "openai_gpt4o_mini"  # Optional; when provided, response is per system
    selected_outcome: "B"
    notes: "Some domain knowledge needed during setup"
  - indicator_id: configuration_complexity
    sut_name: "nova_lite"
    selected_outcome: "C"
    notes: "Requires prompt engineering and additional infra"
```

When any response includes `sut_name`, provide entries for **every** system under test.
If an entry references a system that was not part of the evaluation, the score card will
return an error.
Do **not** mix global (no `sut_name`) and per-system (`sut_name` present) responses for the
same indicator—this combination is rejected with an explicit error.

#### Complete Example

```yaml
score_card_name: "Comprehensive Assessment"
indicators:
  # Simple metric
  - id: "basic_success"
    name: "Success Check"
    apply_to: { test_id: "compatibility_test" }
    metric: "success"
    assessment:
      - { outcome: "PASS", condition: "equal_to", threshold: true }
  
  # Weighted composite
  - id: "quality_score"
    name: "Overall Quality"
    apply_to: { test_id: "chatbot_test" }
    metric:
      expression: "0.4 * accuracy + 0.3 * relevance + 0.3 * consistency"
      values:
        accuracy: "average_answer_accuracy"
        relevance: "average_answer_relevance"
        consistency: "consistency_score"
    assessment:
      - { outcome: "Excellent", condition: "greater_equal", threshold: 0.9 }
      - { outcome: "Good", condition: "greater_equal", threshold: 0.75 }
  
  # Minimum threshold
  - id: "min_requirements"
    name: "All Metrics Pass"
    apply_to: { test_id: "chatbot_test" }
    metric:
      expression: "min(accuracy, relevance, consistency)"
      values:
        accuracy: "average_answer_accuracy"
        relevance: "average_answer_relevance"
        consistency: "consistency_score"
    assessment:
      - { outcome: "Pass", condition: "greater_equal", threshold: 0.7 }
      - { outcome: "Fail", condition: "less_than", threshold: 0.7 }

  # Hard gates with AND conditions
  - id: "accuracy_with_quality_gates"
    name: "Accuracy Score with Quality Gates"
    apply_to: { test_id: "chatbot_test" }
    metric:
      expression: "(0.45 * accuracy + 0.35 * relevance + 0.20 * helpfulness) if (faith >= 0.7 and retrieval >= 0.6 and instruction >= 0.7) else -1"
      values:
        accuracy: "metrics.answer_accuracy"
        relevance: "metrics.answer_relevance"
        helpfulness: "metrics.helpfulness"
        faith: "metrics.faithfulness"
        retrieval: "metrics.retrieval"
        instruction: "metrics.instruction_following"
    assessment:
      - { outcome: "A", condition: "greater_equal", threshold: 0.8 }
      - { outcome: "B", condition: "greater_equal", threshold: 0.7 }
      - { outcome: "C", condition: "greater_equal", threshold: 0.6 }
      - { outcome: "F", condition: "less_than", threshold: 0.6 }

  # Gate compliance counting with comparisons
  - id: "gate_compliance"
    name: "Quality Gates Passed"
    apply_to: { test_id: "chatbot_test" }
    metric:
      expression: "(accuracy >= 0.8) + (relevance >= 0.75) + (helpfulness >= 0.7) + (faithfulness >= 0.7)"
      values:
        accuracy: "metrics.answer_accuracy"
        relevance: "metrics.answer_relevance"
        helpfulness: "metrics.helpfulness"
        faithfulness: "metrics.faithfulness"
    assessment:
      - { outcome: "A", condition: "greater_equal", threshold: 4 }
      - { outcome: "B", condition: "greater_equal", threshold: 3 }
      - { outcome: "C", condition: "greater_equal", threshold: 2 }
      - { outcome: "F", condition: "less_than", threshold: 2 }

  # Flexible OR gating
  - id: "performance_or_cost"
    name: "Performance OR Cost Target Met"
    apply_to: { test_id: "benchmark_test" }
    metric:
      expression: "1 if (throughput >= 50 or cost_per_token <= 0.001) else 0"
      values:
        throughput: "metrics.tokens_per_second"
        cost_per_token: "metrics.cost_per_token"
    assessment:
      - { outcome: "Pass", condition: "equal_to", threshold: 1 }
      - { outcome: "Fail", condition: "equal_to", threshold: 0 }

  # Nested conditional tiers
  - id: "risk_tiered_score"
    name: "Safety Score Based on Risk Tier"
    apply_to: { test_id: "security_test" }
    metric:
      expression: "0.95 if (risk < 0.1) else (0.75 if (risk < 0.3) else (0.5 if (risk < 0.5) else 0.2))"
      values:
        risk: "metrics.risk_score"
    assessment:
      - { outcome: "A", condition: "greater_equal", threshold: 0.9 }
      - { outcome: "B", condition: "greater_equal", threshold: 0.7 }
      - { outcome: "C", condition: "greater_equal", threshold: 0.5 }
      - { outcome: "F", condition: "less_than", threshold: 0.5 }

  # Audit indicator
  - id: "configuration_complexity"
    type: "audit"
    name: "Configuration Complexity"
    assessment:
      - outcome: "A"
        description: "Simple configuration with minimal technical effort"
      - outcome: "B"
        description: "Moderate configuration requiring some understanding"
      - outcome: "C"
        description: "Requires expert knowledge or prompt engineering skill"
```

**Security:** Expressions run in a sandboxed environment with AST validation—no code execution, imports, or file access allowed.

See `config/score_cards/expression_examples_score_card.yaml` for more examples.

## Container Interface Specification

### Standardized Entry Point

All test containers must implement a standardized interface:

```python
# entrypoint.py
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems-params", required=True, help="JSON string with system configurations")
    parser.add_argument("--test-params", required=True, help="JSON string with test parameters")
    args = parser.parse_args()
    
    systems_params = json.loads(args.systems_params)
    test_params = json.loads(args.test_params)
    
    # Extract systems
    sut_params = systems_params.get("system_under_test", {})
    simulator_system = systems_params.get("simulator_system", sut_params)
    evaluator_system = systems_params.get("evaluator_system", sut_params)
    
    # Run your test logic here
    results = run_test(sut_params, test_params, simulator_system, evaluator_system)
    
    # Output JSON results to stdout
    print(json.dumps(results))
```

#### Entry Point Return Format

Your test container should print a JSON to stdout. There are two simple options:

- **Just metrics:**
  ```json
  {
    "success": true,
    "score": 0.95,
    "test_count": 10
  }
  ```
  (All fields match your `output_metrics` in the manifest)

- **Metrics and Reports:**
  ```json
  {
    "test_results": {
      "success": true,
      "score": 0.95,
      "test_count": 10
    },
    "generated_reports": [
      {
        "report_name": "quick_summary",
        "report_type": "html",
        "report_path": "/output/quick_summary.html"
      },
      {
        "report_name": "detailed_metrics",
        "report_type": "pdf",
        "report_path": "/output/detailed_metrics.pdf"
      }
    ]
  }
  ```

  Learn how to add a report to the test container: [Technical reports](custom-test-containers.md#adding-technical-reports-in-custom-test-containers)

### Manifest Declaration

Each container includes a `manifest.yaml` describing its capabilities:

```yaml
name: "advanced_security_tester"
version: "2.0.0"
description: "Comprehensive security testing framework"

input_systems:
  - name: "system_under_test"
    type: "llm_api"
    required: true
    description: "The primary system being tested"
  - name: "evaluator_system"
    type: "llm_api"
    required: false
    description: "Optional system for result evaluation"

input_schema:
  - name: "attack_types"
    type: "array"
    required: true
    description: "List of attack vectors to test"
  - name: "max_iterations"
    type: "integer"
    required: false
    description: "Maximum number of test iterations"

output_metrics:
  - name: "success"
    type: "boolean"
    description: "Whether the test completed successfully"
  - name: "vulnerabilities_found"
    type: "integer"
    description: "Number of vulnerabilities discovered"
  - name: "attack_success_rate"
    type: "float"
    description: "Percentage of successful attacks (0.0 to 1.0)"

output_reports:
  - name: "quick_summary"
    type: "html"
    description: "A quick HTML summary report of the Advanced Security Tester"
  - name: "detailed_metrics"
    type: "pdf"
    description: "PDF metrics report for the Advanced Security Tester"

input_datasets:
  - name: "evaluation_data"
    type: "huggingface"
    required: true
    description: "Evaluation dataset for testing"
    features:
      - name: "prompt"
        dtype: "string"
        description: "Input prompt text"
      - name: "response"
        dtype: "string"
        description: "Expected response"

output_datasets:
  - name: "augmented_dataset"
    type: "huggingface"
    description: "Generated synthetic dataset"
    features:
      - name: "prompt"
        dtype: "string"
      - name: "response"
        dtype: "string"
      - name: "context"
        dtype: "string"
```

### Input Datasets in Manifest

Containers can declare input dataset requirements:

```yaml
input_datasets:
  # HuggingFace dataset with required features
  - name: "evaluation_data"
    type: "huggingface"
    required: true
    description: "Evaluation dataset for testing"
    features:
      - name: "prompt"
        dtype: "string"
        description: "Input prompt text"
      - name: "response"
        dtype: "string"
        description: "Expected response"
  
  # PDF document input
  - name: "source_documents_pdf"
    type: "pdf"
    required: true
    description: "Source PDF documents for processing"
  
  # Text file input
  - name: "corpus_txt"
    type: "txt"
    required: false
    description: "Optional text corpus"
```

**Dataset Types:**
- `huggingface`: Structured datasets (requires `features` field)
- `pdf`: PDF documents
- `txt`: Plain text files

**Feature Data Types:**
Common HuggingFace dataset dtypes include:
- `string`: Text data
- `int32`, `int64`: Integer values
- `float32`, `float64`, `float`, `double`: Floating-point values
- `bool`: Boolean values

See [HuggingFace documentation](https://huggingface.co/docs/datasets/about_dataset_features) for complete list.

### Output Datasets in Manifest

Containers can declare datasets they will generate:

```yaml
output_datasets:
  - name: "augmented_dataset"
    type: "huggingface"
    description: "Augmented version of input dataset"
    features:
      - name: "prompt"
        dtype: "string"
      - name: "response"
        dtype: "string"
      - name: "metadata"
        dtype: "string"
```

Containers return generated dataset information in JSON output:

```json
{
  "results": {
    "success": true
  },
  "generated_datasets": [
    {
      "dataset_name": "augmented_dataset",
      "dataset_type": "huggingface",
      "dataset_path": "/output/augmented_data",
      "format": "parquet",
      "metadata": {
        "num_rows": 1000,
        "num_columns": 3
      }
    }
  ]
}
```

## Validation and Error Handling

### Fail-Fast Validation
- Input validation occurs before expensive operations
- Clear error messages with context and suggestions
- Centralized validation functions in `validation.py`

### Cross-Validation
ASQI performs comprehensive compatibility checking:

1. **System-Test Compatibility**: Ensures systems match test container requirements
2. **Parameter Validation**: Validates test parameters against container schemas
3. **Resource Availability**: Checks Docker image availability and manifest validity
4. **Environment Requirements**: Validates API keys and environment configuration

### Error Recovery
- **DBOS Durability**: Workflows can resume from checkpoints after failures
- **Container Isolation**: Failed containers don't affect other tests
- **Graceful Degradation**: Partial results are preserved even if some tests fail
- **Detailed Logging**: Comprehensive logs for debugging and troubleshooting
