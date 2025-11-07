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
