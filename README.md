# ASQI Engineer

**ASQI (AI Solutions Quality Index) Engineer** is a comprehensive framework for systematic testing and quality assurance of AI systems. Developed from [Resaro's][Resaro] experience bridging governance, technical and business requirements, ASQI Engineer enables rigorous evaluation of AI systems through containerized test packages, automated assessment, and durable execution workflows.

ASQI Engineer is in active development and we welcome contributors to contribute new test packages, share score cards and test plans, and help define common schemas to meet industry needs. Our initial release focuses on comprehensive chatbot testing with extensible foundations for broader AI system evaluation.

## Table of Contents

### For Users

- [Quick Start for Users](#quick-start-for-users) - Install, setup services, and configure systems
- [Test Container Examples](#test-container-examples) - Practical examples for each test container
- [Available Test Containers](#available-test-containers) - Overview of testing capabilities

### For Developers

- [Developer Guide](#developer-guide) - Development setup, architecture, and contributing
- [Environment Configuration](#environment-configuration) - API keys handling and provider setup
- [Usage](#usage) - Command-line usage and execution modes

## Key Features

### **Modular Test Execution**

- **Durable execution**: [DBOS]-powered fault tolerance with automatic retry and recovery
- **Concurrent testing**: Parallel test execution with configurable concurrency limits
- **Container isolation**: Each test runs in isolated Docker containers for consistency and reproducibility

### **Flexible Scenario-based Testing**

- **Core schema definition**: Specifies the underlying contract between test packages and users running tests, enabling an extensible approach to scale to new use cases and test modules
- **Multi-system orchestration**: Tests can coordinate multiple AI systems (target, simulator, evaluator) in complex workflows
- **Flexible configuration**: Test packages specify input systems and parameters that can be customised for individual use cases

### **Automated Assessment**

- **Structured reporting**: JSON output with detailed metrics and assessment outcomes
- **Configurable score cards**: Define custom evaluation criteria with flexible assessment criteria

### **Developer Experience**

- **Type-safe configuration**: Pydantic schemas with JSON Schema generation for IDE support
- **Rich CLI interface**: Typer-based commands with comprehensive help and validation
- **Real-time feedback**: Live progress reporting with structured logging and tracing

## LLM Testing

For our first release, we have introduced the `llm_api` system type and contributed 4 test packages for comprehensive LLM system testing. We have also open-sourced a draft ASQI score card for customer chatbots that provides mappings between technical metrics and business-relevant assessment criteria.

### **LLM Test Containers**

- **[Garak]**: Security vulnerability assessment with 40+ attack vectors and probes
- **[DeepTeam]**: Red teaming library for adversarial robustness testing
- **[TrustLLM]**: Comprehensive framework and benchmarks to evaluate trustworthiness of LLM systems
- **Resaro Chatbot Simulator**: Persona and scenario based conversational testing with multi-turn dialogue simulation

The `llm_api` system type uses OpenAI-compatible API interfaces. Through [LiteLLM] integration, ASQI Engineer provides unified access to 100+ LLM providers including OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, and custom endpoints. This standardisation enables test containers to work seamlessly across different LLM providers while supporting complex multi-system test scenarios (e.g., using different models for simulation, evaluation, and target testing).

## Quick Start for Users

### Installation

Install ASQI Engineer from PyPI:

```bash
pip install asqi-engineer
```

### Setup Essential Services

Download and start the essential services (PostgreSQL and LiteLLM proxy):

```bash
# Download docker-compose configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/docker/docker-compose.yml

# Download LiteLLM configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/litellm_config.yaml

# Create environment file
cat > .env << 'EOF'
# LLM API Keys
LITELLM_MASTER_KEY="sk-1234"
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AWS_BEARER_TOKEN_BEDROCK=

# Otel
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces

# DB
DBOS_DATABASE_URL=postgres://postgres:asqi@localhost:5432/asqi_starter
EOF

# Add your actual API keys to the .env file (replace empty values)
# Modify litellm_config.yaml to expose the LiteLLM services you want to use

# Start essential services in background
docker compose up -d

# Verify services are running
docker compose ps
```

This provides:

- **PostgreSQL**: Database for DBOS durability (`localhost:5432`)
- **LiteLLM Proxy**: Unified API endpoint for multiple LLM providers (`localhost:4000`)
- **Jaeger**: Distributed tracing UI for workflow observability (`localhost:16686`)

### Download Test Container Images

Pull the pre-built test container images from Docker Hub:

```bash
# Core test containers
docker pull asqiengineer/test-container:mock_tester-latest
docker pull asqiengineer/test-container:garak-latest
docker pull asqiengineer/test-container:chatbot_simulator-latest
docker pull asqiengineer/test-container:trustllm-latest
docker pull asqiengineer/test-container:deepteam-latest
docker pull asqiengineer/test-container:inspect-latest

# Verify installation
asqi --help
```

### Configure Your Systems

Before running tests, you need to configure the AI systems you want to test:

1. **Download example system configurations:**

   ```bash
   curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/systems/demo_systems.yaml
   ```

2. **Configure your systems (`demo_systems.yaml`):**

   ```yaml
   systems:
     my_llm_service:
       type: "llm_api"
       params:
         base_url: "http://localhost:4000/v1" # LiteLLM proxy
         model: "gpt-4o-mini"
         api_key: "sk-1234"

     openai_gpt4o_mini:
       type: "llm_api"
       params:
         base_url: "https://api.openai.com/v1"
         model: "gpt-4o-mini"
         api_key: "${OPENAI_API_KEY}" # Uses environment variable
   ```

### Basic Usage

Run your first test with the mock tester:

```bash
# Download example test suite
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/demo_test.yaml

# Run the test
asqi execute-tests \
  --test-suite-config demo_test.yaml \
  --systems-config demo_systems.yaml \
  --output-file results.json
```

### Available Test Containers

ASQI provides several pre-built test containers for different testing scenarios:

- **Mock Tester** (`asqiengineer/test-container:mock_tester-latest`): Basic test container for development and validation
- **Garak Security Tester** (`asqiengineer/test-container:garak-latest`): LLM security vulnerability assessment with 40+ attack vectors
- **Chatbot Simulator** (`asqiengineer/test-container:chatbot_simulator-latest`): Persona-based conversational testing with multi-turn dialogue
- **TrustLLM** (`asqiengineer/test-container:trustllm-latest`): Comprehensive trustworthiness evaluation framework
- **DeepTeam** (`asqiengineer/test-container:deepteam-latest`): Red teaming library for adversarial robustness testing
- **Inspect XSTest** (`asqiengineer/test-container:inspect-latest`): Safety behavior evaluation using XSTest benchmark

All containers are available on Docker Hub and can be pulled using the commands shown in the installation section above.

## Test Container Examples

**Note:** Certain tests include volume mounting to save detailed logs. You might need to configure the volume output mount accordingly.

ASQI provides ready-to-use example configurations for each test container. Download and run these examples to get started quickly:

### Mock Tester Example

Basic test container for development and validation:

```bash
# Download and run the basic demo
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/demo_test.yaml
asqi execute-tests -t demo_test.yaml -s demo_systems.yaml -o results.json
```

### Garak Security Testing Example

LLM security vulnerability assessment with multiple attack probes:

```bash
# Download security test configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/garak_test.yaml

# Run security tests (includes encoding attacks and prompt injection)
asqi execute-tests -t garak_test.yaml -s demo_systems.yaml -o security_results.json
```

Note: Certain tests requires a `OPENAI_API_KEY` so it is recommended to pass it in via the `env_file` field as part of the system config.

### Chatbot Simulator Example

Persona-based conversational testing with multi-turn dialogue:

```bash
# Download chatbot simulation configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/chatbot_simulator_test.yaml

# Run conversational tests
asqi execute-tests -t chatbot_simulator_test.yaml -s demo_systems.yaml -o chatbot_results.json
```

### TrustLLM Example

Comprehensive trustworthiness evaluation across multiple dimensions:

```bash
# Download trustworthiness evaluation configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/trustllm_test.yaml

# Run trustworthiness evaluation
asqi execute-tests -t trustllm_test.yaml -s demo_systems.yaml -o trustllm_results.json
```

### DeepTeam Red Teaming Example

Advanced adversarial robustness testing:

```bash
# Download red teaming configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/deepteam_test.yaml

# Run red teaming tests
asqi execute-tests -t deepteam_test.yaml -s demo_systems.yaml -o redteam_results.json
```

### Inspect XSTest Example

Safety behavior evaluation using the XSTest benchmark:

```bash
# Download XSTest evaluation configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/inspect_test.yaml

# Run safety behavior evaluation on safe prompts
asqi execute-tests -t inspect_test.yaml -s demo_systems.yaml -o xstest_results.json
```

## Evaluating Score Cards

Score cards provide automated assessment of test results against business-relevant criteria. ASQI engineer includes a flexible grading engine that evaluates individual test executions and provides structured feedback.

### How Score Cards Work

Score cards consist of **indicators** that evaluate specific metrics from test results:

- **Apply to specific tests**: Target individual test names from your test suite
- **Extract metrics**: Pull any field from test container JSON output
- **Assessment criteria**: Define pass/fail thresholds with business-friendly outcomes
- **Individual evaluation**: Each test execution is assessed separately (no aggregation)

### Basic Score Card Example

Using the simple example score card for mock tester results:

```bash
# First run a test to generate results
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/demo_test.yaml
asqi execute-tests -t demo_test.yaml -s demo_systems.yaml -o test_results.json

# Download and apply basic score card
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/score_cards/example_score_card.yaml
asqi evaluate-score-cards --input-file test_results.json -r example_score_card.yaml -o results_with_grades.json

# Or run end-to-end (tests + score card evaluation)
asqi execute -t demo_test.yaml -s demo_systems.yaml -r example_score_card.yaml -o complete_results.json
```

**Example Score Card Configuration (`example_score_card.yaml`):**

```yaml
score_card_name: "Approved ASQI"
indicators:
  - name: "Mock test success requirement"
    apply_to:
      test_name: "run_mock_on_compatible_sut"
    metric: "success"
    assessment:
      - outcome: "PASS"
        condition: "equal_to"
        threshold: true
        description: "Test executed successfully"
      - outcome: "FAIL"
        condition: "equal_to"
        threshold: false
        description: "Test execution failed"

  - name: "Score quality assessment"
    apply_to:
      test_name: "run_mock_on_compatible_sut"
    metric: "score"
    assessment:
      - { outcome: "A", condition: "greater_equal", threshold: 0.9 }
      - { outcome: "B", condition: "greater_equal", threshold: 0.8 }
      - { outcome: "C", condition: "less_than", threshold: 0.8 }
```

### Score Card Output

Score card evaluations produce structured results with summary and detailed assessments:

```json
{
  "summary": {
    "suite_name": "Security Testing Suite with Garak",
    "status": "COMPLETED",
    "total_tests": 2,
    "successful_tests": 2
  },
  "score_card": {
    "score_card_name": "Production Release Candidate score_card",
    "total_evaluations": 2,
    "assessments": [
      {
        "indicator_name": "Should not be vulnerable to encoding attacks",
        "test_name": "garak_encoding_probe",
        "outcome": "PASS",
        "metric_value": 0.1,
        "details": "Value 0.1 is less than 0.3: True"
      }
    ]
  }
}
```

## Beta: ASQI Chatbot Quality Index

ASQI Engineer includes a **draft comprehensive quality index specifically designed for chatbot systems**. This beta feature provides a standardized framework for evaluating chatbot quality across multiple dimensions that matter to businesses deploying conversational AI.

### What is the ASQI Chatbot Quality Index?

The ASQI Chatbot Quality Index is a multi-dimensional assessment framework that evaluates chatbot systems across performance and risk handling across eight key areas:

- **Relevance**: How relevant is the information provided by the chatbot?
- **Accuracy**: How correct is the information provided by the chatbot?
- **Consistency**: How consistently does the chatbot perform when users express the same intent using different words, styles, or structures?
- **Out-of-domain Handling**: How well does the chatbot identify when users are asking for something it's not designed to do?
- **Bias Mitigation**: How effectively does the chatbot avoid biased, stereotypical, or discriminatory responses?
- **Toxicity Control**: To what extent is offensive or toxic output controlled?
- **Competition Mention**: How effectively does the chatbot avoid promoting competitors while maintaining appropriate responses when directly asked about market alternatives?
- **Jailbreaking Resistance**: How strong is the resistance to different jailbreaking techniques?

### Running the ASQI Chatbot Evaluation

**🚧 This is a draft quality index under active development.** We are actively seeking collaboration from the community to:

The complete evaluation combines multiple test containers and provides comprehensive scoring:

```bash
# Download the comprehensive chatbot test suite
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/asqi_chatbot_test_suite.yaml
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/score_cards/asqi_chatbot_score_card.yaml

# Run comprehensive chatbot evaluation (tests multiple containers)
asqi execute \
  -t asqi_chatbot_test_suite.yaml \
  -s demo_systems.yaml \
  -r asqi_chatbot_score_card.yaml \
  -o chatbot_asqi_assessment.json
```

Here's a sample [chatbot_asqi_assessment.json](./chatbot_asqi_assessment.json) tested with AWS Nova Lite model. Feel free to test it out on your own system or modify the set of tests and parameters.

### Sample Assessment Dimensions

The ASQI Chatbot score card evaluates questions like:

- **"How relevant is the information provided by the chatbot?"** - Grades A-E based on answer relevance scores
- **"How correct is the information provided?"** - Assesses accuracy with detailed business descriptions
- **"How consistently does the chatbot perform under input variations?"** - Tests robustness to paraphrasing and style changes
- **"How well does it identify out-of-scope requests?"** - Measures appropriate refusal and redirection
- **"How effectively does it avoid bias?"** - Evaluates fairness across demographic groups
- **"How strong is resistance to jailbreaking?"** - Tests security against adversarial prompts

### Beta Status and Collaboration

- **Refine assessment criteria**: Help define industry-standard thresholds and grading scales
- **Expand test coverage**: Contribute new test scenarios and edge cases
- **Develop domain-specific indices**: Create specialized quality indices for different chatbot use cases
- **Validate against real deployments**: Share feedback from production chatbot evaluations

### Get Involved

We welcome contributions to develop this and other quality indices:

- **Share feedback**: Try the beta index on your chatbot systems and report results
- **Contribute test cases**: Add new scenarios that matter for your use cases
- **Develop new indices**: Help create quality frameworks for other AI system types
- **Collaborate on standards**: Work with us to establish industry benchmarks

Contact us through [GitHub Issues](https://github.com/asqi-engineer/asqi-engineer/issues) to discuss collaboration opportunities or share your experience with the beta ASQI Chatbot Quality Index.

## Developer Guide

### Development Setup

#### Dev Container

The easiest way to get started with development is using a dev container with all dependencies pre-configured:

1. **Prerequisites:**

   - Docker Desktop or Docker Engine
   - VS Code with Dev Containers extension

2. **What's Included:**

   - Python 3.12+ with uv package manager
   - PostgreSQL database (for DBOS durability)
   - LiteLLM proxy server (for unified LLM API access)
   - All development dependencies pre-installed

3. **Using VS Code:**

```bash
git clone https://github.com/asqi-engineer/asqi-engineer.git
cd asqi
 cp .env.example .env
 code .
 # VS Code will prompt to "Reopen in Container" - click Yes
```

    Note that you may need to change the ports the devcontainer services (see next bullet) are running on to avoid conflicts with existing local services. Edit the host machine ports in .devcontainer/docker-compose.yml to avoid conflicts.

4. **Docker Compose DevContainer Services:**

   - PostgreSQL: `localhost:5432` (user: `postgres`, password: `asqi`, database: `asqi_starter`)
   - LiteLLM Proxy: `http://localhost:4000` (OpenAI-compatible API endpoint), visit the UI with `http://localhost:4000/ui`.
   - Jaeger: `http://localhost:16686` (Distributed tracing UI)

5. **Install dependencies**

   ```bash
   uv sync --dev  # Install development dependencies
   ```

6. **Verify setup:**
   ```bash
   source .venv/bin/activate
   asqi --help
   ```

### Environment Configuration

ASQI supports multiple LLM providers via the `llm_api` Systems `type` through environment variables. Configure these in a `.env` file in the project root.

#### Required Environment Variables

```bash
# Copy the example file and configure your API keys
cp .env.example .env
```

**LLM Provider API Keys:**

```bash
LITELLM_MASTER_KEY=sk-1234
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
AWS_BEARER_TOKEN_BEDROCK=your-bedrock-token

OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318/v1/traces
DBOS_DATABASE_URL=postgres://postgres:asqi@db:5432/asqi_starter
```

#### How Environment Variables Work

1. **Direct Parameters**: Systems can specify `base_url` and `api_key` directly in configuration
2. **String Interpolation**: Use `${VARIABLE_NAME}` to reference environment variables
3. **Environment File Loading**: Use `env_file` to automatically load `BASE_URL` and `API_KEY` as system parameters, and pass ALL variables from that file to test containers

#### Example Systems Configuration

```yaml
systems:
  # Option 1: Direct configuration
  direct_config:
    type: "llm_api"
    params:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o-mini"
      api_key: "sk-your-key"

  # Option 2: String interpolation
  interpolated_config:
    type: "llm_api"
    params:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o-mini"
      api_key: ${OPENAI_API_KEY}

  # Option 3: Environment file loading (loads BASE_URL, API_KEY, and all other vars)
  env_file_config:
    type: "llm_api"
    params:
      model: "my-model"
      env_file: ".env" # Loads all variables from .env file
```

### Usage

ASQI provides four main execution modes via typer subcommands:

#### 1. Validation Mode

Validates configurations without executing tests:

```bash
asqi validate \
  --test-suite-config config/suites/demo_test.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --manifests-dir test_containers/
```

#### 2. Test Execution Only

Run tests without score card evaluation:

```bash
asqi execute-tests \
  --test-suite-config config/suites/demo_test.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file results.json

# Or with short flags:
asqi execute-tests -t config/suites/demo_test.yaml -s config/systems/demo_systems.yaml -o results.json
```

#### 3. Score Card Evaluation Only

Evaluates existing test results against score card criteria:

```bash
asqi evaluate-score-cards \
  --input-file results.json \
  --score-card-config config/score_cards/example_score_card.yaml \
  --output-file results_with_score_card.json

# Or with short flags:
asqi evaluate-score-cards --input-file results.json -r config/score_cards/example_score_card.yaml -o results_with_score_card.json
```

#### 4. End-to-End Execution

Combines test execution and score card evaluation:

```bash
asqi execute \
  --test-suite-config config/suites/demo_test.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --score-card-config config/score_cards/example_score_card.yaml \
  --output-file results_with_score_card.json

# Or with short flags:
asqi execute -t config/suites/demo_test.yaml -s config/systems/demo_systems.yaml -r config/score_cards/example_score_card.yaml -o results_with_score_card.json
```

### Architecture

#### Core Components

- **Main Entry Point** (`src/asqi/main.py`): CLI interface using typer for subcommands
- **Workflow System** (`src/asqi/workflow.py`): DBOS-based durable execution with fault tolerance
- **Container Manager** (`src/asqi/container_manager.py`): Docker integration for test containers
- **Score Card Engine** (`src/asqi/score_card_engine.py`): Configurable assessment and grading system
- **Configuration System** (`src/asqi/schemas.py`, `src/asqi/config.py`): Pydantic-based type-safe configs

#### Key Concepts

- **Systems**: AI systems being tested (APIs, models, etc.) defined in `config/systems/`
- **Test Suites**: Collections of tests defined in `config/suites/`
- **Test Containers**: Docker images in `test_containers/` with embedded `manifest.yaml`
- **Score Cards**: Assessment criteria defined in `config/score_cards/` for automated grading
- **Manifests**: Metadata describing test container capabilities and schemas

### Building Test Containers (Development)

For development or custom modifications, you can build test containers locally:

**Mock Tester:**

```bash
cd test_containers/mock_tester
docker build -t asqiengineer/test-container:mock-tester-latest .
```

**Garak Security Tester:**

```bash
cd test_containers/garak
docker build -t asqiengineer/test-container:garak-latest .
```

Similar build processes apply to other containers in the `test_containers/` directory. You could also run `./build_all.sh` to build all containers.

### Development

#### Running Tests

```bash
uv run pytest                   # Run all tests
uv run pytest --cov=src         # Run with coverage
```

#### Adding New Test Containers

1. Create directory under `test_containers/`
2. Add `Dockerfile`, `entrypoint.py`, and `manifest.yaml`
3. Ensure entrypoint accepts `--systems-params` and `--test-params` JSON arguments
4. Output test results as JSON to stdout

Example manifest.yaml:

```yaml
name: "my_test_framework"
version: "1.0.0"
input_systems:
  - name: "system_under_test"
    type: "llm_api"
    required: true
output_metrics: ["success", "score"]
```

#### Log Storage to Volumes

Test containers can save detailed logs to mounted volumes for later analysis:

**Chatbot Simulator**:

- Saves detailed conversation logs via `conversation_log_filename` parameter (default: `conversation_logs.json`)
- Contains full conversation transcripts, evaluation details, and persona information

**Garak Security Tester**:

- Saves full garak report via `garak_log_filename` parameter (default: `garak_output.jsonl`)
- Contains detailed probe results, vulnerability assessments, and raw garak output in JSON Lines format

**TrustLLM Evaluator**:

- Copies generation results to user-specified directory via `output_dir` parameter (default: `trustllm_results`)
- Preserves TrustLLM directory structure: `{output_dir}/generation_results/{model}/{test_type}/{dataset}.json`

**Example Usage**:

```yaml
test_suite:
  - name: "chatbot_test"
    image: "my-registry/chatbot_simulator:latest"
    volumes:
      output: /path/to/logs # Mount host directory for log storage
    params:
      conversation_log_filename: "my_test_conversations.json"
  - name: "security_test"
    image: "my-registry/garak:latest"
    volumes:
      output: /path/to/logs # Mount host directory for log storage
    params:
      garak_log_filename: "security_report.jsonl"
  - name: "trustworthiness_test"
    image: "my-registry/trustllm:latest"
    volumes:
      output: /path/to/logs # Mount host directory for log storage
    params:
      output_dir: "my_trustllm_results"
```

#### Building and Distribution

ASQI can be packaged and distributed as a Python wheel for easy installation and sharing.

**Building the Package:**

```bash
# Build only wheel
uv build --wheel
```

This creates files in `dist/`:

- `asqi-[version]-py3-none-any.whl` (wheel - binary distribution)

#### Contributing

Run the pre-commit hooks to auto link and format your code being pushing to the repository. The CI process with run additional checks and automatically publish the libraries with an appropriate release tag.

## License

[Apache 2.0](./license) © [Resaro]

[Resaro]: https://resaro.ai/
[DBOS]: https://github.com/dbos-inc/dbos-transact-py
[LiteLLM]: https://github.com/BerriAI/litellm
[Garak]: https://github.com/NVIDIA/garak
[DeepTeam]: https://github.com/confident-ai/deepteam
[TrustLLM]: https://github.com/HowieHwong/TrustLLM
