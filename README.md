# ASQI Engineer

**ASQI (AI Systems Quality Index) Engineer** is a comprehensive framework for systematic testing and quality assurance of AI systems. Developed from [Resaro's][Resaro] experience bridging governance, technical and business requirements, ASQI Engineer enables rigorous evaluation of AI systems through containerized test packages, automated assessment, and durable execution workflows.

ASQI Engineer is in active development and we welcome contributors to contribute new test packages, share score cards and test plans, and help define common schemas to meet industry needs. Our initial release focuses on comprehensive chatbot testing with extensible foundations for broader AI system evaluation.

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

## Quick Start

### Option 1: Dev Container (Recommended)

The easiest way to get started is using a dev container with all dependencies pre-configured:

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
   git clone <repository-url>
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

5. **Verify setup:**
   ```bash
   asqi --help
   ```

### Option 2: Local Development

If you prefer local development:

**Prerequisites:**
- Python 3.12+
- Docker (for running test containers)
- uv (Python package manager)

**Installation:**
1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd asqi
   uv sync --dev  # Install dependencies including dev tools
   ```

2. Setup Postgres for DBOS. See `.devcontainer/docker-compose.yaml` for example configuration.

2. **Verify installation:**
   ```bash
   # source ./.venv/bin/activate
   asqi --help
   ```

## Environment Configuration

ASQI supports multiple LLM providers via the `llm_api` Systems `type` through environment variables. Configure these in a `.env` file in the project root.

### Required Environment Variables

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

BASE_URL=http://localhost:4000
API_KEY=sk-1234
```

### How Environment Variables Work

1. **Systems Configuration**: Systems can specify `base_url` and optionally reference an `env_file` for API keys
2. **Environment Fallbacks**: If not specified, ASQI uses `BASE_URL` and `API_KEY` from `.env`
3. **Provider Keys**: Specific provider keys (e.g., `OPENAI_API_KEY`) are passed to test containers

### Example Systems Configuration

```yaml
systems:
  # Recommended: Uses env_file for API key security
  direct_openai:
    type: "llm_api"
    params:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o-mini"
      env_file: ".env"  # References OPENAI_API_KEY from .env file
```

## Usage

ASQI provides four main execution modes via typer subcommands:

### 1. Validation Mode
Validates configurations without executing tests:
```bash
asqi validate \
  --test-suite-config config/suites/demo_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --manifests-dir test_containers/
```

### 2. Test Execution Only
First, build the required test container:
```bash
cd test_containers/mock_tester
docker build -t my-registry/mock_tester:latest .
cd ../..
```

Then run tests without score card evaluation:
```bash
asqi execute-tests \
  --test-suite-config config/suites/demo_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file results.json

# Or with short flags:
asqi execute-tests -t config/suites/demo_suite.yaml -s config/systems/demo_systems.yaml -o results.json
```

### 3. Score Card Evaluation Only
Evaluates existing test results against score card criteria:
```bash
asqi evaluate-score-cards \
  --input-file results.json \
  --score-card-config config/score_cards/example_score_card.yaml \
  --output-file results_with_score_card.json

# Or with short flags:
asqi evaluate-score-cards --input-file results.json -r config/score_cards/example_score_card.yaml -o results_with_score_card.json
```

### 4. End-to-End Execution
Combines test execution and score card evaluation:
```bash
asqi execute \
  --test-suite-config config/suites/demo_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --score-card-config config/score_cards/example_score_card.yaml \
  --output-file results_with_score_card.json

# Or with short flags:
asqi execute -t config/suites/demo_suite.yaml -s config/systems/demo_systems.yaml -r config/score_cards/example_score_card.yaml -o results_with_score_card.json
```

## Architecture

### Core Components

- **Main Entry Point** (`src/asqi/main.py`): CLI interface using typer for subcommands
- **Workflow System** (`src/asqi/workflow.py`): DBOS-based durable execution with fault tolerance
- **Container Manager** (`src/asqi/container_manager.py`): Docker integration for test containers
- **Score Card Engine** (`src/asqi/score_card_engine.py`): Configurable assessment and grading system
- **Configuration System** (`src/asqi/schemas.py`, `src/asqi/config.py`): Pydantic-based type-safe configs

### Key Concepts

- **Systems**: AI systems being tested (APIs, models, etc.) defined in `config/systems/`
- **Test Suites**: Collections of tests defined in `config/suites/`
- **Test Containers**: Docker images in `test_containers/` with embedded `manifest.yaml` 
- **Score Cards**: Assessment criteria defined in `config/score_cards/` for automated grading
- **Manifests**: Metadata describing test container capabilities and schemas

## Available Test Containers

### Mock Tester
Basic test container for development and validation:
```bash
cd test_containers/mock_tester
docker build -t my-registry/mock_tester:latest .
```

### Garak Security Tester
Real-world LLM security testing:
```bash
# Requires API keys for target LLM services
export OPENAI_API_KEY="your_api_key_here"
cd test_containers/garak
docker build -t my-registry/garak:latest .

# Run security tests
asqi execute-tests \
  --test-suite-config config/suites/security_test.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file garak_results.json

# Or with short flags:
asqi execute-tests -t config/suites/security_test.yaml -s config/systems/demo_systems.yaml -o garak_results.json
```

## Score Cards

ASQI includes a simple grading engine for automated test result evaluation:

```yaml
score_card_name: "Example Assessment"
indicators:
  - name: "Test success requirement"
    apply_to:
      test_name: "run_mock_on_compatible_sut"
    metric: "success"
    assessment:
      - { outcome: "PASS", condition: "equal_to", threshold: true }
      - { outcome: "FAIL", condition: "equal_to", threshold: false }
```

## Development

### Running Tests
```bash
uv run pytest                    # Run all tests
uv run pytest --cov=src         # Run with coverage
```

### Adding New Test Containers

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

## Building and Distribution

ASQI can be packaged and distributed as a Python wheel for easy installation and sharing.

### Building the Package

```bash
# Build only wheel
uv build --wheel
```

This creates files in `dist/`:
- `asqi-[version]-py3-none-any.whl` (wheel - binary distribution)


#### CLI Entry Point
The `asqi` command maps to `src/asqi/main.py` and provides all functionality:
```bash
asqi execute --test-suite-config config/suites/demo_suite.yaml --systems-config config/systems/demo_systems.yaml
```

## Contributing

1. Install development dependencies: `uv sync --dev`
2. Run tests: `uv run pytest`
3. Check code quality: `uv run ruff check && uv run ruff format`
4. Run security scan: `uv run bandit -r src/`

## License

TODO

[Resaro]: https://resaro.ai/
[DBOS]: https://github.com/dbos-inc/dbos-transact-py
[LiteLLM]: https://github.com/BerriAI/litellm
[Garak]: https://github.com/NVIDIA/garak
[DeepTeam]: https://github.com/confident-ai/deepteam
[TrustLLM]: https://github.com/HowieHwong/TrustLLM
