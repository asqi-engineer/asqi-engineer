# ASQI - AI Systems Quality Index

ASQI (AI Systems Quality Index) executor is a test executor for AI systems using containerized test frameworks. It provides a comprehensive platform for running quality assessments, security tests, and performance evaluations against AI systems with configurable score cards and automated reporting.

## Quick Start

### Option 1: Dev Container (Recommended)

The easiest way to get started is using a dev container with all dependencies pre-configured:

1. **Prerequisites:**
   - Docker Desktop or Docker Engine
   - VS Code with Dev Containers extension

2. **Using VS Code:**
   ```bash
   git clone <repository-url>
   cd asqi
   code .
   # VS Code will prompt to "Reopen in Container" - click Yes
   ```

4. **Verify setup:**
   ```bash
   uv run python -m asqi.main --help
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
   uv run python -m asqi.main --help
   ```


## Usage

ASQI provides four main execution modes via typer subcommands:

### 1. Validation Mode
Validates configurations without executing tests:
```bash
uv run python -m asqi.main validate \
  --suite-file config/suites/demo_suite.yaml \
  --suts-file config/suts/demo_suts.yaml \
  --manifests-dir test_containers/
```

### 2. Test Execution Only
Runs tests without score card evaluation:
```bash
uv run python -m asqi.main execute-tests \
  --suite-file config/suites/demo_suite.yaml \
  --suts-file config/suts/demo_suts.yaml \
  --output-file results.json
```

### 3. Score Card Evaluation Only
Evaluates existing test results against score card criteria:
```bash
uv run python -m asqi.main evaluate-score-cards \
  --input-file results.json \
  --score-card-file config/score_cards/example_score_card.yaml \
  --output-file results_with_score_card.json
```

### 4. End-to-End Execution
Combines test execution and score card evaluation:
```bash
uv run python -m asqi.main execute \
  --suite-file config/suites/demo_suite.yaml \
  --suts-file config/suts/demo_suts.yaml \
  --score-card-file config/score_cards/example_score_card.yaml \
  --output-file results_with_score_card.json
```

## Architecture

### Core Components

- **Main Entry Point** (`src/asqi/main.py`): CLI interface using typer for subcommands
- **Workflow System** (`src/asqi/workflow.py`): DBOS-based durable execution with fault tolerance
- **Container Manager** (`src/asqi/container_manager.py`): Docker integration for test containers
- **Score Card Engine** (`src/asqi/score_card_engine.py`): Configurable assessment and grading system
- **Configuration System** (`src/asqi/schemas.py`, `src/asqi/config.py`): Pydantic-based type-safe configs

### Key Concepts

- **SUTs (Systems Under Test)**: AI systems being tested (APIs, models, etc.) defined in `config/suts/`
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
uv run python -m asqi.main execute-tests \
  --suite-file config/suites/security_test.yaml \
  --suts-file config/suts/demo_suts.yaml \
  --output-file garak_results.json
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
3. Ensure entrypoint accepts `--sut-config` and `--test-params` JSON arguments
4. Output test results as JSON to stdout

Example manifest.yaml:
```yaml
name: "my_test_framework"
version: "1.0.0"
image_name: "my-registry/my_test:latest"
supported_suts:
  - type: "llm_api"
    required_config: ["provider", "model"]
output_metrics: ["success", "score"]
```

## Contributing

1. Install development dependencies: `uv sync --dev`
2. Run tests: `uv run pytest`
3. Check code quality: `uv run ruff check && uv run ruff format`
4. Run security scan: `uv run bandit -r src/`

## License

TODO
