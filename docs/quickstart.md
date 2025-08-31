# Quick Start

Get up and running with ASQI Engineer in minutes using our pre-configured development environment or local installation.

## Option 1: Dev Container (Recommended)

The easiest way to get started is using a dev container with all dependencies pre-configured.

### Prerequisites
- Docker Desktop or Docker Engine
- VS Code with Dev Containers extension

### What's Included
- Python 3.12+ with uv package manager
- PostgreSQL database (for DBOS durability)
- LiteLLM proxy server (for unified LLM API access)
- All development dependencies pre-installed

### Setup Steps

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd asqi
   cp .env.example .env
   code .
   # VS Code will prompt to "Reopen in Container" - click Yes
   ```

2. **Configure environment variables:**
   Edit `.env` with your API keys:
   ```bash
   OPENAI_API_KEY=sk-your-openai-key
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
   LITELLM_MASTER_KEY=sk-1234
   ```

3. **Verify setup:**
   ```bash
   asqi --help
   ```

### DevContainer Services

The dev container automatically starts these services:

- **PostgreSQL**: `localhost:5432` (user: `postgres`, password: `asqi`, database: `asqi_starter`)
- **LiteLLM Proxy**: `http://localhost:4000` (OpenAI-compatible API endpoint)
  - Management UI: `http://localhost:4000/ui`
- **Jaeger**: `http://localhost:16686` (Distributed tracing UI)

```{note}
You may need to change the ports in `.devcontainer/docker-compose.yml` to avoid conflicts with existing local services.
```

## Option 2: Local Development

If you prefer local development without containers:

### Prerequisites
- Python 3.12+
- Docker (for running test containers)
- uv (Python package manager)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd asqi
   uv sync --dev  # Install dependencies including dev tools
   ```

2. **Setup PostgreSQL for DBOS:**
   See `.devcontainer/docker-compose.yaml` for example configuration, or use your existing PostgreSQL instance.

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Verify installation:**
   ```bash
   asqi --help
   ```

## First Test Run

Let's run a simple test to verify everything works:

### 1. Build the Mock Test Container
```bash
cd test_containers/mock_tester
docker build -t my-registry/mock_tester:latest .
cd ../..
```

### 2. Run a Validation
```bash
asqi validate \
  --test-suite-config config/suites/demo_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --manifests-dir test_containers/
```

### 3. Execute Tests
```bash
asqi execute-tests \
  --test-suite-config config/suites/demo_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file my_first_results.json
```

### 4. Evaluate with Score Cards
```bash
asqi evaluate-score-cards \
  --input-file my_first_results.json \
  --score-card-config config/score_cards/example_score_card.yaml \
  --output-file my_first_results_graded.json
```

## Next Steps

- Explore [Configuration](configuration.md) to understand how to configure systems and test suites
- Review [Test Containers](test-containers.md) to see available testing frameworks
- Check out [Examples](examples.md) for practical usage scenarios
- See [CLI Reference](cli.rst) for complete command documentation