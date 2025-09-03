# Quick Start

Get up and running with ASQI Engineer in minutes.

## Installation

Install ASQI Engineer from PyPI:

```bash
pip install asqi-engineer
```

### Setup Essential Services

Download and start the essential services (PostgreSQL and LiteLLM proxy):

```bash
# Download docker-compose configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/docker-compose.yaml

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
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318/v1/traces

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

# Verify installation
asqi --help
```

### Configure Your Systems

Before running tests, you need to configure the AI systems you want to test:

1. **Download example configurations:**
   ```bash
   curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/systems/demo_systems.yaml
   curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/.env.example
   ```

2. **Setup environment variables:**
   ```bash
   # Copy and configure your API keys
   cp .env.example .env
   # Edit .env file with your actual API keys
   ```

3. **Configure your systems (`demo_systems.yaml`):**
   ```yaml
   systems:
     my_llm_service:
       type: "llm_api"
       params:
         base_url: "http://localhost:4000/v1"  # LiteLLM proxy
         model: "gpt-4o-mini"
         api_key: "sk-1234"
     
     openai_gpt4o_mini:
       type: "llm_api"
       params:
         base_url: "https://api.openai.com/v1"
         model: "gpt-4o-mini"
         api_key: "${OPENAI_API_KEY}"  # Uses environment variable
   ```

## Basic Usage

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

## Evaluate with Score Cards

Score cards provide automated assessment of test results against business-relevant criteria:

```bash
# Download and apply basic score card
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/score_cards/example_score_card.yaml
asqi evaluate-score-cards \
  --input-file results.json \
  --score-card-config example_score_card.yaml \
  --output-file results_with_grades.json

# Or run end-to-end (tests + score card evaluation)
asqi execute \
  --test-suite-config demo_test.yaml \
  --systems-config demo_systems.yaml \
  --score-card-config example_score_card.yaml \
  --output-file complete_results.json
```

## Next Steps

- Review [Available Test Containers](test-containers.md) to see all testing frameworks and examples
- Explore [Configuration](configuration.md) to understand how to configure systems, test suites, and score cards
- Check out [Examples](examples.md) for advanced usage scenarios
- See [CLI Reference](cli.rst) for complete command documentation