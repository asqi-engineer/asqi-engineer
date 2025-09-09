# ASQI Engineer

![ASQI Engineer](./docs/asqi-engineer-cover.png)

ASQI (AI Solutions Quality Index) Engineer helps teams test and evaluate AI systems. It runs containerized test packages, automates scoring, and provides durable execution workflows.

The project focuses first on chatbot testing and supports extensions for other AI system types. [Resaro][Resaro] welcomes contributions of test packages, score cards, and schemas.

## Table of Contents

- [LLM Testing](#llm-testing)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Key Highlights](#key-highlights)
- [Contributing & development](#contributing--development)
- [License](#license)

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

Install the package:

```bash
pip install asqi-engineer
```

Prepare a minimal `.env` with your LLM key(s) and DB URL:

```bash
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
```

Download the compose and LiteLLM config and start essential services:

```bash
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/docker/docker-compose.yml
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/litellm_config.yaml
docker compose up -d
docker compose ps
```

Run a quick demo using the example configs already in this repo:

```bash
asqi execute-tests \
  --test-suite-config config/suites/demo_test.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file demo_results.json
```

This short flow gets you to a runnable demo. For full setup, advanced examples, and provider-specific notes see the docs: https://www.asqi.ai/quickstart.html

## Documentation

Detailed documentation lives on the project docs site — use the links below to jump to the full guides and examples:

- Quickstart (installation & environment): https://www.asqi.ai/quickstart.html
- CLI & usage reference: https://www.asqi.ai/cli.html
- Configuration & environment variables: https://www.asqi.ai/configuration.html
- Test container examples & how-to: https://www.asqi.ai/examples.html
- LLM test containers overview (Garak, DeepTeam, TrustLLM, Chatbot Simulator): https://www.asqi.ai/llm-test-containers.html
- Score cards & evaluation: https://www.asqi.ai/examples.html#score-cards
- Developer guide & architecture: https://www.asqi.ai/architecture.html
- Creating custom test containers: https://www.asqi.ai/custom-test-containers.html

If a link is missing or the page content is unclear, please open an issue: https://github.com/asqi-engineer/asqi-engineer/issues

## Key Highlights

- Durable, DBOS-backed execution with retries and recovery
- Containerized test packages for isolation and reproducibility
- Extensible test-suite and score-card model for automated assessment
- Pydantic-based schemas and rich CLI (Typer) for developer ergonomics

## Contributing & development

We keep contributor-facing documentation split into two focused documents so each file stays concise and actionable.

Quick actions:

- To see how to contribute (PR process, templates, commit guidance), open [CONTRIBUTION.md].
- To get your dev environment ready and run tests locally (venv, `uv` commands, and devcontainer), open [DEVELOPMENT.md].
- Example configs and test containers live under `config/` and `test_containers/` respectively.

If you're unsure where to start, read [CONTRIBUTION.md] first for the workflow and then follow the setup steps in [DEVELOPMENT.md] to run the test suite locally.

## License

[Apache 2.0](./license) © [Resaro]

[Resaro]: https://resaro.ai/
[DBOS]: https://github.com/dbos-inc/dbos-transact-py
[LiteLLM]: https://github.com/BerriAI/litellm
[Garak]: https://github.com/NVIDIA/garak
[DeepTeam]: https://github.com/confident-ai/deepteam
[TrustLLM]: https://github.com/HowieHwong/TrustLLM
[CONTRIBUTION.md]: ./CONTRIBUTION.md
[DEVELOPMENT.md]: ./DEVELOPMENT.md
