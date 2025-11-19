# ASQI Engineer

![ASQI Engineer](https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/docs/asqi-engineer-cover.png)

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
- **Configurable score cards**: Define custom evaluation criteria with flexible assessment conditions
- **Metric expressions**: Combine multiple metrics using mathematical operations (`+`, `-`, `*`, `/`) and functions (`min`, `max`, `avg`, `sum`, `abs`, `round`, `pow`) for sophisticated composite scoring

### **Developer Experience**

- **Type-safe configuration**: Pydantic schemas with JSON Schema generation for IDE support
- **Rich CLI interface**: Typer-based commands with comprehensive help and validation
- **Real-time feedback**: Live progress reporting with structured logging and tracing

## LLM Testing

For our first release, we have introduced the `llm_api` system type and contributed six test packages for comprehensive LLM system testing. We have also open-sourced a draft ASQI score card for customer chatbots that provides mappings between technical metrics and business-relevant assessment criteria.

### **LLM Test Containers**

- **[Garak]**: Security vulnerability assessment with 40+ attack vectors and probes
- **[DeepTeam]**: Red teaming library for adversarial robustness testing
- **[TrustLLM]**: Comprehensive framework and benchmarks to evaluate trustworthiness of LLM systems
- **[Inspect Evals]**: Comprehensive evaluation suite with 80+ tasks across cybersecurity, mathematics, reasoning, knowledge, bias, and safety domains
- **[LLMPerf](https://github.com/ray-project/llmperf)**: Token-level performance benchmarking for latency, throughput, and request metrics
- **Resaro Chatbot Simulator**: Persona and scenario based conversational testing with multi-turn dialogue simulation

The `llm_api` system type uses OpenAI-compatible API interfaces. Through [LiteLLM] integration, ASQI Engineer provides unified access to 100+ LLM providers including OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, and custom endpoints. This standardisation enables test containers to work seamlessly across different LLM providers while supporting complex multi-system test scenarios (e.g., using different models for simulation, evaluation, and target testing).

## Quick Start

Get started with ASQI Engineer in 3 simple steps:

### Requirements

- **Python 3.12+** is required
- Docker for running test containers
    > **Note:** If you are facing issues detecting your Docker daemon, you might need to set the `DOCKER_HOST` environment variable in your `.env` file. See `.env` for details.

**1. Install the package:**

```bash
pip install asqi-engineer
```

**2. Run the setup script:**

```bash
curl -sSL https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/setup.sh | bash
```

This downloads all required configuration files and creates a `.env` template.

**3. Configure and run:**

```bash
# Start the services and run your first test:
docker compose up -d
asqi execute-tests -t config/suites/demo_test.yaml -s config/systems/demo_systems.yaml
```

This short flow should download a demo test container and generate the test results in `output.json`. Now, to actually test your AI system, configure the `.env` file and try out the other test packages in: https://www.asqi.ai/quickstart.html


## Documentation

Detailed documentation lives on the project docs site — use the links below to jump to the full guides and examples:

- Quickstart (installation & environment): https://www.asqi.ai/quickstart.html
- Library usage & workflow customization: [docs/library.md](./docs/library.md)
- CLI & usage reference: https://www.asqi.ai/cli.html
- Configuration & environment variables: https://www.asqi.ai/configuration.html
- Test container examples & how-to: https://www.asqi.ai/examples.html
- LLM test containers overview (Garak, DeepTeam, TrustLLM, Inspect Evals, LLMPerf, Chatbot Simulator): https://www.asqi.ai/llm-test-containers.html
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

- To see how to contribute (PR process, templates, commit guidance), open [CONTRIBUTING.md].
- To get your dev environment ready and run tests locally (venv, `uv` commands, and devcontainer), open [DEVELOPMENT.md].
- Example configs and test containers live under `config/` and `test_containers/` respectively.

If you're unsure where to start, read [CONTRIBUTING.md] first for the workflow and then follow the setup steps in [DEVELOPMENT.md] to run the test suite locally.

## License

[Apache 2.0](./license) © [Resaro]

[Resaro]: https://resaro.ai/
[DBOS]: https://github.com/dbos-inc/dbos-transact-py
[LiteLLM]: https://github.com/BerriAI/litellm
[Garak]: https://github.com/NVIDIA/garak
[DeepTeam]: https://github.com/confident-ai/deepteam
[TrustLLM]: https://github.com/HowieHwong/TrustLLM
[Inspect Evals]: https://github.com/UKGovernmentBEIS/inspect_evals
[CONTRIBUTING.md]: ./CONTRIBUTING.md
[DEVELOPMENT.md]: ./DEVELOPMENT.md
