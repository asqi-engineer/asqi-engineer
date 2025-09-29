## Developer guide (authoritative setup & commands)

This document is the canonical developer setup reference: devcontainer, environment variables, how to install dev deps, run tests, and build test containers. For contribution workflow (PRs, issue templates), see [CONTRIBUTING.md]. For a high-level project overview and quickstart, see [README.md].

### Prerequisites

- Docker Desktop or Docker Engine
- VS Code (optional) with Dev Containers extension if you plan to use the devcontainer
- Python 3.12+ (the devcontainer already includes an appropriate runtime)

### Quick start (local)

1. Clone the repo and copy the example env:

```bash
git clone https://github.com/asqi-engineer/asqi-engineer.git
cd asqi-engineer
cp .env.example .env
```

2. Install development dependencies and create the virtualenv (uses `uv`):

```bash
uv sync --dev
source .venv/bin/activate
```

3. Verify the CLI help is available:

```bash
asqi --help
```

If you use VS Code, opening the repository will prompt to reopen in the devcontainer. The devcontainer includes PostgreSQL, LiteLLM proxy, and tracing services for a near-production environment.

### Environment variables

Configure providers and local services by editing `.env`. Typical keys:

```
LITELLM_MASTER_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AWS_BEARER_TOKEN_BEDROCK=

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
DBOS_DATABASE_URL=postgres://postgres:asqi@localhost:5432/asqi_starter
```

Tip: `env_file` support in system configs lets you load variables from `.env` into container runtimes.

### Running tests

Run the full test suite locally with:

```bash
uv run pytest
```

Run a focused run during development:

```bash
uv run pytest tests/test_container_manager.py -q
```

Run with coverage:

```bash
uv run pytest --cov=src
```

If tests fail after your changes, iterate locally until green before opening a PR (see [CONTRIBUTING.md] for PR expectations).

### CLI usage examples

Validation only:

```bash
asqi validate \
  -t config/suites/demo_test.yaml \
  -s config/systems/demo_systems.yaml \
  --manifests-dir test_containers/
```

Run tests (execution only):

```bash
asqi execute-tests \
  -t config/suites/demo_test.yaml \
  -s config/systems/demo_systems.yaml \
  -o output.json
```

End-to-end (tests + score card evaluation):

```bash
asqi execute \
  -t config/suites/demo_test.yaml \
  -s config/systems/demo_systems.yaml \
  -r config/score_cards/example_score_card.yaml \
  -o results_with_score_card.json
```

### Building test containers locally

From the repo root you can build an example container locally for development:

```bash
cd test_containers/mock_tester
docker build -t asqiengineer/test-container:mock-tester-latest .

# or build all containers (Linux/macOS):
./test_containers/build_all.sh
```

### Logs and volumes

When running test containers, mount a host directory for persistent logs. Example in a test-suite YAML:

```yaml
test_suite:
  - name: "chatbot_test"
    image: "my-registry/chatbot_simulator:latest"
    volumes:
      output: /path/to/logs
    params:
      conversation_log_filename: "my_test_conversations.json"
```

### Packaging

Build a wheel locally:

```bash
uv build --wheel
```

### Troubleshooting

- If the devcontainer fails to start, check `.devcontainer/docker-compose.yml` for port conflicts.
- If tests error due to missing environment variables, confirm `.env` contains required keys or that system configs use `env_file`.

### Next steps

- For PR and contribution process details, see [CONTRIBUTING.md].
- For a project overview and quickstart, see [README.md].

Thank you for improving ASQI Engineer — open a PR when ready and link the relevant issue.

[CONTRIBUTING.md]: ./CONTRIBUTING.md
[README.md]: ./README.md
