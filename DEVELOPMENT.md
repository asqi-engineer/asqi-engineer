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
  - id: "chatbot_test"
    name: "chatbot test"
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

### Writing tests for a test container

Every container in `test_containers/` has a `tests/` directory for pytest-based tests.
The shared `asqi.pytest_plugin` (in `src/asqi/pytest_plugin.py`) provides fixtures and
marks that remove boilerplate. The plugin is registered via the `pytest11` entry point
in `asqi-engineer`'s `pyproject.toml`; when the package is installed it is auto-discovered
by pytest.

Because container lockfiles currently pin a PyPI release of `asqi-engineer` that
predates the plugin, each container's rootdir `conftest.py` also contains a sys.path
bootstrap that makes the local source importable. This bootstrap will be removed once
containers upgrade to the version that ships the plugin.

#### Available fixtures

| Fixture | What it does |
|---------|-------------|
| `container_root` | `Path` to the container's root directory (parent of `tests/`). Derived automatically from the test file's location. |
| `asqi_execute` | High-level runner. Accepts the same config file arguments as `asqi execute` CLI, resolves YAML → param dicts, sets up mount dirs, imports `entrypoint.main` in-process, and returns `(ContainerOutput, exit_code)`. |

#### Available marks

| Mark | When to use |
|------|-------------|
| `@pytest.mark.requires_hf_token` | Test needs a HuggingFace dataset. Skips when `HF_TOKEN` is not set. |
| `@pytest.mark.requires_llm_api` | Test calls a real LLM endpoint. Skips when no API key env var is present. |

#### Container conftest pattern

Each container has a `conftest.py` at its **root** (next to `entrypoint.py`). The
lazy import of `entrypoint` inside `run_container` is required — importing it at
module level would pull in `asqi.*` before the sys.path bootstrap runs:

```python
# <container_root>/conftest.py

import sys
from pathlib import Path

_asqi_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_asqi_src) not in sys.path:
    sys.path.insert(0, str(_asqi_src))

pytest_plugins = ["asqi.pytest_plugin"]


@pytest.fixture
def run_container(mocker, valid_systems_params, valid_test_params):
    from entrypoint import main  # lazy — must be inside fixture body
    ...
```

#### Using `asqi_execute` (config-based tests)

`asqi_execute` resolves YAML configs the same way the `asqi execute` CLI does.
Container-specific API mocks are still applied via `mocker.patch` before calling it.
Decorate with `@pytest.mark.requires_hf_token` if the suite references a private dataset.

```python
@pytest.mark.requires_hf_token
def test_scores_within_range(mock_full_run, api_success, asqi_execute):
    mock_full_run(dataset_rows=[...], api_side_effect=api_success())
    results = asqi_execute(
        test_suite_config="config/suite.yaml",
        systems_config="config/systems.yaml",
        datasets_config="config/datasets.yaml",  # optional
    )
    output, exit_code = results[0]
    assert exit_code == 0
    assert output.results is not None
```

#### Path conventions

```
my_container/
├── conftest.py            # rootdir: sys.path bootstrap + pytest_plugins + container fixtures
├── entrypoint.py
├── pyproject.toml         # must include: [tool.pytest.ini_options] pythonpath = ["."]
├── config/                # optional — needed for asqi_execute tests
│   ├── suite.yaml
│   ├── systems.yaml
│   ├── datasets.yaml
│   └── score_card.yaml
└── tests/
    └── test_e2e.py
```

#### Running container tests locally

```bash
cd test_containers/my_container
uv sync --locked
uv run pytest tests/ -v
```

The `-v` flag prints each test name and result as the suite runs.

To skip tests that require credentials:

```bash
uv run pytest tests/ -v -m "not requires_hf_token and not requires_llm_api"
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
