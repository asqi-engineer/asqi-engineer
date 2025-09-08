# Contribution Guide

Thank you for contributing to ASQI Engineer. This document lists every common way you may interact with the project.

## Table of contents
- How to interact (issues, discussions, support)
- Reporting bugs
- Requesting features
- Contributing code (PR workflow)
- Local development & devcontainer
- Running tests and linting
- Adding or updating test containers
- Contributing score cards or suites
- Documentation contributions
- CI, releases and packaging
- Pre-commit and style guidance
- Help & contact

## How to interact

- Report bugs using GitHub Issues with clear title and steps to reproduce.
- Request features by opening an Issue and tagging it as a proposal/feature request.
- For discussion or design questions, use the repository Discussions (or open an issue and mark it as a discussion).

If you need quick help, open an Issue and label it `help-wanted`.

## Reporting bugs

When filing a bug report include:

1. A short, descriptive title.
2. Environment (OS, Python version, docker version, uv version if used).
3. Reproduction steps (commands or YAML files). Prefer minimal steps and attach example config files when possible.
4. Expected vs actual behavior, with logs and error traces.
5. Version information: output of `git rev-parse --short HEAD` and `pip show asqi-engineer` (if installed).

Suggested issue template fields:
- `title` - short description
- `environment` - OS, Python, docker
- `repro steps` - copy/paste commands and config
- `logs` - paste relevant logs or attach files
- `commit` - commit hash

## Requesting features

Open an issue and include:

- Motivation and problem statement
- Proposed behaviour and API changes (if any)
- Backwards-compatibility concerns
- Suggested tests and examples

If your feature is substantial, propose a short design in a discuss/issue before starting implementation.

## Contributing code (PR workflow)

1. Fork the repository, create a branch with a clear name: `feat/<short>-summary` or `fix/<short>-summary`.
2. Write tests for your change. Tests live in `tests/` and must cover the new behaviour.
3. Run the test suite locally and ensure `uv run pytest` passes.
4. Lint and format code: run pre-commit hooks (see section below).
5. Push your branch, open a Pull Request against `main` (or the repo's default branch). Reference any related issues.
6. Use descriptive PR titles and include a short changelog entry when relevant.

PR reviewers will check tests, style, and compatibility. Be prepared to iterate on feedback.

### Commit message guidance

Use conventional commit-like messages, e.g. `feat: add X`, `fix: correct Y`, `docs: update Z`.

## Local development & devcontainer

Recommended quick-start:

1. Clone the repo and open in VS Code. Reopen in devcontainer if prompted.
2. Create a `.env` from `.env.example` and populate required keys (LLM keys are optional for local testing).
3. Install dev deps:

```bash
uv sync --dev
```

4. Activate venv (if the devcontainer does not do it):

```bash
source .venv/bin/activate
```

5. Run CLI help to ensure installation: `asqi --help`.

Devcontainer: If you prefer to use the provided devcontainer, open the repo in VS Code and accept the prompt to reopen in container. See `.devcontainer/` for details.

## Running tests and linting

Basic test commands:

```bash
uv run pytest
uv run pytest --maxfail=1 -q
```

Run the repository pre-commit hooks locally (recommended before committing):

```bash
pre-commit run --all-files
```

If you use black/isort/ruff in your editor, configure it to follow the repo settings.

## Adding or updating test containers

Test containers live in `test_containers/`. When adding a new container:

1. Create a new directory `test_containers/<name>/`.
2. Include the following minimal files:
   - `Dockerfile`
   - `entrypoint.py` (accepts `--systems-params` and `--test-params` JSON)
   - `manifest.yaml` (metadata: name, version, input_systems, output_metrics)
3. Ensure `entrypoint.py` outputs a JSON results object to stdout with at least `success` and `score` fields (see existing containers for examples).
4. Add an example suite under `config/suites/` and reference it in `docs/examples.md`.
5. Add tests that validate the manifest and basic compatibility via `tests/` (see test helpers already in the repo).

Recommended manifest fields (example):

```yaml
name: "my_test_framework"
version: "1.0.0"
input_systems:
  - name: "system_under_test"
    type: "llm_api"
    required: true
output_metrics: ["success", "score"]
```

If containers save logs to a mounted volume, document the default filenames in `manifest.yaml` so users can find them.

## Contributing score cards or suites

- Add score card YAMLs to `config/score_cards/` and test suites to `config/suites/`.
- Include an example run command in `docs/examples.md`.
- When changing scoring logic, provide unit tests that exercise indicator evaluation and edge cases.

## Documentation contributions

Docs live in `docs/`. To update docs:

1. Edit the appropriate `.md` or `.rst` file under `docs/`.
2. Preview local docs (if you run Sphinx): `make -C docs html`.
3. Keep docs concise and add examples for common flows.

## CI, releases and packaging

- The repository uses CI pipeline in `.github/workflows/` that runs tests and builds artifacts.
- To create a release: follow the repository release process (tagging + CI publishing). See `.github/workflows/asqi-cd.yaml` for build steps.

## Pre-commit and style guidance

- Run `pre-commit` hooks before pushing. The hooks enforce formatting and basic linting.
- Use `black` for formatting and `ruff`/`flake8` for linting rules the repo uses.

## Help & contact

- For general help, open an Issue and tag `help-wanted`.
- For contribution process questions, ping maintainers in PRs or use Discussions.

Thank you for helping improve ASQI Engineer!
