# Contribution Guide

Thank you for contributing to ASQI Engineer. This document focuses on the contribution workflow, PR expectations, and where to find the technical setup instructions.

## Key Cross-References

- Landing / overview: `README.md` (project purpose, high-level quick start).
- Developer setup and hands-on commands: `DEVELOPMENT.md` (devcontainer, env, install, tests).

## How to Interact

- Report bugs using GitHub Issues with a concise title and a minimal reproduction.
- Request features by opening an Issue with motivation, desired behaviour, and suggested tests.
- Use Discussions for design conversations or larger proposals.

## Suggested Issue Template Fields

- title — short description
- environment — OS, Python, Docker, uv version
- repro steps — commands or YAML snippets
- logs — relevant log excerpts or attachments
- commit — relevant git SHA

## Contributing Code (PR Workflow)

1. Fork the repository and create a branch: `feat/<short>` or `fix/<short>`.
2. Add tests that exercise the behaviour in `tests/`.
3. Run tests and linters locally (see `DEVELOPMENT.md` for commands).
4. Push your branch and open a Pull Request against the default branch with a clear summary and linked issues.
5. Respond to reviewer feedback and keep PRs small and focused when possible.

### Commit Message Guidance

Prefer conventional-style prefixes such as `feat:`, `fix:`, `docs:`.

## Adding or Updating Test Containers

- Place containers under `test_containers/<name>/` and include `Dockerfile`, `entrypoint.py`, and `manifest.yaml`.
- `entrypoint.py` must accept `--systems-params` and `--test-params` JSON and write a JSON results object to stdout with the metrics declared in `manifest.yaml`.
- Add a simple example suite under `config/suites/` and lightweight tests in `tests/` to validate the manifest.

## Score Cards and Suites

- Add score cards to `config/score_cards/` and test suites to `config/suites/`.
- When changing scoring logic, include unit tests that cover edge cases and indicator thresholds.

## Documentation Contributions

- Edit files under `docs/` and preview locally with Sphinx (`make -C docs html`).
- Keep docs actionable: include example commands and minimum reproductions.

## CI, Releases, and Packaging

- The CI workflow lives in `.github/workflows/` and runs tests and packaging.
- To create releases follow the repo's tagging and CI release process explained in the workflows.

## Help & Contact

- For help, open an Issue with reproduction steps and tag `help-wanted`.
- For process questions, comment on PRs or start a Discussion.

See `DEVELOPMENT.md` for step-by-step environment setup and `README.md` for quick overview and links.
