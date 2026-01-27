# Harbor test container — quick start

The container evaluates systems using the Harbor evaluation framework.

## Build the container

From this directory (`test_containers/harbor/`) you can build the test container. Two common options:

- Quick build (uses Docker cache):

```bash
docker build -t asqiengineer/test-container:harbor-latest .
```

- Clean build (no cache — recommended after dependency changes):

```bash
docker build --no-cache -t asqiengineer/test-container:harbor-latest .
```

Use the clean build when you changed Python dependencies or the Dockerfile; use the cached build for fast iterative edits that don't touch dependencies.

## Run via ASQI

Use ASQI CLI to run the test container and produce ASQI-standard outputs.

- Run tests only (produce raw test results):

```bash
asqi execute-tests -t config/suites/coding_agent_test.yaml -s config/systems/coding_agents.yaml -o test_results.json
```