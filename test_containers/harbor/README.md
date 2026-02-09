# Harbor Agentic Coding Test Container

A comprehensive test container for evaluating agentic coding systems using the [Harbor](https://harborframework.com/) framework. This container orchestrates agent-based code generation tasks across multiple datasets and measures key performance metrics.

## Quick Start

### 1. Build the Container

```bash
docker build --no-cache -t asqiengineer/test-container:harbor-latest .
```

### 2. Configure Your Test Suite

Edit `config/suites/coding_agent_test_suite.yaml`:

```yaml
test_suite:
  - id: "harbor_terminalbenchsample"
    name: "Terminal-Bench-Sample"
    image: "asqiengineer/test-container:harbor-latest"
    params:
      dataset: terminal-bench-sample@2.0
      tasks: ["chess-best-move", "polyglot-c-py"]
      concurrency: 1  # Number of parallel tasks
    volumes:
      output: /path/to/output
```

### 3. Run the Test

```bash
asqi execute-tests \
  -t config/suites/coding_agent_test_suite.yaml \
  -s config/systems/coding_agents.yaml \
  --container-config config/container_configs/harbor_long_timeout.yaml \
  -o output.json
```
