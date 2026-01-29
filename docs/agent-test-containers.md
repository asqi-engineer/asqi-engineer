# Harbor Agentic Coding Tester

**Purpose**: Run agentic coding / terminal-bench style evaluations and collect verifier metrics across tasks (pass/fail, tokens, timing).

**Framework**: [Harbor](https://github.com/laude-institute/harbor) - Agentic evaluation and benchmark runner for terminal-based tasks
**Location**: `test_containers/harbor/`

### System Requirements
- **System Under Test**: `agent_cli` (required) - The agent CLI system being tested

### Input Parameters
- `dataset` (string, required): Harbor dataset name (e.g. `hello-world@1.0`)
- `tasks` (list, optional): List of task names within the dataset to run (enables parallel execution)

### Output Metrics
- `success` (boolean): Whether the container completed successfully
- `pass_rate` (float): Dataset-level mean metric from Harbor (0.0 to 1.0)
- `n_total_trials` (integer): Total trials run
- `n_errors` (integer): Number of errors encountered
- `avg_tokens_per_task` (float): Average output tokens used per task
- `avg_task_execution_time` (float): Average execution time per task (s)
- `avg_throughput_tokens_per_sec` (float)
- `avg_latency_ms_per_token` (float)

### Example Configuration
```yaml
test_suite:
  - id: "harbor_test"
    name: "Harbor Hello World"
    image: "asqiengineer/test-container:harbor-latest"
    params:
      dataset: hello-world
      tasks: ["hello-world"]
    volumes:
      output: /path/to/output
```

### Build Instructions
```bash
cd test_containers/harbor
docker build -t asqiengineer/test-container:harbor-latest .
```
# Agent Test Containers

This document describes test containers focused on agent-style systems (e.g., `agent_cli`).

## Harbor Agentic Coding Tester

**Purpose**: Run agentic coding / terminal-bench style evaluations and collect verifier metrics across tasks (pass/fail, tokens, timing).

**Framework**: [Harbor](https://github.com/laude-institute/harbor) - Agentic evaluation and benchmark runner for terminal-based tasks
**Location**: `test_containers/harbor/`

### System Requirements
- **System Under Test**: `agent_cli` (required) - The agent CLI system being tested

### Input Parameters
- `dataset` (string, required): Harbor dataset name (e.g. `hello-world@1.0`)
- `tasks` (list, optional): List of task names within the dataset to run (enables parallel execution)

### Output Metrics
- `success` (boolean): Whether the container completed successfully
- `pass_rate` (float): Dataset-level mean metric from Harbor (0.0 to 1.0)
- `n_total_trials` (integer): Total trials run
- `n_errors` (integer): Number of errors encountered
- `avg_tokens_per_task` (float): Average output tokens used per task
- `avg_task_execution_time` (float): Average execution time per task (s)
- `avg_throughput_tokens_per_sec` (float)
- `avg_latency_ms_per_token` (float)

### Example Configuration
```yaml
test_suite:
  - id: "harbor_test"
    name: "Harbor Hello World"
    image: "asqiengineer/test-container:harbor-latest"
    params:
      dataset: hello-world
      tasks: ["hello-world"]
    volumes:
      output: /path/to/output
```

### Build Instructions
```bash
cd test_containers/harbor
docker build -t asqiengineer/test-container:harbor-latest .
```
