# Agent Test Containers

ASQI Engineer provides specialized test containers for evaluating agent-style systems that interact with terminal environments and perform multi-step reasoning tasks.

---

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
- `success` (boolean): Whether the execution completed successfully
- `pass_rate` (float): Proportion of tasks passed (0.0 to 1.0)
- `n_total_trials` (integer): Total number of trials run
- `n_errors` (integer): Number of errors encountered
- `avg_tokens_per_task` (float): Average number of output tokens used per task
- `avg_e2e_task_latency` (float): Average end-to-end task execution time in seconds
- `avg_throughput_tokens_per_sec` (float): Average throughput in tokens per second
- `avg_time_per_output_token_ms` (float): Average per token generation speed in milliseconds
- `avg_ttft_ms` (float): Average time-to-first-token (TTFT) in milliseconds

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
