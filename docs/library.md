# Library Usage Guide

This document explains how to call ASQI Engineer from Python without going through the CLI. It highlights the most important modules, shows end-to-end usage patterns, and demonstrates how to extend the default workflows with custom pre- and post-processing steps.

## Core Functions

### Environment prerequisites

- `DBOS_DATABASE_URL` **must** be set before importing `asqi.workflow`. It should point at a PostgreSQL database (for example `postgresql://postgres:postgres@localhost:5432/asqi`). The DBOS runtime uses this database to record workflow state.
- `OTEL_EXPORTER_OTLP_ENDPOINT` is optional. When provided, traces from the workflow are exported to the configured OpenTelemetry collector.
- Docker must be reachable from the host where your code runs. The workflow module connects with `docker.from_env()` and expects access to `/var/run/docker.sock` (or the path exposed through `DOCKER_HOST`).

### Import quick reference

```python
from dbos import DBOS

from asqi.config import (
    ContainerConfig,
    ExecutorConfig,
    load_config_file,
    merge_defaults_into_suite,
)
from asqi.workflow import (
    TestExecutionResult,
    run_end_to_end_workflow,
    run_test_suite_workflow,
    start_score_card_evaluation,
    start_test_execution,
)
```

### Key APIs

- `start_test_execution(...)` orchestrates validation, image management, and container execution. Pass file paths for the suite and systems configs plus dictionaries describing executor behavior (concurrency, failure reporting, progress cadence). Optional arguments let you select individual tests (`test_names`), supply score cards, and persist the results to disk. The function blocks until the workflow finishes and returns the DBOS workflow ID.
- `start_score_card_evaluation(...)` evaluates one or more score cards against an existing JSON results file (generated either by the CLI or by `start_test_execution`).
- `run_test_suite_workflow(...)` and `run_end_to_end_workflow(...)` are decorated with `@DBOS.workflow`. Use them when you need direct access to workflow handles or when you want to compose additional steps around the standard execution pipeline.
- `ContainerConfig` centralises Docker options such as timeouts, memory limits, and capabilities. Instantiate it directly (`ContainerConfig()`) or derive variants with helpers like `ContainerConfig.with_streaming(True)` and `ContainerConfig.from_run_params(...)`.
- `ExecutorConfig` exposes defaults for concurrency (`DEFAULT_CONCURRENT_TESTS`), failure reporting, and progress updates. You can reference these constants when building the dictionary passed to the workflows.
- `TestExecutionResult` describes the payload returned by the execution workflow for every test: metadata, raw container output, parsed JSON results, timing, and error details.

## Basic Library Usage Examples

### Run tests from Python code

```python
import os
from pathlib import Path

from asqi.config import (
    ContainerConfig,
    ExecutorConfig,
    load_config_file,
    merge_defaults_into_suite,
)
from asqi.workflow import start_test_execution

os.environ.setdefault(
    "DBOS_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/asqi"
)

executor_config = {
    "concurrent_tests": ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
    "max_failures": ExecutorConfig.MAX_FAILURES_DISPLAYED,
    "progress_interval": ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
}
container_config = ContainerConfig()

suite_path = Path("config/suites/demo_test.yaml")
systems_path = Path("config/systems/demo_systems.yaml")
score_cards = [
    load_config_file("config/score_cards/asqi_chatbot_score_card.yaml"),
]

workflow_id = start_test_execution(
    suite_path=str(suite_path),
    systems_path=str(systems_path),
    executor_config=executor_config,
    container_config=container_config,
    score_card_configs=score_cards,
    output_path="artifacts/demo_results.json",
)

print(f"Workflow completed: {workflow_id}")
```

Notes:

- `start_test_execution` loads the YAML files, validates manifests, and streams progress to the console. When `output_path` is provided, the final JSON (tests plus score cards) is written to disk.
- To filter the suite down to a subset of tests, pass `test_names=["run_mock_on_compatible_sut"]` (additional values can be comma-separated or repeated).
- Exceptions raised by validation (`ValueError`) or I/O (`FileNotFoundError`, `PermissionError`) bubble up so they can be handled by the caller.

### Access workflow results in-memory

When you need the aggregated results programmatically, launch the workflow yourself and read the handle. The helper functions above already configure DBOS, so you only need to provide serialised configs.

```python
from dbos import DBOS

from asqi.config import ContainerConfig, load_config_file, merge_defaults_into_suite
from asqi.workflow import run_end_to_end_workflow

suite_cfg = merge_defaults_into_suite(load_config_file("config/suites/demo_test.yaml"))
systems_cfg = load_config_file("config/systems/demo_systems.yaml")
score_cards = [load_config_file("config/score_cards/asqi_chatbot_score_card.yaml")]

executor_cfg = {
    "concurrent_tests": 2,
    "max_failures": 5,
    "progress_interval": 4,
}
container_cfg = ContainerConfig.with_streaming(True)

handle = DBOS.start_workflow(
    run_end_to_end_workflow,
    suite_cfg,
    systems_cfg,
    score_cards,
    executor_cfg,
    container_cfg,
)
results = handle.get_result()

suite_summary = results["summary"]
first_test = results["results"][0]
score_card_report = results.get("score_card")
```

`results["results"]` contains serialised `TestExecutionResult` dictionaries. Use `evaluate_score_cards_workflow` or `convert_test_results_to_objects` from `asqi.workflow` if you need to convert them back into `TestExecutionResult` instances.

### Score card only runs

```python
from asqi.config import load_config_file
from asqi.workflow import start_score_card_evaluation

score_cards = [load_config_file("config/score_cards/asqi_chatbot_score_card.yaml")]
workflow_id = start_score_card_evaluation(
    input_path="artifacts/demo_results.json",
    score_card_configs=score_cards,
    output_path="artifacts/demo_results_with_scores.json",
)
print(f"Score card evaluation workflow: {workflow_id}")
```

## Workflow Customization & Extension

### Default workflow anatomy

`run_test_suite_workflow` performs these phases inside DBOS:

1. Validate suite and systems configs, including volume mount safety checks.
2. Resolve Docker images (pulling if needed) and extract container manifests.
3. Build an execution queue (`Queue`) and run tests with the specified concurrency.
4. Collect `TestExecutionResult` objects, log failures, and produce a JSON-friendly structure.

`run_end_to_end_workflow` chains the execution workflow with `evaluate_score_cards_workflow`, which converts the raw dictionaries back into `TestExecutionResult` instances before running `evaluate_score_card`.

### Custom pre- and post-processing hooks

Use the DBOS decorators to compose additional steps around the stock workflow. For example, you can enrich the suite before execution and attach a custom report afterwards.

```python
from copy import deepcopy
from typing import Any, Dict

from dbos import DBOS

from asqi.workflow import run_test_suite_workflow


@DBOS.step()
def inject_preprocessor_tags(suite_config: Dict[str, Any]) -> Dict[str, Any]:
    updated = deepcopy(suite_config)
    for test in updated.get("test_suite", []):
        test.setdefault("params", {})["pre_processor_run"] = True
    return updated


@DBOS.step()
def attach_custom_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    annotated = results.copy()
    annotated["metadata"] = {"processed_by": "my-team", "version": "2025.10"}
    return annotated


@DBOS.workflow()
def run_suite_with_hooks(
    suite_config: Dict[str, Any],
    systems_config: Dict[str, Any],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
) -> Dict[str, Any]:
    prepped_suite = inject_preprocessor_tags(suite_config)
    raw_results = run_test_suite_workflow(
        prepped_suite, systems_config, executor_config, container_config
    )
    return attach_custom_summary(raw_results)
```

Launch the custom workflow exactly like the built-in one:

```python
handle = DBOS.start_workflow(
    run_suite_with_hooks,
    suite_cfg,
    systems_cfg,
    executor_cfg,
    container_cfg,
)
customised = handle.get_result()
```

### Extending container behaviour

`ContainerConfig` exposes every argument passed to `docker.containers.run`. Combine the helpers to tailor execution for a specific test family.

```python
gpu_enabled = ContainerConfig.from_run_params(
    device_requests=[{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}],
    mem_limit="8g",
)
```

You can also layer image-specific settings by updating `container_config.run_params` inside a DBOS step before calling `run_container_with_args`. Remember to clean up any temporary adjustments to avoid leaking state between tests.

### Handling results programmatically

The score card evaluation pipeline is modular:

1. `convert_test_results_to_objects` turns dictionaries back into `TestExecutionResult` instances.
2. `evaluate_score_card` (a DBOS step) calls `ScoreCardEngine`.
3. `add_score_cards_to_results` merges the evaluation output back into the workflow results.

Reuse any of these steps in your own workflow to add bespoke analytics (for example, pushing metrics to a dashboard or enriching results with organisation-specific grading logic).

