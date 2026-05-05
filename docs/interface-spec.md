# ASQI-Engineer Interface Specification

## Purpose

This document fixes the **public interfaces of asqi-engineer** so that internal refactors cannot silently break downstream consumers — applications embedding the library, container authors building test packages, and operators driving the CLI.

**Three interface surfaces are covered:**

1. **Host-side library API** (§1.1–§1.5) — Python symbols importable from `asqi.*` to orchestrate workloads from an embedding application.
2. **CLI** (§2) — every `asqi <command>` entry point.
3. **In-container SDK** (§1.6) — Python symbols that test and synthetic-data-generation (SDG) containers import from `asqi.*` at runtime, inside the container, to read inputs and write standardized outputs. Distinct from §1.1–§1.5 because it executes in a different process, in a different image, with a different lifecycle.

Plus the cross-cutting implicit contracts: environment variables, file conventions, error mapping, and DBOS state assumptions.

---

## 1. Library API

§1.1–§1.5 cover the **host-side library API** — symbols an embedding application imports to orchestrate workloads. §1.6 covers the **in-container SDK** — symbols test and SDG containers import inside their Docker images.

### 1.0 Public surface inventory

#### 1.0.a Host-side

The following symbols form the public host-side API. Embedding applications import these from `asqi.*` to validate configuration, run test suites, generate datasets, and evaluate score cards.

| Module | Public symbols |
|---|---|
| `asqi.workflow` | `run_test_suite_workflow`, `evaluate_score_cards_workflow`, `run_data_generation_workflow` |
| `asqi.config` | `ContainerConfig`, `ExecutorConfig`, `ExecutionMode`, `merge_defaults_into_suite` |
| `asqi.container_manager` | `extract_manifest_from_image` |
| `asqi.schemas` | `Manifest`, `SystemInput`, `InputParameter`, `InputDataset`, `DatasetType`, `DatasetFeature` |
| `asqi.errors` | `ManifestExtractionError`, `AuditResponsesRequiredError` (see §1.5) |

#### 1.0.b In-container SDK

A separate set of asqi-engineer modules is imported at *container runtime* by test and SDG containers. The asqi-engineer wheel is installed into each container image, and entrypoints depend on these helpers. Documented exhaustively in §1.6.

| Module | Symbols available to containers | Typical use |
|---|---|---|
| `asqi.datasets` | `load_hf_dataset`, `validate_dataset_features`, `Dataset` (re-exported from HF `datasets`) | Loading and validating HuggingFace / local datasets |
| `asqi.loaders` | `load_test_cases` | Loading and validating test cases as typed Pydantic instances |
| `asqi.response_schemas` | `ContainerOutput`, `GeneratedDataset`, `GeneratedReport`, `validate_container_output` | Writing the standardized JSON output |
| `asqi.utils` | `get_openai_tracking_kwargs` | Forwarding metadata into LLM client calls (LiteLLM / OpenAI) |
| `asqi.output` | `parse_container_json_output` | Container-side test fixtures |
| `asqi.rag_response_schema` | `RAGCitation`, `validate_rag_response` | RAG citation validation |
| `asqi.schemas` | `Manifest` | Reading container manifests from container code (e.g. SDG entrypoints) |

A breaking change to any of these modules ripples through every dependent container image — a much larger blast radius than a host-side change. Treat the in-container SDK with at least the same stability bar as §1.1–§1.5.

### 1.1 Workflow entry points (`asqi.workflow`)

All three are decorated with `@DBOS.workflow()`. They assume `DBOS.launch()` has already been called by the host process and return JSON-serializable results (DBOS persists the result via the configured Postgres database).

#### `run_test_suite_workflow`

`src/asqi/workflow.py:949`

```python
def run_test_suite_workflow(
    suite_config: Dict[str, Any],
    systems_config: Dict[str, Any],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
    datasets_config: Optional[Dict[str, Any]] = None,
    score_card_configs: Optional[List[Dict[str, Any]]] = None,
    metadata_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]
```

**Behavior:** Validates image availability, extracts manifests, cross-validates tests/systems/manifests, runs tests concurrently, aggregates results.

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `suite_config` | `Dict[str, Any]` | yes | Serialized [`SuiteConfig`](#13-schema-models-asqischemas) dict. Callers should pre-process with `merge_defaults_into_suite` before calling. |
| `systems_config` | `Dict[str, Any]` | yes | Serialized [`SystemsConfig`](#13-schema-models-asqischemas) dict. |
| `executor_config` | `Dict[str, Any]` | yes | Plain dict with three keys: `concurrent_tests: int`, `max_failures: int`, `progress_interval: int`. **Not** an [`ExecutorConfig`](#122-executorconfig-dataclass) instance — see §1.2.2. |
| `container_config` | [`ContainerConfig`](#121-containerconfig-pydantic) | yes | Pydantic instance, **not** a dict. |
| `datasets_config` | `Optional[Dict[str, Any]]` | no (default `None`) | Serialized `DatasetsConfig` dict for resolving dataset references in the suite. |
| `score_card_configs` | `Optional[List[Dict[str, Any]]]` | no (default `None`) | List of score-card dicts. Used here only for validation; this function does **not** evaluate score cards (see `evaluate_score_cards_workflow`). |
| `metadata_config` | `Optional[Dict[str, Any]]` | no (default `None`) | Forwarded into test containers as metadata. Common keys: `parent_id`, `job_type`, `user_id`. |

**Return:** `Tuple[Dict[str, Any], List[Dict[str, Any]]]` — `(test_results, container_results)`.

`test_results` shape:
```jsonc
{
  "summary": {
        "suite_name": str,
        "workflow_id": str,
        "status": str,                        // see "Status values used" below
        "total_tests": int,
        "successful_tests": int,
        "failed_tests": int,
        "success_rate": float,                // successful_tests / total_tests; 0.0 when total_tests==0
        "total_execution_time": float,        // seconds
        "timestamp": str,                     // ISO 8601, set when summary is created
        // The keys below are conditional:
        "images_checked": int,                // happy path (status=COMPLETED) only
        "manifests_extracted": int,           // happy path only
        "validation_errors": list[str],       // happy path; also set on VALIDATION_FAILED when validate_test_plan produced errors
        "error": str                          // CONFIG_ERROR and VALIDATION_FAILED-volume early returns only
    },
  "results": [ /* one entry per test, shape below */ ]
}
```

Each entry in `results` is produced by `TestExecutionResult.result_dict()` and has this fixed shape:
```jsonc
{
  "metadata": {
        "test_id": str,
        "test_name": str,
        "sut_name": str | null,
        "system_type": str | null,
        "image": str,
        "start_time": float,                  // epoch seconds
        "end_time": float,                    // epoch seconds
        "execution_time_seconds": float,
        "container_id": str,                  // empty string if container never started
        "exit_code": int,                     // -1 if container never started
        "timestamp": str,                     // ISO 8601, set at serialization time
        "success": bool
    },
  "test_results": { /* container-emitted dict — the `results` field of ContainerOutput */ },
  "generated_reports": [ /* list of GeneratedReport dicts (model_dump) */ ],
  "generated_datasets": [ /* list of GeneratedDataset dicts (model_dump) */ ]
}
```

**Note on the field name:** `run_test_suite_workflow` emits the per-test container dict under the key `test_results`. `run_data_generation_workflow` emits the same shape but under the key `results` (see that function's return shape note). The wrapping top-level structure (`{"summary", "results"}`) is identical between the two.

`container_results` shape — one entry per test, in the same order as `test_results.results`:
```jsonc
{
  "test_id": str,
  "error_message": str,                      // empty when success=true
  "container_output": str                    // raw container stdout
}
```

On configuration / validation failure, the returned `test_results` has the same wrapper shape with `summary.status` set to `CONFIG_ERROR` or `VALIDATION_FAILED` and `results: []` — **the function does not raise** for these classes. The `container_results` element will be `[]`.

**Status values used:**
- `CONFIG_ERROR` — Pydantic validation or structural type errors during config parsing.
- `VALIDATION_FAILED` — volume validation, test-plan cross-validation, or score-card report validation failed.
- `NO_TESTS` — config parsed, validated, and produced an empty execution plan (no tests to run).
- `COMPLETED` — workflow ran tests to completion (regardless of per-test success/failure; aggregate counts are in `successful_tests` / `failed_tests`).

**Exceptions raised (uncaught):** Two distinct try/except blocks at the top of the workflow swallow the classes documented above:

- Config parsing catches `ValidationError` and `(TypeError, AttributeError)` and re-shapes them into a `CONFIG_ERROR` summary.
- Volume validation (`validate_test_volumes`) catches `ValueError` and re-shapes it into a `VALIDATION_FAILED` summary — *not* `CONFIG_ERROR`.

Anything outside those sets — and any exception from the inner execution layers (Docker, container manager, score-card validation) — propagates to the caller.

**Pre-conditions:**
- `DBOS.launch()` already called.
- `DBOS_DATABASE_URL` env var set and Postgres reachable.
- Docker daemon reachable from the host (or via `DOCKER_HOST`).
- All test images either present locally or pullable from their registries.

**Side effects:**
- Creates a per-workflow DBOS `Queue` named `test_execution_<workflow_id>`.
- Pulls Docker images, runs containers (with `workflow_id=<uuid>` label), writes container logs under `$LOGS_PATH` (default `logs/`).
- Persists workflow state to Postgres (DBOS).
- Prints progress to stdout via `rich.console`.

---

#### `evaluate_score_cards_workflow`

`src/asqi/workflow.py:1370`

```python
def evaluate_score_cards_workflow(
    test_results_data: Dict[str, Any],
    test_container_data: List[Dict[str, Any]],
    score_card_configs: List[Dict[str, Any]],
    audit_responses_data: Optional[Dict[str, Any]] = None,
    execution_mode: ExecutionMode = ExecutionMode.END_TO_END,
) -> Dict[str, Any]
```

**Behavior:** Converts test result dicts back into `TestExecutionResult` objects, evaluates score cards against them, merges score card output back into the result dict.

**Parameters:**

| Name | Type | Required | Description |
|---|---|---|---|
| `test_results_data` | `Dict[str, Any]` | yes | First element of the tuple returned by `run_test_suite_workflow`. |
| `test_container_data` | `List[Dict[str, Any]]` | yes | Second element of the tuple returned by `run_test_suite_workflow`. |
| `score_card_configs` | `List[Dict[str, Any]]` | yes | List of `ScoreCard` dicts. |
| `audit_responses_data` | `Optional[Dict[str, Any]]` | no (default `None`) | Manual audit responses. Required if any score card has audit indicators and `--skip-audit` was not used at the CLI layer. |
| `execution_mode` | [`ExecutionMode`](#123-executionmode-enum) | no (default `END_TO_END`) | Expected values: `END_TO_END` or `EVALUATE_ONLY`. Callers may invoke this positionally with 4 args, leaving `execution_mode` at its default. |

**Return:** `Dict[str, Any]` — the input `test_results_data` dict copied with an added `"score_card"` key carrying the evaluation output.

**Exceptions raised:** No top-level try/except wrapper. Underlying score-card evaluation may raise `MetricExpressionError`. **`AuditResponsesRequiredError` is *not* raised here** — it is raised by the CLI-layer helper `resolve_audit_options` (see §2.2 / §2.5). Inside this workflow, missing audit responses produce per-indicator error entries in the result rather than an exception. `ReportValidationError` is also caught inside the workflow (`workflow.py:923`) and converted to an error entry.

**Side effects:**
- Prints progress to stdout via `rich.console`.
- Persists workflow state to Postgres (DBOS).

---

#### `run_data_generation_workflow`

`src/asqi/workflow.py:2004`

```python
def run_data_generation_workflow(
    generation_config: Dict[str, Any],
    systems_config: Optional[Dict[str, Any]],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
    datasets_config: Optional[Dict[str, Any]] = None,
    metadata_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]
```

**Behavior:** Mirrors `run_test_suite_workflow` but operates on a `DataGenerationConfig` and produces datasets instead of test results.

**Parameter notes:**
- `systems_config` is `Optional` here (data generation may not require AI systems).
- `metadata_config` accepts the same keys as in `run_test_suite_workflow` (`parent_id`, `job_type`, `user_id`).
- `score_card_configs` is **not** a parameter — score cards are not applicable to data generation.

**Return:** `Tuple[Dict[str, Any], List[Dict[str, Any]]]` — `(generation_results, container_results)`.

`generation_results` shape:
```jsonc
{
  "summary": {
        "suite_name": str,                    // generation.job_name
        "workflow_id": str,
        "status": str,                        // see "Status values used" below
        "total_tests": int,                   // count of generation jobs (key kept as "total_tests" for symmetry)
        "successful_tests": int,
        "failed_tests": int,
        "success_rate": float,                // successful_tests / total_tests; 0.0 when total_tests==0
        "total_execution_time": float,        // seconds
        "timestamp": str,                     // ISO 8601, set when summary is created
        // The keys below are conditional:
        "images_checked": int,                // happy path (status=COMPLETED) only
        "manifests_extracted": int,           // happy path only
        "validation_errors": list[str],       // happy path; also set on VALIDATION_FAILED when validate_test_plan produced errors
        "error": str                          // CONFIG_ERROR and VALIDATION_FAILED-volume early returns only
    },
  "results": [ /* one entry per generation job, shape below */ ]
}
```

Each entry in `results` is produced by `TestExecutionResult.result_dict(use_results_field=True)`:
```jsonc
{
  "metadata": {
        "test_id": str,
        "test_name": str,
        "sut_name": str | null,
        "system_type": str | null,
        "image": str,
        "start_time": float,                  // epoch seconds
        "end_time": float,                    // epoch seconds
        "execution_time_seconds": float,
        "container_id": str,                  // empty string if container never started
        "exit_code": int,                     // -1 if container never started
        "timestamp": str,                     // ISO 8601, set at serialization time
        "success": bool
    },
  "results": { /* container-emitted dict — the `results` field of ContainerOutput */ },
  "generated_reports": [ /* list of GeneratedReport dicts (model_dump) */ ],
  "generated_datasets": [ /* list of GeneratedDataset dicts (model_dump) */ ]
}
```

This is identical to the per-test shape from `run_test_suite_workflow` except that the container-emitted dict lives under the key `"results"` instead of `"test_results"`. The wrapping `{"summary", "results"}` structure and the `metadata` block are the same.

`container_results` shape — one entry per generation job, in the same order as `generation_results.results`:
```jsonc
{
  "test_id": str,
  "error_message": str,                      // empty when success=true
  "container_output": str                    // raw container stdout
}
```

**Status values used:** same set as `run_test_suite_workflow` — `CONFIG_ERROR`, `VALIDATION_FAILED`, `NO_TESTS`, `COMPLETED`.

On configuration / validation failure, the returned `generation_results` has the same wrapper shape with `summary.status` set to `CONFIG_ERROR` or `VALIDATION_FAILED` and `results: []` — **the function does not raise** for these classes. The `container_results` element will be `[]`.

**Side effects:** Same as `run_test_suite_workflow`, but the per-workflow DBOS queue is named `data_generation_<workflow_id>`.

---

### 1.2 Configuration types (`asqi.config`)

#### 1.2.1 `ContainerConfig` (Pydantic)

`src/asqi/config.py:67`

Pydantic `BaseModel` controlling Docker container execution.

**Class variables (immutable contract):**

| Name | Type | Value |
|---|---|---|
| `MANIFEST_PATH` | `ClassVar[str]` | `"/app/manifest.yaml"` |
| `DEFAULT_RUN_PARAMS` | `ClassVar[Dict[str, Any]]` | `{"detach": True, "remove": False, "network_mode": "host", "mem_limit": "2g", "cpu_period": 100000, "cpu_quota": 200000, "cap_drop": ["ALL"]}` |

**Instance fields:**

| Name | Type | Default | Description |
|---|---|---|---|
| `timeout_seconds` | `int` | `300` | Maximum container execution time in seconds. |
| `stream_logs` | `bool` | `False` | Stream container logs during execution. |
| `cleanup_on_finish` | `bool` | `True` | Cleanup containers on finish. |
| `cleanup_force` | `bool` | `True` | Force cleanup even if graceful stop fails. |
| `run_params` | `Dict[str, Any]` | copy of `DEFAULT_RUN_PARAMS` | Docker `containers.run()` kwargs (merged with defaults). |

**Class methods:**

```python
@classmethod
def load_from_yaml(cls, path: str) -> "ContainerConfig"
```
Reads YAML at `path` (with env-var interpolation), merges `run_params` over defaults, returns a new instance. Raises `FileNotFoundError` if `path` does not exist.

```python
@classmethod
def with_streaming(cls, enabled: bool) -> "ContainerConfig"
```
Returns a fresh default-configured instance with `stream_logs=enabled`.

```python
@classmethod
def from_run_params(
    cls,
    *,
    detach: Optional[bool] = None,
    remove: Optional[bool] = None,
    network_mode: Optional[str] = None,
    mem_limit: Optional[str] = None,
    cpu_period: Optional[int] = None,
    cpu_quota: Optional[int] = None,
    **extra: Any,
) -> "ContainerConfig"
```
Returns a fresh instance with `DEFAULT_RUN_PARAMS` merged with non-`None` overrides and any extra kwargs.

**Common usage patterns:**
- `ContainerConfig.with_streaming(True)` — minimal-config instance with log streaming on.
- Direct mutation of `container_config.run_params["privileged"] = True` when a manifest declares `host_access: true` (see §3.7).
- **Pitfall:** `ContainerConfig(timeout_seconds=3000).with_streaming(True)` discards the prior instance — `with_streaming` is a classmethod that returns a fresh default-configured object. Only `stream_logs=True` is preserved; other fields revert to defaults. Use `from_run_params` or direct construction if you need to combine custom fields with streaming.

#### 1.2.2 `ExecutorConfig` (dataclass)

`src/asqi/config.py:175`

```python
@dataclass
class ExecutorConfig:
    DEFAULT_CONCURRENT_TESTS: int = 3
    MAX_FAILURES_DISPLAYED: int = 3
    PROGRESS_UPDATE_INTERVAL: int = 4
```

**Important: this dataclass is a constants holder, not the workflow argument type.** Workflows take `executor_config` as a plain `dict`, not an `ExecutorConfig` instance. The class is exported for callers that want a single source of truth for default values.

The contract is the **dict shape**:
```python
executor_config = {
    "concurrent_tests": int,    # required, range checked at CLI: 1..20
    "max_failures": int,         # required, range checked at CLI: 1..10
    "progress_interval": int,    # required, range checked at CLI: 1..10
}
```

#### 1.2.3 `ExecutionMode` (enum)

`src/asqi/config.py:268`

```python
class ExecutionMode(str, Enum):
    END_TO_END = "end_to_end"
    TESTS_ONLY = "tests_only"
    EVALUATE_ONLY = "evaluate_only"
    VALIDATE_ONLY = "validate_only"
```

Callers don't typically import this directly; it surfaces only as a parameter default on `evaluate_score_cards_workflow`. Listed here because changing the enum or its default would be a breaking change.

#### 1.2.4 `merge_defaults_into_suite`

`src/asqi/config.py:201`

```python
def merge_defaults_into_suite(config: Dict[str, Any]) -> Dict[str, Any]
```

Deep-merges the `test_suite_default` block into each entry in `test_suite`. Per-test fields override defaults; nested dicts are recursively merged. Mutates and returns the input `config`. If `test_suite_default` is absent, returns `config` unchanged.

Callers should invoke this on every suite config before passing it to `run_test_suite_workflow`.

---

### 1.3 Schema models (`asqi.schemas`)

Six classes form the public schema surface. Five are Pydantic `BaseModel`s; `DatasetType` is a `StrEnum`, not a model. The regex `^[0-9a-z_]{1,32}$` (`IDsStringPattern` in `schemas.py`) is enforced selectively, not "across the system" — see §3.3 for the actual enforcement scope.

#### `Manifest` — `src/asqi/schemas.py:513`

Validates the `manifest.yaml` embedded in every test container.

| Field | Type | Default | Required |
|---|---|---|---|
| `name` | `str` | — | yes |
| `version` | `str` | — | yes |
| `description` | `str \| None` | `None` | no |
| `host_access` | `bool` | `False` | no |
| `input_systems` | `list[SystemInput]` | `[]` | no |
| `input_schema` | `list[InputParameter]` | `[]` | no |
| `input_datasets` | `list[InputDataset]` | `[]` | no |
| `output_metrics` | `list[str] \| list[OutputMetric]` | `[]` | no |
| `output_artifacts` | `list[OutputArtifact] \| None` | `None` | no |
| `environment_variables` | `list[EnvironmentVariable]` | `[]` | no |
| `output_reports` | `list[OutputReports]` | `[]` | no |
| `output_datasets` | `list[OutputDataset]` | `[]` | no |

#### `SystemInput` — `src/asqi/schemas.py:272`

| Field | Type | Required |
|---|---|---|
| `name` | `str` | yes — e.g. `system_under_test`, `simulator_system`, `evaluator_system` |
| `type` | `str \| list[str]` | yes — accepted values: `llm_api`, `rest_api`, `rag_api`, `image_generation_api`, `image_editing_api`, `vlm_api`, `agent_cli` |
| `required` | `bool` | no (default `True`) |
| `description` | `str \| None` | no |

#### `InputParameter` — `src/asqi/schemas.py:295`

User-supplied test parameter schema. `type` is one of `"string" | "integer" | "float" | "boolean" | "list" | "object" | "enum"`. Recursive: `items` may itself be an `InputParameter`. Includes a `ui_config` field intended for consumers that auto-generate UI from the schema.

#### `InputDataset` — `src/asqi/schemas.py:429`

| Field | Type | Required |
|---|---|---|
| `name` | `str` | yes |
| `type` | `DatasetType \| list[DatasetType]` | yes |
| `required` | `bool` | no (default `True`) |
| `description` | `str \| None` | no |
| `features` | `list[DatasetFeature \| HFFeature] \| None` | no — **required when `type` includes `huggingface`** |

Validator `_validate_features_for_huggingface` raises `ValueError` if `huggingface` is in `type` but `features` is empty.

#### `DatasetType` — `src/asqi/schemas.py:421`

```python
class DatasetType(StrEnum):
    HUGGINGFACE = "huggingface"
    PDF = "pdf"
    TXT = "txt"
```

#### `DatasetFeature` — `src/asqi/schemas.py:223`

| Field | Type | Required |
|---|---|---|
| `name` | `str` | yes |
| `dtype` | `HFDtype` | yes — e.g. `"string"`, `"int64"`, `"float32"`, `"bool"` |
| `required` | `bool` | no (default `False`) |
| `description` | `str \| None` | no |

---

### 1.4 Helper functions

#### `extract_manifest_from_image` — `src/asqi/container_manager.py:156`

```python
def extract_manifest_from_image(
    image: str,
    manifest_path: str = "/app/manifest.yaml",
) -> Optional[Manifest]
```

Creates a temporary container from `image`, copies `manifest_path` out, parses it through the `Manifest` Pydantic model, returns the instance. The temporary container is always cleaned up.

**Raises `ManifestExtractionError`** with a `error_type` discriminator covering Docker-layer errors, tar-extraction errors, YAML/file-read errors, and schema-validation errors. See §1.5 for the full enumeration.

Intended for callers that need to inspect a container's declared inputs/outputs before scheduling a run (e.g., for caching, UI rendering, or pre-flight validation).

---

### 1.5 Error classes (`asqi.errors`)

`src/asqi/errors.py`

Two error classes carry structured payloads designed for callers to catch and inspect:

| Class | Raised by | Carries |
|---|---|---|
| `ManifestExtractionError(Exception)` | `extract_manifest_from_image` (§1.4) | `error_type: str`, `original_error: Optional[Exception]`. The full set of `error_type` values (`container_manager.extract_manifest_from_image`): `IMAGE_NOT_FOUND`, `MANIFEST_FILE_NOT_FOUND`, `DOCKER_API_ERROR`, `TAR_EXTRACTION_ERROR`, `TAR_IO_ERROR`, `MANIFEST_FILE_MISSING_AFTER_EXTRACTION`, `YAML_PARSING_ERROR`, `FILE_READ_ERROR`, `EMPTY_MANIFEST_FILE`, `SCHEMA_VALIDATION_ERROR`, `UNEXPECTED_ERROR`. Branch on `error_type` for differentiated handling; treat the set as open-ended (new values may be added without notice). |
| `AuditResponsesRequiredError(Exception)` | `asqi.main.resolve_audit_options` (CLI helper, `main.py:153`) when audit indicators exist but neither `--audit-responses` nor `--skip-audit` was supplied. **Not raised inside `evaluate_score_cards_workflow`** — host-side library callers do not see this error from the workflow itself. | `score_card_name: str`, `audit_indicators: List[Dict[str, Any]]`. The rendered message includes a YAML template; catchers typically use this to re-prompt for audit responses and retry. |

**Other exceptions that may propagate uncaught** — plain exceptions with no extra attributes, so catching them specifically is no different from catching `Exception`. Listed for completeness only:
- From `evaluate_score_cards_workflow`: `MetricExpressionError`. (`ReportValidationError` is **caught** inside `evaluate_score_card` (`workflow.py:923`) alongside `KeyError` / `AttributeError` / `TypeError` / `ValueError` and converted into a per-indicator error entry in `assessments` — it does not propagate.)
- From `run_test_suite_workflow` / `run_data_generation_workflow` (Docker layer): `MissingImageError`, `MountExtractionError`.
- From `asqi.validation.validate_ids` (CLI-internal, not reachable via the documented host-side API): `DuplicateIDError`, `MissingIDFieldError`.

---

### 1.6 In-container SDK

Modules below are imported by code running **inside** test/SDG container images. Container authors install the asqi-engineer wheel into the container image so these helpers are available at entrypoint time. They convert the container contract (env vars in, JSON out) from raw boilerplate into typed, validated calls.

The host-side workflows write to and read from `/input` and `/output` mounts. Inside the container, the SDK is what bridges those mounts and the user's Python code.

#### 1.6.1 `asqi.datasets`

`src/asqi/datasets.py`

| Symbol | Source | Description |
|---|---|---|
| `Dataset` | re-exported from HuggingFace `datasets` package (line 6) | Standard HuggingFace `Dataset` type. Containers import this from `asqi.datasets` rather than `datasets` directly to keep their import surface single-rooted. |
| `load_hf_dataset` | `datasets.py:237` | Load a HuggingFace dataset (Hub or local files) with optional feature validation. |
| `validate_dataset_features` | `datasets.py:105` | Validate that a `Dataset` has all required features and matching dtypes. |

```python
def load_hf_dataset(
    dataset_config: Union[dict, HFDatasetDefinition],
    input_mount_path: Path | None = None,
    expected_features: Sequence[Union[DatasetFeature, HFFeature]] | None = None,
    dataset_name: str = "dataset",
) -> Dataset | IterableDataset
```

**Behavior:** if `loader_params.hub_path` is set, loads from the HuggingFace Hub; otherwise loads from local files (with `input_mount_path` prepended to relative paths — typically `/input` inside the container). If `mapping` is set on the dataset config, columns are renamed. If `expected_features` is supplied, validates after loading (skipped for `IterableDataset` / streaming).

**Raises:**
- `ValidationError` (Pydantic) if `dataset_config` dict fails validation.
- `ValueError` if `expected_features` validation fails.

```python
def validate_dataset_features(
    dataset: Dataset,
    expected_features: Sequence[Union[DatasetFeature, HFFeature]],
    dataset_name: str = "dataset",
) -> None
```

**Behavior:** asserts that every required feature is present and types match. No-op for streaming datasets (where `column_names` / `features` are `None`).

**Raises:** `ValueError` aggregating missing features and type mismatches into a single multi-line error.

#### 1.6.2 `asqi.response_schemas`

`src/asqi/response_schemas.py`

The Pydantic schema for the JSON object every container emits as the **last `{...}` block of its stdout**. This is the **wire contract** between containers and the host workflow: the host captures container stdout via `container.logs()` (`container_manager.py:609`), extracts the trailing JSON via `parse_container_json_output` (`output.py`), and validates it through this schema. The `/output` mount is **not** where this payload lives — that mount is for side artifacts (generated reports/datasets, referenced by path from inside the JSON via `GeneratedReport.report_path` / `GeneratedDataset.dataset_path`).

##### `ContainerOutput`

```python
class ContainerOutput(BaseModel):
    results: dict[str, Any] | None = None              # Recommended field name
    test_results: dict[str, Any] | None = None         # Legacy alias (deprecated)
    generated_reports: list[GeneratedReport] = []
    generated_datasets: list[GeneratedDataset] = []
    model_config = {"extra": "allow"}                  # Forward compatibility
```

**Validation rules:**
- At least one of `results` / `test_results` must be present (`validate_container_output` enforces; `ContainerOutput(...)` alone does not).
- If present, the results dict must be non-empty — the schema's docstring requires it to "contain at least 'success' field," but the validator only rejects an empty `{}`.
- Extra top-level fields are allowed (forward-compat).

**Helper:** `output.get_results()` returns `results` if set, falling back to `test_results`, falling back to `{}`.

##### `GeneratedDataset`

```python
class GeneratedDataset(BaseModel):
    dataset_name: str                                  # min_length=1, non-whitespace
    dataset_type: Literal["huggingface", "pdf", "txt"]
    dataset_path: str                                  # min_length=1, non-whitespace, container-internal path
    format: str | None = None                          # e.g. "parquet", "json", "csv"
    metadata: DatasetMetadata | dict[str, Any] | None = None
```

`dataset_path` is a path **inside the container** — the workflow translates it to a host path via `asqi.output._translate_container_path` after the container exits.

##### `GeneratedReport`

```python
class GeneratedReport(BaseModel):
    report_name: str                                   # min_length=1, non-whitespace
    report_type: Literal["html", "pdf", "json"]
    report_path: str                                   # min_length=1, non-whitespace, container-internal path
    metadata: dict[str, Any] | None = None
```

##### `DatasetMetadata` / `ColumnMetadata`

Optional structured wrappers for `GeneratedDataset.metadata`:

```python
class ColumnMetadata(BaseModel):
    name: str
    dtype: str
    description: str | None = None

class DatasetMetadata(BaseModel):
    columns: list[ColumnMetadata]
    row_count: int = Field(..., ge=0)
    size_bytes: int | None = Field(None, ge=0)
```

##### `validate_container_output`

```python
def validate_container_output(output_dict: dict[str, Any]) -> ContainerOutput
```

Stricter constructor: raises `ValueError` if neither `results` nor `test_results` is present. The host uses this when validating untrusted container JSON (`asqi.output`). Container entrypoints, however, typically *write* their output by directly instantiating `ContainerOutput(...)` (e.g. `sdg_containers/example_data_generator/entrypoint.py:227`), since they own the dict and don't need the extra validation step. Use `validate_container_output` on the read side, direct construction on the write side.

#### 1.6.3 `asqi.utils`

`src/asqi/utils.py`

```python
def get_openai_tracking_kwargs(
    metadata: Optional[Union[Dict[str, Any], ExecutionMetadata]] = None,
) -> Dict[str, Any]
```

Converts the host-provided `metadata_config` (passed into the container as a dict / `ExecutionMetadata`) into kwargs spreadable into an OpenAI / LiteLLM client call. Output shape:

```python
{
    "user": "<metadata.user_id>",                 # only when user_id is truthy
    "extra_body": {
        "metadata": {
            "tags": ["k:v", ...],                  # from metadata.tags (dict → list of "key:value")
            "<other_top_level_key>": "...",        # any top-level metadata key except user_id / tags
        }
    }
}
```

**Implicit contract:** containers that call OpenAI / LiteLLM are expected to splat this into their client calls (`client.chat.completions.create(..., **kwargs)`). Removing or renaming this function silently breaks observability tracking across every LLM-using container.

#### 1.6.4 `asqi.output`

`src/asqi/output.py`

The single function intended for container test fixtures is:

```python
def parse_container_json_output(output: str) -> Dict[str, Any]
```

**Behavior:** robust JSON extraction from raw container stdout. Strategy:
1. Strip whitespace; reject empty input with `ValueError("Empty container output ...")`.
2. If output starts with `{` and ends with `}`, attempt a direct `json.loads`.
3. Otherwise, walk lines from the bottom up, find the last line starting with `{`, treat from there to the final `}` as a JSON candidate, retry parse.
4. If nothing parses, raise `ValueError("No valid JSON found in container output...")`.

**Raises:** `ValueError` (with a 100-char output preview in the message).

Useful in container test fixtures to assert the entrypoint produces parseable output. Not intended for use in container entrypoints themselves (the entrypoint *writes* JSON; the host *reads* and parses it).


#### 1.6.5 `asqi.rag_response_schema`

`src/asqi/rag_response_schema.py`

Used by RAG-evaluating containers. Validates that a RAG API response carries the expected `choices[0].message.context.citations` structure.

```python
class RAGCitation(BaseModel):
    retrieved_context: str        # min_length=1
    document_id: str | None = None
    score: float | None = None    # ge=0.0, le=1.0
    source_id: str | None = None

class RAGContext(BaseModel):
    citations: list[RAGCitation] = []

def validate_rag_response(response_dict: dict[str, Any]) -> list[RAGCitation]
```

**`validate_rag_response` raises:**
- `KeyError` if the path `response_dict["choices"][0]["message"]["context"]` cannot be resolved. The function explicitly catches `(KeyError, IndexError, TypeError)` while navigating the structure and re-raises a single `KeyError` with a unified message — so an empty `choices` list, a non-dict node, and a missing key all surface as `KeyError`. Catchers should not branch on `IndexError` or `TypeError` here.
- `pydantic.ValidationError` if the `context` dict exists but does not match the `RAGContext` schema (e.g. a citation has `score` outside `[0.0, 1.0]`, or `retrieved_context` is empty).

**Module `__all__`:** explicitly exports `["RAGCitation", "RAGContext", "validate_rag_response"]` — a rare explicit public-API declaration in this codebase. Treat as a stable contract.

#### 1.6.6 `asqi.loaders`

`src/asqi/loaders.py`

| Symbol | Source | Description |
|---|---|---|
| `load_test_cases` | `loaders.py:32` | Load and validate test cases from a dataset, yielding one typed Pydantic instance per row. |

```python
def load_test_cases[T: BaseModel](
    path: str | Path | dict | HFDatasetDefinition,
    test_case_class: type[T],
    *,
    input_mount_path: Path | None = None,
) -> Generator[T, None, None]
```

**Behavior:** Accepts either a plain file path or an `HFDatasetDefinition` / dict config (the same format produced by dataset registry YAMLs such as `rag_datasets.yaml`).

- When `path` is a `str | Path`, supported file formats are JSONL (`.jsonl`), JSON (`.json`), CSV (`.csv`), and Parquet (`.parquet`). Relative paths are resolved against `input_mount_path` (typically `/input` inside the container).
- When `path` is a `dict | HFDatasetDefinition`, loads from the HuggingFace Hub if `loader_params.hub_path` is set; otherwise loads from local files (with `input_mount_path` prepended to relative paths). If `mapping` is set on the config, columns are renamed before iteration.

Each row is passed through `test_case_class(**row)` and the validated Pydantic instance is yielded. A single dataset row may be valid for more than one test-case schema — call `load_test_cases` once per schema; each call is independent and validates only the fields its schema requires.

**Raises:**
- `FileNotFoundError` if the resolved file path does not exist.
- `ValueError` if a row fails schema validation. The message includes the row index, the target class name, and the failing field name(s) extracted from the underlying `pydantic.ValidationError`.

---

## 2. CLI

CLI entry point: `asqi` (declared in `pyproject.toml` `[project.scripts]`, dispatches to `asqi.main:app`).
Source: `src/asqi/main.py`. CLI framework: Typer.

**Global flag (any subcommand):**

| Flag | Short | Type | Default | Behavior |
|---|---|---|---|---|
| `--version` | `-V` | `bool` | `None` | Eager. Prints `asqi-engineer version <ver>, build <git-hash>` when the installed version carries a `+local` build suffix, or `asqi-engineer version <ver>` otherwise. If the package metadata is unavailable, prints `asqi-engineer version: unknown (not installed)`. Exits 0 in all cases. |

**Process-wide signal handling:** The startup callback registers `SIGINT` and `SIGTERM` handlers that call `shutdown_containers()` and an `atexit` cleanup hook. Triggered once per process when any subcommand runs.

### 2.1 `asqi validate`

`src/asqi/main.py:329` — Validate test plan configuration without execution. No DBOS, no Docker.

| Option | Short | Type | Default | Required | Help |
|---|---|---|---|---|---|
| `--test-suite-config` | `-t` | `str` | — | ✅ | Path to the test suite YAML file. |
| `--systems-config` | `-s` | `str` | — | ✅ | Path to the systems YAML file. |
| `--manifests-dir` | — | `str` | — | ✅ | Path to dir with test container manifests. |

**Dispatch:** `_validate_unique_ids(test_suite_config)` then `load_and_validate_plan(suite_path, systems_path, manifests_path)`.
**Exit codes:** `0` valid, `1` any validation failure (duplicate IDs, missing ID field, file not found, YAML parse error, schema violation, manifest cross-validation error).
**Stdout:** `[blue]--- Running Verification ---[/blue]`, `[green]✅ IDs verified[/green]`, then either `[green]✨ Success! The test plan is valid.[/green]` or `[red]❌ Test Plan Validation Failed:[/red]` followed by `  - <error>` per error line.

### 2.2 `asqi execute`

`src/asqi/main.py:366` — Execute end-to-end (tests + score cards). Requires Docker, requires DBOS.

| Option | Short | Type | Default | Range | Required | Help |
|---|---|---|---|---|---|---|
| `--test-suite-config` | `-t` | `str` | — | — | ✅ | Path to the test suite YAML file. |
| `--systems-config` | `-s` | `str` | — | — | ✅ | Path to the systems YAML file. |
| `--score-card-config` | `-r` | `str` | — | — | ✅ | Path to grading score card YAML file. |
| `--output-file` | `-o` | `str` | `output_scorecard.json` | — | — | Path to save execution results JSON file. |
| `--audit-responses` | `-a` | `str` | `None` | — | — | Path to YAML file with manual audit indicator responses. |
| `--skip-audit` | — | `bool` | `False` | — | — | Skip 'audit' type indicators if no audit responses provided. |
| `--concurrent-tests` | `-c` | `int` | `3` | 1..20 | — | Number of tests to run concurrently. |
| `--max-failures` | `-m` | `int` | `3` | 1..10 | — | Maximum number of failures to display. |
| `--progress-interval` | `-p` | `int` | `4` | 1..10 | — | Progress update interval. |
| `--container-config` | — | `str` | `None` | — | — | Optional path to container configuration YAML. |
| `--datasets-config` | `-d` | `str` | `None` | — | — | Path to the datasets registry YAML file. |

**Dispatch:** loads container config (or `ContainerConfig()`), launches DBOS, loads + audit-resolves the score card, calls `start_test_execution(..., execution_mode=ExecutionMode.END_TO_END, ...)`. (`start_test_execution` is an internal helper, not part of the public library contract — embed via `run_test_suite_workflow` + `evaluate_score_cards_workflow` directly.)
**Exit codes:** `0` success, `1` for any of: validation failure, `AuditResponsesRequiredError`, score-card load error, missing DBOS dependencies (`ImportError`), any `Exception` during execution.
**Audit logic:** if score card has `type: audit` indicators, exactly one of `--audit-responses` or `--skip-audit` must be supplied; passing both is rejected with exit 1.
**Side effects:** DBOS workflow registered + persisted to Postgres; Docker images pulled if missing; container logs written under `$LOGS_PATH`; final results JSON written to `--output-file`.

### 2.3 `asqi execute-tests`

`src/asqi/main.py:503` — Execute test suite only, skip score-card evaluation. Requires Docker, requires DBOS.

Same option set as `execute` **minus** `--score-card-config`, `--audit-responses`, `--skip-audit`, **plus**:

| Option | Short | Type | Default | Required | Help |
|---|---|---|---|---|---|
| `--output-file` | `-o` | `str` | `output.json` | — | (Note: different default than `execute`.) |
| `--test-ids` | `-tids` | `List[str]` | `None` | — | Filter the suite to run only these tests. **Despite the flag name, values are matched against each test's `name` field (case-insensitive), not its `id` field** (`workflow.py` filter loop). Accepts repeated flags (`-tids a -tids b`) and/or comma-separated values (`-tids a,b`). Unknown names raise `ValueError` with `difflib`-based "did you mean" suggestions. |

**Dispatch:** `start_test_execution(..., execution_mode=ExecutionMode.TESTS_ONLY, test_ids=test_ids, ...)`.
**Exit codes:** `0` / `1` as in `execute`.

### 2.4 `asqi generate-dataset`

`src/asqi/main.py:611` — Generate synthetic data via data-generation containers. Requires Docker, requires DBOS.

| Option | Short | Type | Default | Range | Required | Help |
|---|---|---|---|---|---|---|
| `--generation-config` | `-t` | `str` | — | — | ✅ | Path to the Generation YAML file. |
| `--systems-config` | `-s` | `str` | `None` | — | — | Path to the systems YAML file. |
| `--output-file` | `-o` | `str` | `output.json` | — | — | Path to save execution results JSON file. |
| `--concurrent-tests` | `-c` | `int` | `3` | 1..20 | — | (As above.) |
| `--max-failures` | `-m` | `int` | `3` | 1..10 | — | (As above.) |
| `--progress-interval` | `-p` | `int` | `4` | 1..10 | — | (As above.) |
| `--container-config` | — | `str` | `None` | — | — | (As above.) |
| `--datasets-config` | `-d` | `str` | `None` | — | — | (As above.) |

**Dispatch:** `start_data_generation(...)` (internal helper). Exit codes `0` / `1`.
**Note:** stdout banner currently reads `"--- 🚀 Executing Test Suite ---"` (cosmetic mismatch — the success banner correctly says `"Data Generation Completed"`). Treat the banner string as **not** part of the contract.

### 2.5 `asqi evaluate-score-cards`

`src/asqi/main.py:708` — Evaluate score cards against an existing test-results JSON file. Requires DBOS. **Does not run Docker containers.**

| Option | Short | Type | Default | Required | Help |
|---|---|---|---|---|---|
| `--input-file` | — | `str` | — | ✅ | Path to JSON file with existing test results. |
| `--score-card-config` | `-r` | `str` | — | ✅ | Path to grading score card YAML file. |
| `--output-file` | `-o` | `str` | `output_scorecard.json` | — | Path to save evaluation results JSON file. |
| `--audit-responses` | `-a` | `str` | `None` | — | (As above.) |
| `--skip-audit` | — | `bool` | `False` | — | (As above.) |

**Dispatch:** `start_score_card_evaluation(...)`.
**Container log sidecar lookup:** the writer (`execute-tests`) saves to `$LOGS_PATH/<basename(--output-file)>`, but the reader looks at `$LOGS_PATH/<--input-file as given>`. They only line up when both flags are bare filenames. With any directory prefix the sidecar is silently missing and evaluation proceeds with `test_container_data = []` — no warning.

**Exit codes:** `0` / `1` (validation, file errors, audit, DBOS launch, `Exception`).

---

## 3. Implicit contracts

### 3.1 Environment variables

| Name | Default | Required when | Consumed by | Purpose |
|---|---|---|---|---|
| `ASQI_LOG_LEVEL` | `INFO` | never | `asqi.logging_config.configure_logging()` (read when called, not at module import) | Log level for the `asqi.*` namespace. Callers that want the env var to take effect must call `configure_logging()` (the CLI does this on startup). |
| `DBOS_DATABASE_URL` | — | running any of `execute`, `execute-tests`, `generate-dataset`, `evaluate-score-cards`, or invoking any `*_workflow` function | `asqi.workflow` (import time, via DBOS init) | Postgres URL for DBOS workflow state. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | never | `asqi.workflow` (DBOS init) | OTEL collector endpoint. Tracing disabled if unset. |
| `LOGS_PATH` | `logs` | never | `asqi.workflow` (`save_container_results_to_file`) | Directory where container logs / per-run results JSON are written. |
| `HF_TOKEN` | unset | reading gated HuggingFace datasets without an inline token | `asqi.datasets` | HuggingFace API token. |
| `DOCKER_HOST` | auto-detect | Docker daemon not at the default socket | `asqi.container_manager` (host-side) and `asqi.workflow._configure_docker_in_docker` | Docker daemon socket for the host-side Docker SDK. **Stripped from the env passed into Docker-in-Docker containers** (those whose manifest declares `host_access: true`) so the container's Docker client falls back to the bind-mounted socket at `/var/run/docker.sock` instead of dialing the host's address (which isn't reachable inside the container's namespace). Untouched for non-DinD containers. |

asqi-engineer calls `dotenv.load_dotenv()` in `main.py` at module import time. Library callers must populate the environment themselves before importing `asqi.workflow`.

### 3.2 File conventions

- **Test container manifest:** `/app/manifest.yaml` inside every test image (`ContainerConfig.MANIFEST_PATH`, not configurable per-image). Schema: §1.3 `Manifest`.
- **Container input mount:** `/input` (read-only). Containers read datasets / configs from here using the in-container SDK (§1.6).
- **Container result JSON — stdout, *not* a file.** The host reads the structured `ContainerOutput` JSON (§1.6.2) by capturing the container's stdout (`container.logs()` in `container_manager.py`) and parsing the trailing JSON object via `parse_container_json_output` (§1.6.4). Container entrypoints emit the JSON with `print(json.dumps(...))`. Anything before the final `{...}` block (logs, progress lines, banners) is tolerated and ignored by the parser.
- **Container output mount:** `/output` (read-write). Used for *side artifacts* only — generated datasets and reports written by the container. The container references those files in its stdout JSON via `GeneratedDataset.dataset_path` / `GeneratedReport.report_path`; the host translates those container-internal paths to host paths after the run via `asqi.output._translate_container_path`. Nothing is required to be written to `/output`; tests that emit only metrics can leave it empty.
- **Test results JSON** (CLI `-o`): host-side aggregate; one entry per test result, plus a top-level `summary`. For `asqi execute`, also includes a top-level `score_card` key.
- **Container logs sidecar:** `$LOGS_PATH/<basename(--output-file)>` — the writer keeps the basename verbatim, **including** whatever extension the caller passed (no `.json` is appended). Read back during `evaluate-score-cards` to retrieve container outputs / errors. See §2.5 for the read/write path mismatch when `--output-file` / `--input-file` include a directory prefix.
- **YAML config files:** support env-var interpolation `${VAR}`, `${VAR:-default}`, `${VAR-default}` (see §3.4).

### 3.3 ID validation

The regex `^[0-9a-z_]{1,32}$` (`IDsStringPattern`) is enforced **only** on these schema fields:
- `TestDefinition.id` (`schemas.py:1140`)
- `ScoreCardIndicator.id` and `AuditScoreCardIndicator.id` (`schemas.py:1250`, `:1313`)

It is **not** enforced on:
- `SuiteConfig.suite_name`
- `ScoreCardFilter.test_id` (free-form `str`, even though it references a `TestDefinition.id`)
- `AuditResponse.indicator_id` (free-form `str`, even though it references an indicator id)
- System names (keys of `SystemsConfig.systems` — `dict[str, SystemConfig]`)

Callers cross-referencing these fields are responsible for ensuring values stay within the same character set.

`asqi.validation.validate_ids` (raises `DuplicateIDError` / `MissingIDFieldError`) checks **uniqueness**, not the regex. The CLI invokes it from most commands but **not all** — `generate-dataset` does not call it (`main.py:611`). The commands that do: `validate`, `execute`, `execute-tests`, `evaluate-score-cards`.

### 3.4 YAML environment-variable interpolation

`asqi.config.interpolate_env_vars` recursively walks loaded YAML and substitutes:

| Pattern | Behavior |
|---|---|
| `${VAR}` | Direct substitution; empty string if `VAR` is unset. |
| `${VAR:-default}` | `VAR` if set and non-empty, else `default`. |
| `${VAR-default}` | `VAR` if set (even empty), else `default`. |

Applied to both `load_yaml_file` (CLI) and `ContainerConfig.load_from_yaml`.

### 3.5 DBOS state assumptions

- DBOS schema is created on `DBOS.launch()`. Schema version compatibility is the caller's responsibility.
- Workflow IDs are UUIDs (36 chars). DBOS appends suffixes (e.g. `-1`, `-1-6`) for child workflows; callers needing the parent ID can truncate to the first 36 chars.
- Re-execution semantics: workflow durability assumes the same `DBOS_DATABASE_URL` between launches. Switching databases between runs orphans in-flight workflows.
- Per-workflow DBOS queues are created with deterministic names: `test_execution_<workflow_id>` and `data_generation_<workflow_id>`.

### 3.6 Container labeling

Every container started by asqi-engineer is labeled `workflow_id=<uuid>`. Callers can use this label for orphan-container cleanup during host-process shutdown.

### 3.7 Privileged mode escalation

When a manifest declares `host_access: true`, the calling application is responsible for setting `privileged=True` in `container_config.run_params` before invoking `run_test_suite_workflow`. asqi-engineer does **not** automatically escalate privilege based on the manifest field — privilege grants are an explicit decision of the embedding host.

---
