"""Contract tests for the host-side library workflow API.

Locks in the documented shapes, status values, defaults, and exception
propagation rules of:

- ``asqi.workflow.run_test_suite_workflow``
- ``asqi.workflow.evaluate_score_cards_workflow``
- ``asqi.workflow.run_data_generation_workflow``

These tests target the workflow contract, not internal behaviour. They run
against the *unwrapped* function (bypassing the ``@DBOS.workflow()`` decorator)
because the contract is the function shape callers see when DBOS has launched
the workflow on their behalf.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import patch

import pytest
from asqi.config import ContainerConfig, ExecutionMode, ExecutorConfig
from asqi.errors import MetricExpressionError, ReportValidationError
from asqi.schemas import Manifest, SystemInput
from asqi.workflow import (
    TestExecutionResult as _TestExecutionResult,
)
from asqi.workflow import (
    evaluate_score_cards_workflow,
    execute_data_generation,
    run_data_generation_workflow,
    run_test_suite_workflow,
)
from pydantic import ValidationError

# Aliased to avoid pytest collection of the class as if it were a test.
ExecResult = _TestExecutionResult


SUMMARY_REQUIRED_KEYS = {
    "suite_name",
    "workflow_id",
    "status",
    "total_tests",
    "successful_tests",
    "failed_tests",
    "success_rate",
    "total_execution_time",
    "timestamp",
}

PER_TEST_METADATA_KEYS = {
    "test_id",
    "test_name",
    "sut_name",
    "system_type",
    "image",
    "start_time",
    "end_time",
    "execution_time_seconds",
    "container_id",
    "exit_code",
    "timestamp",
    "success",
}

CONTAINER_RESULT_KEYS = {"test_id", "error_message", "container_output"}


def _executor_dict(
    *,
    concurrent_tests: int = ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
    max_failures: int = ExecutorConfig.MAX_FAILURES_DISPLAYED,
    progress_interval: int = ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
) -> dict[str, Any]:
    """Build the dict the workflow contract expects (NOT an ExecutorConfig)."""
    return {
        "concurrent_tests": concurrent_tests,
        "max_failures": max_failures,
        "progress_interval": progress_interval,
    }


def _unwrap(workflow_fn):
    return getattr(workflow_fn, "__wrapped__", workflow_fn)


class _StubHandle:
    """Minimal stand-in for a DBOS Queue.enqueue() handle."""

    def __init__(self, result):
        self._result = result

    def get_result(self):
        return self._result


def _call_test_suite(
    suite_config: dict[str, Any],
    systems_config: dict[str, Any],
    *,
    executor_config: dict[str, Any] | None = None,
    container_config: ContainerConfig | None = None,
    datasets_config: dict[str, Any] | None = None,
    score_card_configs: list[dict[str, Any]] | None = None,
    metadata_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _unwrap(run_test_suite_workflow)(
        suite_config,
        systems_config,
        executor_config or _executor_dict(),
        container_config or ContainerConfig(),
        datasets_config,
        score_card_configs,
        metadata_config,
    )


def _call_data_generation(
    generation_config: dict[str, Any],
    systems_config: dict[str, Any] | None,
    *,
    executor_config: dict[str, Any] | None = None,
    container_config: ContainerConfig | None = None,
    datasets_config: dict[str, Any] | None = None,
    metadata_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _unwrap(run_data_generation_workflow)(
        generation_config,
        systems_config,
        executor_config or _executor_dict(),
        container_config or ContainerConfig(),
        datasets_config,
        metadata_config,
    )


def _call_evaluate(
    test_results_data: dict[str, Any],
    test_container_data: list[dict[str, Any]],
    score_card_configs: list[dict[str, Any]],
    *,
    audit_responses_data: dict[str, Any] | None = None,
    execution_mode: ExecutionMode = ExecutionMode.END_TO_END,
):
    return _unwrap(evaluate_score_cards_workflow)(
        test_results_data,
        test_container_data,
        score_card_configs,
        audit_responses_data,
        execution_mode,
    )


@pytest.fixture(autouse=True)
def _isolate_dbos_env_and_queue(monkeypatch):
    """Don't let real DBOS env vars affect the tests, and never declare a real
    DBOS ``Queue`` (each declaration with the same name raises).

    The workflows construct their queue *before* config validation, so even
    early-return error paths attempt to register a Queue. Patching it here
    makes every test start clean."""
    monkeypatch.delenv("DBOS_DATABASE_URL", raising=False)
    monkeypatch.setenv("TESTING_DATABASE_URL", "sqlite://:memory:")
    with patch("asqi.workflow.Queue") as mock_queue_cls:
        mock_queue_cls.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(None)
        yield mock_queue_cls


def _valid_suite_config() -> dict[str, Any]:
    return {
        "suite_name": "demo",
        "test_suite": [
            {
                "name": "t1",
                "id": "t1",
                "image": "test/image:latest",
                "systems_under_test": ["systemA"],
                "params": {"p": "v"},
            }
        ],
    }


def _valid_systems_config() -> dict[str, Any]:
    return {
        "systems": {
            "systemA": {
                "type": "llm_api",
                "params": {"base_url": "http://x", "model": "x-model"},
            }
        }
    }


def _make_exec_result(
    test_id: str,
    *,
    success: bool,
    sut_name: str = "systemA",
    image: str = "test/image:latest",
) -> _TestExecutionResult:
    """Build a TestExecutionResult stand-in that the workflow can consume."""
    r = ExecResult(test_id, test_id, sut_name, image)
    r.start_time = 1.0
    r.end_time = 2.0
    r.exit_code = 0 if success else 1
    r.success = success
    r.container_id = "abc"
    r.test_results = {"success": success}
    return r


def _make_manifest() -> Manifest:
    return Manifest(
        name="mock",
        version="1",
        description="",
        input_systems=[SystemInput(name="system_under_test", type="llm_api", required=True)],
        input_schema=[],
        output_metrics=[],
        output_artifacts=None,
    )


# ---------------------------------------------------------------------------
# run_test_suite_workflow
# ---------------------------------------------------------------------------


class TestRunTestSuiteWorkflowContract:
    """``run_test_suite_workflow`` contract."""

    def test_happy_path_returns_documented_tuple_shape(self, _isolate_dbos_env_and_queue):
        """COMPLETED status, summary keys, per-test metadata, container_results."""
        success_result = ExecResult("t1", "t1", "systemA", "test/image:latest")
        success_result.start_time = 1.0
        success_result.end_time = 2.0
        success_result.exit_code = 0
        success_result.success = True
        success_result.container_id = "abc"
        success_result.test_results = {"success": True}

        # Override the autouse Queue mock to return our success result
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {
                        "system_under_test": {
                            "type": "llm_api",
                            "endpoint": "http://x",
                        }
                    },
                    "test_params": {"p": "v"},
                }
            ]

            test_results, container_results = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        # Top-level wrapper shape (§1.1, "test_results shape")
        assert set(test_results.keys()) == {"summary", "results"}
        assert SUMMARY_REQUIRED_KEYS.issubset(test_results["summary"].keys())
        assert test_results["summary"]["status"] == "COMPLETED"
        assert test_results["summary"]["total_tests"] == 1
        assert test_results["summary"]["successful_tests"] == 1
        assert test_results["summary"]["failed_tests"] == 0
        # success_rate = successful_tests / total_tests
        assert test_results["summary"]["success_rate"] == 1.0
        # COMPLETED-only conditional keys
        assert "images_checked" in test_results["summary"]
        assert "manifests_extracted" in test_results["summary"]

        # Per-test entry shape (§1.1, "Each entry in `results`")
        assert len(test_results["results"]) == 1
        entry = test_results["results"][0]
        assert set(entry.keys()) == {
            "metadata",
            "test_results",
            "generated_reports",
            "generated_datasets",
        }
        assert PER_TEST_METADATA_KEYS.issubset(entry["metadata"].keys())
        assert entry["test_results"] == {"success": True}
        assert entry["generated_reports"] == []
        assert entry["generated_datasets"] == []

        # container_results shape (§1.1, "container_results shape")
        assert len(container_results) == 1
        assert set(container_results[0].keys()) == CONTAINER_RESULT_KEYS

    def test_uses_test_results_key_not_results_key(self, _isolate_dbos_env_and_queue):
        """Distinguish from run_data_generation_workflow which uses 'results'."""
        success_result = ExecResult("t1", "t1", "systemA", "img")
        success_result.success = True
        success_result.test_results = {"success": True}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"img": True}
            mock_extract.return_value = {"img": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "img",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        entry = results["results"][0]
        assert "test_results" in entry
        assert "results" not in entry

    @pytest.mark.parametrize(
        "bad_suite,expected_status",
        [
            ({"suite_name": "x"}, "CONFIG_ERROR"),
            ({"suite_name": "x", "test_suite": "not-a-list"}, "CONFIG_ERROR"),
        ],
    )
    def test_config_error_does_not_raise(self, bad_suite, expected_status):
        """ValidationError / TypeError / AttributeError reshape to CONFIG_ERROR."""
        results, container_results = _call_test_suite(bad_suite, _valid_systems_config())
        assert results["summary"]["status"] == expected_status
        assert results["results"] == []
        assert container_results == []
        assert "error" in results["summary"]

    def test_volume_validation_failure_returns_validation_failed(self):
        """Volume validation errors surface as VALIDATION_FAILED (no raise)."""
        with patch("asqi.workflow.validate_test_volumes") as mock_vol:
            mock_vol.side_effect = ValueError("bad volume")
            results, container_results = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        assert results["summary"]["status"] == "VALIDATION_FAILED"
        assert results["results"] == []
        assert container_results == []
        assert "error" in results["summary"]

    def test_test_plan_validation_errors_populate_validation_errors(self):
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {}
            mock_validate.return_value = ["plan-error"]
            results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        assert results["summary"]["status"] == "VALIDATION_FAILED"
        assert results["summary"]["validation_errors"] == ["plan-error"]

    def test_no_tests_status_when_execution_plan_empty(self):
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            results, container_results = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        assert results["summary"]["status"] == "NO_TESTS"
        assert results["results"] == []
        assert container_results == []

    def test_unhandled_exception_propagates(self):
        """Exceptions outside (ValidationError, TypeError, ValueError, AttributeError)
        are NOT caught - they propagate to the caller (§1.1, 'Exceptions raised')."""

        class CustomDockerError(RuntimeError):
            pass

        with patch("asqi.workflow.dbos_check_images_availability") as mock_avail:
            mock_avail.side_effect = CustomDockerError("docker daemon dead")
            with pytest.raises(CustomDockerError):
                _call_test_suite(_valid_suite_config(), _valid_systems_config())

    @pytest.mark.parametrize(
        "exc_cls,exc_kwargs",
        [
            ("MissingImageError", {"args": ("img:1 not pullable",)}),
            ("MountExtractionError", {"args": ("bad mount args",)}),
        ],
        ids=["MissingImageError", "MountExtractionError"],
    )
    def test_documented_docker_layer_exceptions_propagate_uncaught(self, exc_cls, exc_kwargs):
        """Per §1.5: ``MissingImageError`` and ``MountExtractionError`` are
        documented as propagating uncaught from the Docker layer through
        ``run_test_suite_workflow``. The catch-set in the workflow only covers
        ``ValidationError`` / ``TypeError`` / ``AttributeError`` for config and
        ``ValueError`` for volume validation — these Docker-layer errors must
        bubble up so callers can branch on the documented class names.

        Pin per-class identity so a regression that wraps either error in a
        broader handler (e.g. catching ``Exception`` and reshaping to
        ``CONFIG_ERROR``) is caught immediately."""
        from asqi.errors import MissingImageError, MountExtractionError

        cls = {"MissingImageError": MissingImageError, "MountExtractionError": MountExtractionError}[exc_cls]
        with patch("asqi.workflow.dbos_check_images_availability") as mock_avail:
            mock_avail.side_effect = cls(*exc_kwargs["args"])
            with pytest.raises(cls):
                _call_test_suite(_valid_suite_config(), _valid_systems_config())

    @pytest.mark.parametrize(
        "exc_type",
        [ValidationError, TypeError, AttributeError],
    )
    def test_config_parsing_catch_set_reshapes_each_to_config_error(self, exc_type):
        """Per §1.1: 'During config parsing the function catches
        ``ValidationError``, ``TypeError``, ``ValueError``, and
        ``AttributeError`` and re-shapes them into a ``CONFIG_ERROR`` summary.'

        Pin each documented type individually so the catch boundary cannot
        silently narrow. We trigger each by patching ``SuiteConfig`` (the
        first call inside the config-parsing ``try``) to raise the exception
        we want to round-trip.

        Note: ``ValueError`` raised inside config parsing is *not* caught by
        the config-parsing handler — it surfaces later at
        ``validate_test_volumes`` and reshapes to ``VALIDATION_FAILED`` (see
        ``test_volume_validation_failure_returns_validation_failed``)."""
        if exc_type is ValidationError:
            from pydantic import BaseModel as _BM

            class _M(_BM):
                x: int

            try:
                _M(x="not-int")  # type: ignore[arg-type]
            except ValidationError as e:
                err: Exception = e
        else:
            err = exc_type("triggered")

        with patch("asqi.workflow.SuiteConfig", side_effect=err):
            results, container_results = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        assert results["summary"]["status"] == "CONFIG_ERROR"
        assert results["results"] == []
        assert container_results == []
        assert "error" in results["summary"]

    def test_value_error_during_volume_validation_reshapes_to_validation_failed(self):
        """Companion to the catch-set test above: ``ValueError`` is the
        documented type for volume-validation failure (§1.1, status
        ``VALIDATION_FAILED``). Pinning the asymmetric mapping prevents a
        future refactor from collapsing both error types into one status."""
        with patch("asqi.workflow.validate_test_volumes") as mock_vol:
            mock_vol.side_effect = ValueError("bad volume")
            results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())
        assert results["summary"]["status"] == "VALIDATION_FAILED"

    def test_default_optional_args_accepted(self):
        """``datasets_config`` / ``score_card_configs`` / ``metadata_config`` default to None."""
        results, _ = _unwrap(run_test_suite_workflow)(
            {"suite_name": "x"},
            _valid_systems_config(),
            _executor_dict(),
            ContainerConfig(),
        )
        assert results["summary"]["status"] == "CONFIG_ERROR"

    def test_success_rate_zero_when_total_tests_zero(self):
        """`success_rate` is 0.0 when total_tests==0 (§1.1, summary shape)."""
        results, _ = _call_test_suite({"suite_name": "x"}, _valid_systems_config())
        assert results["summary"]["success_rate"] == 0.0

    def test_success_rate_mid_range_with_mixed_pass_fail(self, _isolate_dbos_env_and_queue):
        """§1.1 summary shape: ``success_rate = successful_tests / total_tests``.

        Currently 0.0 (zero total) and 1.0 (all pass) cases are covered. Pin a
        mid-range case (1 of 2 succeeded → 0.5) so a regression that swaps the
        numerator/denominator (e.g. ``failed_tests / total_tests``) is caught
        on the only branch where they would diverge."""
        results_per_call = iter(
            [
                _make_exec_result("t1", success=True),
                _make_exec_result("t2", success=False),
            ]
        )
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(
            next(results_per_call)
        )

        suite = {
            "suite_name": "demo",
            "test_suite": [
                {
                    "name": "t1",
                    "id": "t1",
                    "image": "test/image:latest",
                    "systems_under_test": ["systemA"],
                    "params": {},
                },
                {
                    "name": "t2",
                    "id": "t2",
                    "image": "test/image:latest",
                    "systems_under_test": ["systemA"],
                    "params": {},
                },
            ],
        }

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                },
                {
                    "test_id": "t2",
                    "test_name": "t2",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                },
            ]
            test_results, _ = _call_test_suite(suite, _valid_systems_config())

        summary = test_results["summary"]
        assert summary["status"] == "COMPLETED"
        assert summary["total_tests"] == 2
        assert summary["successful_tests"] == 1
        assert summary["failed_tests"] == 1
        # Pin the mid-range value precisely — both numerator and denominator
        # must be correct to land on 0.5.
        assert summary["success_rate"] == 0.5

    def test_signature_matches_documented_layout(self):
        """Per §1.1: pin the documented 7-arg layout, parameter order, and
        which parameters carry ``Optional`` defaults. Mirrors the equivalent
        check on ``run_data_generation_workflow`` so a future regression that
        re-orders args, drops a parameter, or flips a default is caught."""
        from inspect import Parameter, signature

        params = signature(_unwrap(run_test_suite_workflow)).parameters
        # Documented order — preserves positional-call compatibility.
        assert list(params) == [
            "suite_config",
            "systems_config",
            "executor_config",
            "container_config",
            "datasets_config",
            "score_card_configs",
            "metadata_config",
        ]
        # Required (no default) — first four parameters.
        for required in (
            "suite_config",
            "systems_config",
            "executor_config",
            "container_config",
        ):
            assert params[required].default is Parameter.empty, f"§1.1 documents `{required}` as required (no default)"
        # Optional with default ``None`` — last three parameters.
        for optional in ("datasets_config", "score_card_configs", "metadata_config"):
            assert params[optional].default is None, f"§1.1 documents `{optional}` as Optional with default None"

    @pytest.mark.parametrize(
        "scenario,patches",
        [
            ("CONFIG_ERROR", None),
            (
                "VALIDATION_FAILED_volume",
                {"validate_test_volumes": ValueError("bad volume")},
            ),
            ("VALIDATION_FAILED_plan", "plan_errors"),
            ("NO_TESTS", "empty_plan"),
        ],
    )
    def test_images_checked_and_manifests_extracted_absent_on_non_completed(self, scenario, patches):
        """§1.1 summary shape: ``images_checked`` and ``manifests_extracted``
        are documented as 'happy path (status=COMPLETED) only'. A regression
        that always emits them (or that drops them from COMPLETED) would slip
        through unless the absent-on-error branches are pinned."""
        if scenario == "CONFIG_ERROR":
            results, _ = _call_test_suite({}, _valid_systems_config())
        elif scenario == "VALIDATION_FAILED_volume":
            with patch("asqi.workflow.validate_test_volumes") as mock_vol:
                mock_vol.side_effect = patches["validate_test_volumes"]
                results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())
        elif scenario == "VALIDATION_FAILED_plan":
            with (
                patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
                patch("asqi.workflow.extract_manifests_step") as mock_extract,
                patch("asqi.workflow.validate_test_plan") as mock_validate,
            ):
                mock_avail.return_value = {"test/image:latest": True}
                mock_extract.return_value = {}
                mock_validate.return_value = ["plan-error"]
                results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())
        else:  # NO_TESTS
            with (
                patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
                patch("asqi.workflow.extract_manifests_step") as mock_extract,
                patch("asqi.workflow.validate_test_plan") as mock_validate,
                patch("asqi.workflow.create_test_execution_plan") as mock_plan,
            ):
                mock_avail.return_value = {"test/image:latest": True}
                mock_extract.return_value = {"test/image:latest": _make_manifest()}
                mock_validate.return_value = []
                mock_plan.return_value = []
                results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        summary = results["summary"]
        assert summary["status"] != "COMPLETED"
        # Both keys must be ABSENT on non-COMPLETED returns.
        assert "images_checked" not in summary
        assert "manifests_extracted" not in summary

    def test_container_config_parameter_annotation_is_container_config(self):
        """Per §1.1 parameter table: ``container_config`` is documented as a
        ``ContainerConfig`` instance, **not** a dict. Pin the annotation so a
        future signature change to ``Dict[str, Any]`` (which would silently
        accept the wrong shape) is caught."""
        from inspect import signature

        params = signature(_unwrap(run_test_suite_workflow)).parameters
        # The annotation may be a class or a string (forward ref).
        ann = params["container_config"].annotation
        assert ann is ContainerConfig or ann == "ContainerConfig", (
            f"§1.1 documents container_config as ContainerConfig, got {ann!r}"
        )

    def test_container_results_aligned_with_results_by_index_and_test_id(self, _isolate_dbos_env_and_queue):
        """§1.1 ``container_results shape``: 'one entry per test, in the same
        order as test_results.results'. Pin the index-by-index alignment AND
        the per-entry ``test_id`` correspondence so a regression that re-orders
        one list relative to the other is caught."""
        results_per_call = iter(
            [
                _make_exec_result("alpha", success=True),
                _make_exec_result("beta", success=False),
                _make_exec_result("gamma", success=True),
            ]
        )
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(
            next(results_per_call)
        )

        suite = {
            "suite_name": "demo",
            "test_suite": [
                {
                    "name": tid,
                    "id": tid,
                    "image": "test/image:latest",
                    "systems_under_test": ["systemA"],
                    "params": {},
                }
                for tid in ("alpha", "beta", "gamma")
            ],
        }

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": tid,
                    "test_name": tid,
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
                for tid in ("alpha", "beta", "gamma")
            ]
            test_results, container_results = _call_test_suite(suite, _valid_systems_config())

        results_list = test_results["results"]
        assert len(results_list) == len(container_results) == 3
        for results_entry, container_entry in zip(results_list, container_results, strict=True):
            assert results_entry["metadata"]["test_id"] == container_entry["test_id"]

    def test_container_never_started_metadata_sentinels(self, _isolate_dbos_env_and_queue):
        """§1.1 metadata table: ``container_id`` is empty string and
        ``exit_code`` is -1 'when container never started'. Callers branch on
        these sentinels to distinguish a failed-to-launch container from a
        container that ran and exited non-zero."""
        # A TestExecutionResult with the documented "never started" defaults:
        # success=False, container_id="", exit_code=-1 (the constructor
        # defaults — we only set the timing fields the workflow consumes).
        never_started = ExecResult("t1", "t1", "systemA", "test/image:latest")
        never_started.start_time = 1.0
        never_started.end_time = 2.0
        never_started.error_message = "image pull failed"
        # Sanity: the constructor leaves these at the documented sentinels.
        assert never_started.container_id == ""
        assert never_started.exit_code == -1
        assert never_started.success is False

        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(never_started)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            test_results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        metadata = test_results["results"][0]["metadata"]
        assert metadata["container_id"] == ""
        assert metadata["exit_code"] == -1
        assert metadata["success"] is False

    def test_container_results_error_message_empty_when_success_true(self, _isolate_dbos_env_and_queue):
        """§1.1 container_results shape: ``error_message`` is empty string when
        success=true. Pin so a regression that emits ``None`` (or stringifies
        a stub object) on the success path is caught."""
        success = _make_exec_result("t1", success=True)
        # Sanity: the default error_message is "" on a successful result.
        assert success.error_message == ""

        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            test_results, container_results = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        # success path
        assert test_results["results"][0]["metadata"]["success"] is True
        assert container_results[0]["error_message"] == ""

    def test_summary_and_metadata_timestamps_are_iso8601_strings(self, _isolate_dbos_env_and_queue):
        """§1.1: ``summary.timestamp`` and ``metadata.timestamp`` are
        documented as ISO 8601 strings; ``metadata.start_time`` /
        ``metadata.end_time`` are documented as epoch seconds (floats)."""
        from datetime import datetime

        success = _make_exec_result("t1", success=True)
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            test_results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        summary = test_results["summary"]
        metadata = test_results["results"][0]["metadata"]

        # ISO 8601 round-trips through fromisoformat without raising.
        assert isinstance(summary["timestamp"], str)
        datetime.fromisoformat(summary["timestamp"])
        assert isinstance(metadata["timestamp"], str)
        datetime.fromisoformat(metadata["timestamp"])

        # Epoch seconds — floats, not strings.
        assert isinstance(metadata["start_time"], float)
        assert isinstance(metadata["end_time"], float)
        # Sanity: end >= start.
        assert metadata["end_time"] >= metadata["start_time"]

    def test_generated_reports_and_datasets_round_trip_into_per_test_entry(self, _isolate_dbos_env_and_queue):
        """§1.1: per-test ``generated_reports`` and ``generated_datasets`` are
        documented as 'list of GeneratedReport / GeneratedDataset dicts
        (model_dump)'. Currently only the empty-list case is exercised — pin
        the populated arm so a regression that drops the model_dump step (or
        forgets to forward the lists) is caught."""
        from asqi.response_schemas import GeneratedDataset, GeneratedReport

        result = _make_exec_result("t1", success=True)
        result.generated_reports = [
            GeneratedReport(
                report_name="summary",
                report_type="html",
                report_path="/output/summary.html",
            )
        ]
        result.generated_datasets = [
            GeneratedDataset(
                dataset_name="rows",
                dataset_type="huggingface",
                dataset_path="/output/rows",
            )
        ]
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(result)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "test/image:latest",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            test_results, _ = _call_test_suite(_valid_suite_config(), _valid_systems_config())

        entry = test_results["results"][0]
        # Reports/datasets are surfaced as dicts (model_dump output), not
        # Pydantic instances — that's the documented serialization shape.
        assert len(entry["generated_reports"]) == 1
        assert isinstance(entry["generated_reports"][0], dict)
        assert entry["generated_reports"][0]["report_name"] == "summary"
        assert entry["generated_reports"][0]["report_type"] == "html"
        assert entry["generated_reports"][0]["report_path"] == "/output/summary.html"

        assert len(entry["generated_datasets"]) == 1
        assert isinstance(entry["generated_datasets"][0], dict)
        assert entry["generated_datasets"][0]["dataset_name"] == "rows"
        assert entry["generated_datasets"][0]["dataset_type"] == "huggingface"
        assert entry["generated_datasets"][0]["dataset_path"] == "/output/rows"


# ---------------------------------------------------------------------------
# evaluate_score_cards_workflow
# ---------------------------------------------------------------------------


class TestEvaluateScoreCardsWorkflowContract:
    """§1.1 ``evaluate_score_cards_workflow`` contract."""

    def test_happy_path_returns_input_with_score_card_key(self):
        test_results_data: dict[str, Any] = {
            "summary": {"status": "COMPLETED"},
            "results": [],
        }
        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card") as mock_eval,
        ):
            mock_convert.return_value = []
            mock_eval.return_value = [{"score_card_name": "Test"}]
            result = _call_evaluate(test_results_data, [], [{"score_card_name": "Test"}])

        assert "score_card" in result
        assert result["summary"]["status"] == "COMPLETED"

    def test_audit_indicators_without_responses_does_not_raise_here(self):
        """Per §1.1 / §1.5: missing audit responses produce per-indicator error
        entries inside the workflow (not an exception). The
        ``AuditResponsesRequiredError`` is raised by the CLI helper, not by
        this workflow."""
        score_card_with_audit = {
            "score_card_name": "Audit-only",
            "indicators": [
                {
                    "id": "a1",
                    "type": "audit",
                    "name": "First",
                    "assessment": [
                        {"outcome": "A", "description": "ok"},
                        {"outcome": "B", "description": "bad"},
                    ],
                }
            ],
        }
        result = _call_evaluate(
            {"summary": {"status": "COMPLETED"}, "results": []},
            [],
            [score_card_with_audit],
            audit_responses_data=None,
        )
        assert "score_card" in result

    def test_metric_expression_error_propagates(self):
        with patch("asqi.workflow.evaluate_score_card") as mock_eval:
            mock_eval.side_effect = MetricExpressionError("bad expr")
            with pytest.raises(MetricExpressionError):
                _call_evaluate(
                    {"summary": {"status": "COMPLETED"}, "results": []},
                    [],
                    [{"score_card_name": "Test"}],
                )

    def test_report_validation_error_caught_inside_evaluate_score_card(self):
        """Per §1.1 / §1.5: ``ReportValidationError`` is **caught** inside
        ``evaluate_score_card`` (workflow.py:923) alongside KeyError /
        AttributeError / TypeError / ValueError and converted into a
        per-indicator error entry — it does not propagate.

        We exercise the real ``evaluate_score_card`` (not mocked) to prove the
        catch-and-convert behaviour. ``validate_display_reports`` is the call
        that raises ``ReportValidationError`` in EVALUATE_ONLY mode."""
        from asqi.workflow import evaluate_score_card

        with patch(
            "asqi.workflow.validate_display_reports",
            side_effect=ReportValidationError(["report 'r1' not declared"]),
        ):
            evaluations = evaluate_score_card(
                test_results=[],
                score_card_configs=[
                    {
                        "score_card_name": "ReportCard",
                        "indicators": [
                            {
                                "id": "ind_1",
                                "apply_to": {"test_id": "some_test"},
                                "metric": "m",
                                "assessment": [
                                    {
                                        "outcome": "PASS",
                                        "condition": "greater_equal",
                                        "threshold": 0.0,
                                    }
                                ],
                                "display_reports": ["r1"],
                            }
                        ],
                    }
                ],
                audit_responses_data=None,
                execution_mode=ExecutionMode.EVALUATE_ONLY,
            )

        # The exception is converted into a single error entry, not raised.
        assert len(evaluations) == 1
        entry = evaluations[0]
        assert entry["score_card_name"] == "ReportCard"
        assert entry["indicator_id"] == "score_card_evaluation_error"
        assert "Score card evaluation error" in entry["error"]

    def test_returns_input_dict_with_score_card_key_added(self):
        """§1.1: 'the input ``test_results_data`` dict copied with an added
        ``score_card`` key carrying the evaluation output.'

        Pin both halves of the contract:
        - the returned dict carries ``score_card`` (positive shape), AND
        - the input dict the caller passed in is NOT mutated in place
          (the docstring says 'copied'; an in-place mutation regression
          would silently break callers that hold onto the input).
        """
        original_input = {
            "summary": {"status": "COMPLETED"},
            "results": [],
        }
        # Snapshot the input shape pre-call so we can compare after.
        snapshot_keys = set(original_input.keys())

        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card") as mock_eval,
        ):
            mock_convert.return_value = []
            mock_eval.return_value = [{"score_card_name": "Test", "indicator_id": "ind_1"}]
            result = _call_evaluate(original_input, [], [{"score_card_name": "Test"}])

        # Returned dict carries the new key.
        assert "score_card" in result
        # Input dict was NOT mutated — its key set is unchanged, and
        # ``score_card`` did not appear on it.
        assert set(original_input.keys()) == snapshot_keys
        assert "score_card" not in original_input

    def test_default_execution_mode_is_end_to_end(self):
        """`execution_mode` defaults to END_TO_END per §1.1 signature."""
        captured = {}

        def fake_eval(test_results, score_cards, audit_data, mode):
            captured["mode"] = mode
            return []

        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card", side_effect=fake_eval),
        ):
            mock_convert.return_value = []
            _unwrap(evaluate_score_cards_workflow)(
                {"summary": {"status": "COMPLETED"}, "results": []},
                [],
                [{"score_card_name": "Test"}],
            )

        assert captured["mode"] == ExecutionMode.END_TO_END


# ---------------------------------------------------------------------------
# run_data_generation_workflow
# ---------------------------------------------------------------------------


class TestRunDataGenerationWorkflowContract:
    """§1.1 ``run_data_generation_workflow`` contract."""

    def _gen_config(self) -> dict[str, Any]:
        return {
            "job_name": "gen-job",
            "generation_jobs": [
                {
                    "id": "g1",
                    "name": "g1",
                    "image": "gen/image:latest",
                    "params": {},
                }
            ],
        }

    def test_systems_config_is_optional(self):
        """`systems_config` is Optional for data generation (§1.1, parameter notes).

        Pass None and confirm the workflow accepts the call without raising
        at the config-parsing step. We mock the downstream docker-touching
        steps so the test is self-contained."""
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            results, container_results = _call_data_generation(self._gen_config(), systems_config=None)
        assert isinstance(results, dict)
        assert "summary" in results
        assert isinstance(container_results, list)

    def test_validation_error_returns_config_error(self):
        results, container_results = _call_data_generation({"not-a-valid": "config"}, systems_config=None)
        assert results["summary"]["status"] == "CONFIG_ERROR"
        assert results["results"] == []
        assert container_results == []
        assert "error" in results["summary"]

    def test_no_tests_status_when_no_jobs_to_run(self):
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            results, container_results = _call_data_generation(self._gen_config(), systems_config=None)
        assert results["summary"]["status"] == "NO_TESTS"
        assert results["results"] == []
        assert container_results == []

    def test_per_job_results_use_results_key_not_test_results_key(self, _isolate_dbos_env_and_queue):
        """Differs from run_test_suite_workflow: data generation emits 'results'."""
        success_result = ExecResult("g1", "g1", None, "gen/image:latest")
        success_result.success = True
        success_result.results = {"success": True, "produced": 10}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "job_name": "g1",
                    "job_id": "g1",
                    "image": "gen/image:latest",
                    "systems_params": {},
                    "generation_params": {},
                }
            ]
            results, _ = _call_data_generation(self._gen_config(), systems_config=None)

        assert results["summary"]["status"] == "COMPLETED"
        entry = results["results"][0]
        assert "results" in entry
        assert "test_results" not in entry
        assert entry["results"] == {"success": True, "produced": 10}

    def test_queue_name_uses_data_generation_prefix(self, _isolate_dbos_env_and_queue):
        """Per §1.1 side effects: queue is ``data_generation_<workflow_id>``."""
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            _call_data_generation(self._gen_config(), systems_config=None)

        # First positional arg of Queue() is the queue name
        queue_args, _ = _isolate_dbos_env_and_queue.call_args
        assert queue_args[0].startswith("data_generation_")

    def test_signature_does_not_accept_score_card_configs(self):
        """Per §1.1 parameter notes: 'score_card_configs is **not** a parameter —
        score cards are not applicable to data generation.'

        Pin the documented absence at the signature level so a future
        regression that re-adds the parameter is caught."""
        from inspect import signature

        params = signature(_unwrap(run_data_generation_workflow)).parameters
        assert "score_card_configs" not in params
        # Sanity: the documented parameters ARE present.
        for documented_param in (
            "generation_config",
            "systems_config",
            "executor_config",
            "container_config",
            "datasets_config",
            "metadata_config",
        ):
            assert documented_param in params, f"§1.1 documents `{documented_param}` as a parameter"

    def test_metadata_config_forwarded_into_per_job_enqueue(self, _isolate_dbos_env_and_queue):
        """Per §1.1: 'metadata_config accepts the same keys as in
        run_test_suite_workflow (parent_id, job_type, user_id).' Confirm the
        dict reaches the per-job enqueue call so it can be forwarded into
        generation containers (workflow.py:2178-2189)."""
        success_result = ExecResult("g1", "g1", None, "gen/image:latest")
        success_result.success = True
        success_result.results = {"success": True}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        meta = {"parent_id": "parent-gen-1", "job_type": "data_gen", "user_id": "u-9"}

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "job_name": "g1",
                    "job_id": "g1",
                    "image": "gen/image:latest",
                    "systems_params": {},
                    "generation_params": {},
                }
            ]
            _call_data_generation(
                {
                    "job_name": "gen-job",
                    "generation_jobs": [
                        {
                            "id": "g1",
                            "name": "g1",
                            "image": "gen/image:latest",
                            "params": {},
                        }
                    ],
                },
                systems_config=None,
                metadata_config=meta,
            )

        enqueue_calls = _isolate_dbos_env_and_queue.return_value.enqueue.call_args_list
        assert len(enqueue_calls) == 1
        args = enqueue_calls[0].args
        # Bind captured positional args to the target function's signature so
        # the assertion is robust to argument-order changes in either the
        # enqueue call site or the function definition.
        bound = inspect.signature(execute_data_generation).bind(*args[1:])
        assert bound.arguments["metadata_config"] == meta
        assert bound.arguments["parent_id"] == "parent-gen-1"

    def test_test_suite_queue_name_uses_test_execution_prefix(self, _isolate_dbos_env_and_queue):
        """Symmetric check: run_test_suite_workflow uses ``test_execution_<id>``."""
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            _call_test_suite(_valid_suite_config(), _valid_systems_config())

        queue_args, _ = _isolate_dbos_env_and_queue.call_args
        assert queue_args[0].startswith("test_execution_")

    def test_data_generation_plan_validation_errors_populate_validation_errors(self):
        """§1.1: ``run_data_generation_workflow`` mirrors the test-suite workflow's
        ``VALIDATION_FAILED`` semantics — when ``validate_data_generation_plan``
        returns a non-empty error list, the workflow returns a CONFIG_ERROR-shaped
        dict with status ``VALIDATION_FAILED`` and ``validation_errors`` populated.

        Pinning this independently of ``run_test_suite_workflow`` so a regression
        that breaks status mapping for data generation specifically is caught."""
        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = ["gen-plan-error-1", "gen-plan-error-2"]
            results, container_results = _call_data_generation(self._gen_config(), systems_config=None)

        assert results["summary"]["status"] == "VALIDATION_FAILED"
        assert results["summary"]["validation_errors"] == [
            "gen-plan-error-1",
            "gen-plan-error-2",
        ]
        assert results["results"] == []
        assert container_results == []

    def test_summary_suite_name_carries_generation_job_name(self, _isolate_dbos_env_and_queue):
        """§1.1 data-gen return shape: ``summary.suite_name`` carries
        ``generation.job_name`` (the field name is reused from the test-suite
        wrapper for symmetry; the value is the data-generation job name).

        Pin this mapping so a regression that wires e.g. an empty string or the
        first job's name into the summary is caught."""
        success_result = ExecResult("g1", "g1", None, "gen/image:latest")
        success_result.success = True
        success_result.results = {"success": True}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "job_name": "g1",
                    "job_id": "g1",
                    "image": "gen/image:latest",
                    "systems_params": {},
                    "generation_params": {},
                }
            ]
            gen_config = {
                "job_name": "my-gen-job-name",
                "generation_jobs": [
                    {
                        "id": "g1",
                        "name": "g1",
                        "image": "gen/image:latest",
                        "params": {},
                    }
                ],
            }
            results, _ = _call_data_generation(gen_config, systems_config=None)

        assert results["summary"]["suite_name"] == "my-gen-job-name"

    @pytest.mark.parametrize(
        "exc_cls_name",
        ["MissingImageError", "MountExtractionError"],
    )
    def test_documented_docker_layer_exceptions_propagate_uncaught(self, exc_cls_name):
        """Per §1.5: ``MissingImageError`` and ``MountExtractionError`` are
        documented as propagating uncaught from the Docker layer through
        ``run_data_generation_workflow`` (mirroring ``run_test_suite_workflow``).

        Pinning each class on the data-generation workflow specifically so a
        regression that adds a broad ``except Exception`` to one workflow but
        not the other (or vice versa) is caught."""
        from asqi.errors import MissingImageError, MountExtractionError

        cls = {
            "MissingImageError": MissingImageError,
            "MountExtractionError": MountExtractionError,
        }[exc_cls_name]
        with patch("asqi.workflow.dbos_check_images_availability") as mock_avail:
            mock_avail.side_effect = cls("docker layer failure")
            with pytest.raises(cls):
                _call_data_generation(self._gen_config(), systems_config=None)

    def test_container_results_aligned_with_results_by_index_and_test_id(self, _isolate_dbos_env_and_queue):
        """§1.1 data-gen ``container_results`` shape: 'one entry per
        generation job, in the same order as generation_results.results'.
        Mirrors the test-suite ordering invariant on a different workflow
        so a regression that breaks index alignment for one but not the
        other is caught."""
        gens = []
        for i in range(3):
            r = _make_exec_result(f"g{i}", success=True, sut_name=None, image="gen/image:latest")
            r.results = {"success": True}
            gens.append(r)
        gens_iter = iter(gens)
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(next(gens_iter))

        gen_config = {
            "job_name": "gen-job",
            "generation_jobs": [
                {
                    "id": f"g{i}",
                    "name": f"g{i}",
                    "image": "gen/image:latest",
                    "params": {},
                }
                for i in range(3)
            ],
        }

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "job_name": f"g{i}",
                    "job_id": f"g{i}",
                    "image": "gen/image:latest",
                    "systems_params": {},
                    "generation_params": {},
                }
                for i in range(3)
            ]
            generation_results, container_results = _call_data_generation(gen_config, systems_config=None)

        results_list = generation_results["results"]
        assert len(results_list) == len(container_results) == 3
        for results_entry, container_entry in zip(results_list, container_results, strict=True):
            assert results_entry["metadata"]["test_id"] == container_entry["test_id"]


# ---------------------------------------------------------------------------
# Cross-workflow shared invariants
# ---------------------------------------------------------------------------


class TestSharedSummaryInvariants:
    """Invariants documented as common to both workflow summaries."""

    def test_status_value_set_for_test_suite(self):
        """Documented status values: CONFIG_ERROR / VALIDATION_FAILED / NO_TESTS / COMPLETED."""
        results, _ = _call_test_suite({}, _valid_systems_config())
        assert results["summary"]["status"] in {
            "CONFIG_ERROR",
            "VALIDATION_FAILED",
            "NO_TESTS",
            "COMPLETED",
        }


# ---------------------------------------------------------------------------
# metadata_config / audit_responses_data forwarding (§1.1)
# ---------------------------------------------------------------------------


class TestMetadataConfigForwarding:
    """§1.1 documents ``metadata_config`` as 'Forwarded into test containers as
    metadata. Common keys: parent_id, job_type, user_id'. We pin that the
    workflow actually hands the dict (and the parent_id derived from it) into
    the per-test enqueue call, where it eventually reaches the container."""

    def test_metadata_config_passed_into_enqueue(self, _isolate_dbos_env_and_queue):
        """The dict the caller passes is forwarded verbatim as the 10th
        positional arg to ``Queue.enqueue`` (see workflow.py:1128-1141)."""
        success_result = ExecResult("t1", "t1", "systemA", "img:1")
        success_result.success = True
        success_result.test_results = {"success": True}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        meta = {"parent_id": "parent-wf-1", "job_type": "test", "user_id": "u-1"}

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"img:1": True}
            mock_extract.return_value = {"img:1": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "img:1",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            _call_test_suite(
                _valid_suite_config(),
                _valid_systems_config(),
                metadata_config=meta,
            )

        # enqueue is called once per test plan entry
        enqueue_calls = _isolate_dbos_env_and_queue.return_value.enqueue.call_args_list
        assert len(enqueue_calls) == 1
        args = enqueue_calls[0].args
        # Layout: (execute_single_test, test_name, test_id, image, sut_name,
        # systems_params, test_params, container_config, env_file, environment,
        # metadata_config, parent_id) — metadata_config is index 10.
        assert args[10] == meta
        # parent_id is derived from metadata_config["parent_id"] when present
        assert args[11] == "parent-wf-1"

    def test_metadata_config_none_passed_through(self, _isolate_dbos_env_and_queue):
        """When caller omits metadata_config, the workflow forwards ``None``
        and the parent_id arg is also ``None`` (workflow.py:1140)."""
        success_result = ExecResult("t1", "t1", "systemA", "img:1")
        success_result.success = True
        success_result.test_results = {"success": True}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"img:1": True}
            mock_extract.return_value = {"img:1": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "img:1",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            _call_test_suite(_valid_suite_config(), _valid_systems_config())

        args = _isolate_dbos_env_and_queue.return_value.enqueue.call_args.args
        assert args[10] is None
        assert args[11] is None

    @pytest.mark.parametrize(
        "documented_key,value",
        [
            ("parent_id", "parent-wf-xyz"),
            ("job_type", "test"),
            ("user_id", "user-42"),
        ],
    )
    def test_each_documented_metadata_key_is_preserved(self, _isolate_dbos_env_and_queue, documented_key, value):
        """§1.1 documents three common metadata_config keys: ``parent_id``,
        ``job_type``, ``user_id``. Each must round-trip through the workflow
        unchanged when supplied alone — a partial-forwarding regression that
        drops one key would silently break the documented surface."""
        success_result = ExecResult("t1", "t1", "systemA", "img:1")
        success_result.success = True
        success_result.test_results = {"success": True}
        _isolate_dbos_env_and_queue.return_value.enqueue.side_effect = lambda *a, **kw: _StubHandle(success_result)

        meta = {documented_key: value}

        with (
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_avail.return_value = {"img:1": True}
            mock_extract.return_value = {"img:1": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = [
                {
                    "test_id": "t1",
                    "test_name": "t1",
                    "image": "img:1",
                    "sut_name": "systemA",
                    "systems_params": {"system_under_test": {"type": "llm_api"}},
                    "test_params": {},
                }
            ]
            _call_test_suite(
                _valid_suite_config(),
                _valid_systems_config(),
                metadata_config=meta,
            )

        args = _isolate_dbos_env_and_queue.return_value.enqueue.call_args.args
        # The dict reaches enqueue with the documented key intact.
        assert args[10] == meta
        assert args[10][documented_key] == value
        # Only the parent_id key drives the dedicated parent_id positional
        # (workflow.py:1140); the other two documented keys must NOT be
        # mistakenly extracted as parent_id.
        if documented_key == "parent_id":
            assert args[11] == value
        else:
            assert args[11] is None


class TestAuditResponsesDataThreading:
    """§1.1 / §1.5: ``audit_responses_data`` is threaded through to
    ``evaluate_score_card`` so per-indicator audit responses can be applied.
    The workflow itself does not raise ``AuditResponsesRequiredError``; that is
    the CLI's job. Inside the workflow, missing responses produce per-indicator
    error entries — but the data we *do* pass must reach the evaluator."""

    def test_audit_responses_data_reaches_evaluate_score_card(self):
        captured = {}

        def fake_eval(test_results, score_cards, audit_data, mode):
            captured["audit_data"] = audit_data
            captured["mode"] = mode
            return []

        audit_payload = {"responses": [{"indicator_id": "ind_1", "selected_outcome": "A", "notes": ""}]}
        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card", side_effect=fake_eval),
        ):
            mock_convert.return_value = []
            _call_evaluate(
                {"summary": {"status": "COMPLETED"}, "results": []},
                [],
                [{"score_card_name": "Test"}],
                audit_responses_data=audit_payload,
            )

        assert captured["audit_data"] == audit_payload

    def test_default_audit_responses_data_is_none(self):
        """Per §1.1 signature: ``audit_responses_data`` defaults to ``None`` —
        and that ``None`` is what reaches ``evaluate_score_card``."""
        captured = {}

        def fake_eval(test_results, score_cards, audit_data, mode):
            captured["audit_data"] = audit_data
            return []

        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card", side_effect=fake_eval),
        ):
            mock_convert.return_value = []
            _unwrap(evaluate_score_cards_workflow)(
                {"summary": {"status": "COMPLETED"}, "results": []},
                [],
                [{"score_card_name": "Test"}],
            )

        assert captured["audit_data"] is None


class TestExecutionModePropagation:
    """§1.1: ``ExecutionMode.EVALUATE_ONLY`` and ``END_TO_END`` are both
    documented as expected values for ``evaluate_score_cards_workflow``."""

    @pytest.mark.parametrize(
        "mode",
        [ExecutionMode.EVALUATE_ONLY, ExecutionMode.END_TO_END],
    )
    def test_explicit_mode_propagates_to_evaluate_score_card(self, mode):
        captured = {}

        def fake_eval(test_results, score_cards, audit_data, m):
            captured["mode"] = m
            return []

        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card", side_effect=fake_eval),
        ):
            mock_convert.return_value = []
            _call_evaluate(
                {"summary": {"status": "COMPLETED"}, "results": []},
                [],
                [{"score_card_name": "Test"}],
                execution_mode=mode,
            )

        assert captured["mode"] == mode


class TestScoreCardKeyOutputShape:
    """§3.2 / §1.1: ``execute`` writes a results JSON whose top-level structure
    is ``{summary, results, score_card}``. The wrapping happens in
    ``add_score_cards_to_results``; this test pins both the helper and the
    workflow integration."""

    def test_evaluate_workflow_adds_score_card_key(self):
        """The single value of ``add_score_cards_to_results`` is to attach a
        ``score_card`` key to the input dict. After
        ``evaluate_score_cards_workflow`` runs, the returned dict carries it."""
        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card") as mock_eval,
        ):
            mock_convert.return_value = []
            mock_eval.return_value = [
                {
                    "score_card_name": "Test",
                    "indicator_id": "ind_1",
                    "outcome": "A",
                }
            ]
            result = _call_evaluate(
                {"summary": {"status": "COMPLETED"}, "results": []},
                [],
                [{"score_card_name": "Test"}],
            )

        # Top-level shape: input keys + new score_card key.
        assert "summary" in result
        assert "results" in result
        assert "score_card" in result
        # Single score-card form: dict (not list) per add_score_cards_to_results
        assert result["score_card"]["score_card_name"] == "Test"
        assert result["score_card"]["total_evaluations"] == 1
        # Per-evaluation entries strip the score_card_name key (it is hoisted)
        assert result["score_card"]["assessments"] == [{"indicator_id": "ind_1", "outcome": "A"}]

    def test_score_card_is_none_when_no_evaluations(self):
        """Per ``add_score_cards_to_results``: when ``score_card_evaluation``
        is empty, the ``score_card`` key is set to ``None`` (not omitted)."""
        with (
            patch("asqi.workflow.convert_test_results_to_objects") as mock_convert,
            patch("asqi.workflow.evaluate_score_card") as mock_eval,
        ):
            mock_convert.return_value = []
            mock_eval.return_value = []  # no evaluations
            result = _call_evaluate(
                {"summary": {"status": "COMPLETED"}, "results": []},
                [],
                [],
            )
        assert "score_card" in result
        assert result["score_card"] is None


class TestWorkflowIdUuidFormat:
    """§3.5: 'Workflow IDs are UUIDs (36 chars).' Per-workflow DBOS queues
    are named ``test_execution_<workflow_id>`` / ``data_generation_<workflow_id>``.
    With a real-looking workflow_id, the queue name suffix should be exactly
    36 chars after the prefix."""

    def test_test_suite_queue_embeds_36_char_uuid(self, _isolate_dbos_env_and_queue):
        import uuid

        wf_uuid = str(uuid.uuid4())
        assert len(wf_uuid) == 36  # sanity

        with (
            patch("asqi.workflow.DBOS") as mock_dbos,
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_test_plan") as mock_validate,
            patch("asqi.workflow.create_test_execution_plan") as mock_plan,
        ):
            mock_dbos.workflow_id = wf_uuid
            mock_avail.return_value = {"test/image:latest": True}
            mock_extract.return_value = {"test/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            _call_test_suite(_valid_suite_config(), _valid_systems_config())

        queue_args, _ = _isolate_dbos_env_and_queue.call_args
        queue_name = queue_args[0]
        assert queue_name == f"test_execution_{wf_uuid}"
        # Strip the documented prefix; the remainder is the UUID.
        assert len(queue_name[len("test_execution_") :]) == 36

    def test_data_generation_queue_embeds_36_char_uuid(self, _isolate_dbos_env_and_queue):
        import uuid

        wf_uuid = str(uuid.uuid4())

        with (
            patch("asqi.workflow.DBOS") as mock_dbos,
            patch("asqi.workflow.dbos_check_images_availability") as mock_avail,
            patch("asqi.workflow.extract_manifests_step") as mock_extract,
            patch("asqi.workflow.validate_data_generation_plan") as mock_validate,
            patch("asqi.workflow.create_data_generation_plan") as mock_plan,
        ):
            mock_dbos.workflow_id = wf_uuid
            mock_avail.return_value = {"gen/image:latest": True}
            mock_extract.return_value = {"gen/image:latest": _make_manifest()}
            mock_validate.return_value = []
            mock_plan.return_value = []
            _unwrap(run_data_generation_workflow)(
                {
                    "job_name": "gen-job",
                    "generation_jobs": [
                        {
                            "id": "g1",
                            "name": "g1",
                            "image": "gen/image:latest",
                            "params": {},
                        }
                    ],
                },
                None,
                _executor_dict(),
                ContainerConfig(),
                None,
                None,
            )

        queue_args, _ = _isolate_dbos_env_and_queue.call_args
        queue_name = queue_args[0]
        assert queue_name == f"data_generation_{wf_uuid}"
        assert len(queue_name[len("data_generation_") :]) == 36
