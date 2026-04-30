"""
pytest plugin for ASQI test container authors.

Registered via the ``pytest11`` entry point when ``asqi-engineer[test]`` is
installed. Provides the ``container_root`` and ``asqi_execute`` fixtures and
the ``requires_hf_token`` / ``requires_llm_api`` marks.

Fixtures
--------
container_root
    Absolute path to the container's root directory (parent of ``tests/``).

asqi_execute
    High-level runner that mirrors ``asqi execute`` CLI. Accepts the same
    config file arguments, resolves YAML → param dicts, sets up mount dirs,
    imports ``entrypoint.main`` in-process, and returns
    ``(ContainerOutput, exit_code)``.

Marks
-----
requires_hf_token
    Skip the test when ``HF_TOKEN`` is not set in the environment.

requires_llm_api
    Skip the test when no recognised LLM API key is set in the environment.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from asqi.response_schemas import (
    ContainerOutput,
    GeneratedDataset,
    GeneratedReport,
    validate_container_output,
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

_LLM_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY")

requires_hf_token = pytest.mark.skipif(
    "HF_TOKEN" not in os.environ,
    reason="requires HF_TOKEN environment variable",
)

requires_llm_api = pytest.mark.skipif(
    not any(k in os.environ for k in _LLM_API_KEY_ENV_VARS),
    reason=f"requires at least one LLM API key env var: {', '.join(_LLM_API_KEY_ENV_VARS)}",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def extract_container_output(container_json_output: dict[str, Any]) -> ContainerOutput:
    """
    DBOS-free equivalent of ``asqi.output.extract_container_json_output_fields``.

    The version in ``asqi.output`` calls ``DBOS.logger.warning()`` on validation
    failures, which requires a running DBOS context. This version uses the
    standard ``logging`` module instead so it is safe in test environments.
    """
    has_structured_fields = (
        "results" in container_json_output
        or "test_results" in container_json_output
        or "generated_reports" in container_json_output
        or "generated_datasets" in container_json_output
    )

    if not has_structured_fields:
        return ContainerOutput(results=container_json_output)

    try:
        return validate_container_output(container_json_output)
    except (ValidationError, ValueError):
        results = container_json_output.get("results") or container_json_output.get("test_results")
        if results == {}:
            results = None

        generated_reports: list[GeneratedReport] = []
        for report in container_json_output.get("generated_reports") or []:
            try:
                generated_reports.append(GeneratedReport(**report))
            except ValidationError:
                continue

        generated_datasets: list[GeneratedDataset] = []
        for dataset in container_json_output.get("generated_datasets") or []:
            try:
                generated_datasets.append(GeneratedDataset(**dataset))
            except ValidationError:
                continue

        return ContainerOutput(
            results=results,
            generated_reports=generated_reports,
            generated_datasets=generated_datasets,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def container_root(request: pytest.FixtureRequest) -> Path:
    """
    Absolute path to the container's root directory (parent of ``tests/``).

    Derived automatically from the location of the test file so container
    authors never need to compute ``Path(__file__).parent.parent`` themselves.
    """
    return Path(str(request.fspath)).parent.parent


@pytest.fixture
def asqi_execute(
    container_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., list[tuple[ContainerOutput, int]]]:
    """
    High-level container runner that mirrors the ``asqi execute`` CLI.

    Returns a callable that accepts the same config file arguments as
    ``asqi execute`` and:

    1. Resolves each path against ``container_root``.
    2. Loads and validates the YAML files using the same logic as the CLI.
    3. Resolves dataset name references when ``datasets_config`` is provided.
    4. Derives ``systems_params`` / ``test_params`` via the CLI's config
       resolution pipeline (``create_test_execution_plan``).
    5. Sets up ``OUTPUT_MOUNT_PATH`` / ``INPUT_MOUNT_PATH`` in ``tmp_path``.
    6. Imports ``entrypoint.main`` in-process (importable because the container
       root is on ``sys.path`` via ``pythonpath = ["."]`` in ``pyproject.toml``).
    7. Patches ``sys.argv``, redirects stdout, calls ``main()``, captures the
       exit code, parses the JSON output, and returns ``(ContainerOutput, exit_code)``
       for a single plan entry, or a list of such tuples for multiple entries.

    Usage::

        def test_scores_within_range(mock_full_run, api_success, asqi_execute):
            mock_full_run(dataset_rows=[...], api_side_effect=api_success())
            results = asqi_execute(
                test_suite_config="config/suite.yaml",
                systems_config="config/systems.yaml",
                datasets_config="config/datasets.yaml",  # optional
            )
            output, exit_code = results[0]
            assert exit_code == 0
    """
    import asqi.main as _asqi_main
    from asqi.output import parse_container_json_output
    from asqi.schemas import DatasetsConfig, SuiteConfig, SystemsConfig
    from asqi.validation import create_test_execution_plan, resolve_dataset_references

    def _run(
        test_suite_config: str,
        systems_config: str,
        datasets_config: str | None = None,
    ) -> tuple[ContainerOutput, int] | list[tuple[ContainerOutput, int]]:
        # ── Resolve config paths ──────────────────────────────────────────────
        suite_path = str(container_root / test_suite_config)
        systems_path = str(container_root / systems_config)

        # ── Load and validate YAML configs ────────────────────────────────────
        # Access via module reference so mocker.patch("asqi.main.load_yaml_file")
        # works in tests — a local `from ... import` binding would bypass patches.
        suite = SuiteConfig(**_asqi_main.load_yaml_file(suite_path))
        systems = SystemsConfig(**_asqi_main.load_yaml_file(systems_path))

        # ── Resolve dataset name references ───────────────────────────────────
        if datasets_config is not None:
            datasets_path = str(container_root / datasets_config)
            datasets = DatasetsConfig(**_asqi_main.load_yaml_file(datasets_path))
            suite = resolve_dataset_references(suite, datasets)  # type: ignore[assignment]

        # ── Mark all images as available (no Docker needed in-process) ────────
        image_availability: dict[str, bool] = {
            test.image: True for test in suite.test_suite if getattr(test, "image", None)
        }

        # ── Resolve YAML → (systems_params, test_params) ─────────────────────
        plan = create_test_execution_plan(suite, systems, image_availability)
        if not plan:
            raise ValueError(
                f"No executable test entries found after resolving configs:\n"
                f"  suite:   {suite_path}\n"
                f"  systems: {systems_path}\n"
                "Check that systems_under_test in the suite matches a system "
                "defined in the systems config."
            )

        # ── Mount directories ─────────────────────────────────────────────────
        output_dir = tmp_path / "output"
        input_dir = tmp_path / "input"
        output_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("OUTPUT_MOUNT_PATH", str(output_dir))
        monkeypatch.setenv("INPUT_MOUNT_PATH", str(input_dir))

        # ── Import entrypoint.main ────────────────────────────────────────────
        entrypoint_mod = importlib.import_module("entrypoint")
        main = entrypoint_mod.main

        # ── Run each plan entry ───────────────────────────────────────────────
        results = []
        for entry in plan:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "entrypoint.py",
                    "--systems-params",
                    json.dumps(entry["systems_params"]),
                    "--test-params",
                    json.dumps(entry["test_params"]),
                ],
            )
            captured = io.StringIO()
            exit_code = 0
            with contextlib.redirect_stdout(captured):
                try:
                    main()
                except SystemExit as exc:
                    exit_code = exc.code if exc.code is not None else 0

            parsed = parse_container_json_output(captured.getvalue())
            results.append((extract_container_output(parsed), exit_code))

        return results

    return _run
