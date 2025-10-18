"""
Utilities for generating starter configuration templates for ASQI Engineer.

This module keeps template generation logic separate from the CLI so it can
be tested in isolation. Templates can be generated from scratch or enriched
with metadata from a container manifest to give users a head start when
configuring test suites for specific packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from asqi.schemas import InputParameter, Manifest, SystemInput


class TemplateType(str, Enum):
    """Supported template types for the init-config command."""

    SYSTEMS = "systems"
    SUITE = "suite"
    SCORE_CARD = "score-card"

    @classmethod
    def from_str(cls, value: str) -> "TemplateType":
        """Case-insensitive lookup helper."""
        value_lower = value.lower()
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unsupported template type '{value}'.")


@dataclass
class TemplateResult:
    """Structured response for template generation."""

    header_lines: List[str]
    payload: Dict[str, Any]
    optional_systems: List[str]


_DEFAULT_LLM_PARAMS = {
    "base_url": "http://localhost:4000/v1",
    "model": "openai/gpt-4o-mini",
    "api_key": "${OPENAI_API_KEY}",
    "env_file": ".env",
}


def _default_params_for_system(system_type: str) -> Dict[str, Any]:
    """Return sensible defaults for well-known system types."""
    if system_type == "llm_api":
        return dict(_DEFAULT_LLM_PARAMS)

    # Fallback stub for unknown system types
    return {"example_param": "value"}


def _default_value_for_param(param: InputParameter) -> Any:
    """Provide a starter value based on manifest parameter type."""
    mapping = {
        "string": "update-me",
        "integer": 1,
        "float": 0.0,
        "boolean": False,
        "list": [],
        "object": {},
    }
    return mapping.get(param.type, "update-me")


def _generate_systems_template(manifest: Optional[Manifest]) -> TemplateResult:
    """Build a systems.yaml template."""
    header = [
        "# Starter systems configuration",
        "# Update with your real endpoints and credentials.",
    ]

    systems: Dict[str, Any] = {}
    optional_systems: List[str] = []

    if manifest:
        for system_input in manifest.input_systems:
            params = _default_params_for_system(system_input.type)
            systems[system_input.name] = {
                "type": system_input.type,
                "params": params,
            }
            if not system_input.required:
                optional_systems.append(system_input.name)
    else:
        systems["my_llm_system"] = {
            "type": "llm_api",
            "params": _default_params_for_system("llm_api"),
        }

    return TemplateResult(
        header_lines=header,
        payload={"systems": systems},
        optional_systems=optional_systems,
    )


def _build_system_mappings(system_inputs: List[SystemInput]) -> Dict[str, str]:
    """Create `systems` mapping entries for non-SUT roles."""
    mappings: Dict[str, str] = {}
    for system_input in system_inputs:
        if system_input.name == "system_under_test":
            # systems_under_test handles these
            continue
        mappings[system_input.name] = system_input.name
    return mappings


def _generate_suite_template(
    manifest: Optional[Manifest], image: Optional[str]
) -> TemplateResult:
    """Build a suite YAML template."""
    header = [
        "# Starter suite configuration",
        "# Point image references to the container you want to execute.",
    ]

    if manifest:
        header.append(f"# Generated with manifest: {manifest.name} v{manifest.version}")

    default_image = image or (
        f"{manifest.name}:{manifest.version}"
        if manifest
        else "my-registry/test-package:latest"
    )

    systems_under_test: List[str]
    additional_systems: Dict[str, str] = {}
    optional_systems: List[str] = []

    if manifest and manifest.input_systems:
        sut_inputs = [
            s for s in manifest.input_systems if s.name == "system_under_test"
        ]

        if sut_inputs:
            systems_under_test = [sut_inputs[0].name]
        else:
            # Fallback to the first required system if available
            required_inputs = [s for s in manifest.input_systems if s.required]
            systems_under_test = (
                [required_inputs[0].name]
                if required_inputs
                else [manifest.input_systems[0].name]
            )

        additional_systems = _build_system_mappings(manifest.input_systems)
        optional_systems = [
            system_input.name
            for system_input in manifest.input_systems
            if not system_input.required
        ]

        params_block = {
            param.name: _default_value_for_param(param)
            for param in manifest.input_schema
        }
    else:
        systems_under_test = ["my_llm_system"]
        params_block = {"generations": 1}

    test_definition: Dict[str, Any] = {
        "name": "example_test",
        "image": default_image,
        "systems_under_test": systems_under_test,
    }

    if additional_systems:
        # Only include keys not already listed as systems_under_test
        filtered_mappings = {
            role: name
            for role, name in additional_systems.items()
            if name not in systems_under_test
        }
        if filtered_mappings:
            test_definition["systems"] = filtered_mappings

    if params_block:
        test_definition["params"] = params_block

    suite_payload = {
        "suite_name": "My Test Suite",
        "test_suite": [test_definition],
    }

    return TemplateResult(
        header_lines=header,
        payload=suite_payload,
        optional_systems=optional_systems,
    )


def _generate_score_card_template() -> TemplateResult:
    """Build a score card YAML template."""
    header = [
        "# Starter score card configuration",
        "# Update metric paths and thresholds to match your test outputs.",
    ]

    payload = {
        "score_card_name": "My Score Card",
        "indicators": [
            {
                "name": "Overall Quality",
                "apply_to": {
                    "test_name": "example_test",
                },
                "metric": "metrics.overall_score",
                "assessment": [
                    {
                        "outcome": "PASS",
                        "condition": "greater_equal",
                        "threshold": 0.8,
                        "description": "Score is at or above the target threshold.",
                    },
                    {
                        "outcome": "FAIL",
                        "condition": "less_than",
                        "threshold": 0.8,
                        "description": "Score fell below the required threshold.",
                    },
                ],
            }
        ],
    }

    return TemplateResult(header_lines=header, payload=payload, optional_systems=[])


def generate_template(
    template_type: TemplateType,
    *,
    manifest: Optional[Manifest] = None,
    image: Optional[str] = None,
) -> TemplateResult:
    """Generate a template for the requested configuration type."""
    if template_type is TemplateType.SYSTEMS:
        return _generate_systems_template(manifest)
    if template_type is TemplateType.SUITE:
        return _generate_suite_template(manifest, image)
    if template_type is TemplateType.SCORE_CARD:
        return _generate_score_card_template()

    raise ValueError(f"Unsupported template type '{template_type}'.")


def resolve_manifest_path(path: Path) -> Path:
    """
    Resolve a manifest path that might point to either a manifest file or a directory.

    Raises:
        FileNotFoundError: if the manifest file cannot be located.
    """
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "manifest.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate manifest.yaml at '{path}'. Provide a file path or directory containing manifest.yaml."
    )
