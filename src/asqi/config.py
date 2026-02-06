import copy
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


def interpolate_env_vars(data: Any) -> Any:
    """
    Recursively interpolate environment variables in configuration data.

    Supports the following syntax:
    - ${VAR} - Direct substitution of VAR
    - ${VAR:-default} - Use VAR if set and non-empty, otherwise use default
    - ${VAR-default} - Use VAR if set (including empty), otherwise use default

    Args:
        data: The data structure to interpolate (dict, list, or primitive)

    Returns:
        The data structure with environment variables interpolated
    """
    if isinstance(data, dict):
        return {key: interpolate_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [interpolate_env_vars(item) for item in data]
    elif isinstance(data, str):
        return _interpolate_string(data)
    else:
        return data


def _interpolate_string(text: str) -> str:
    """
    Interpolate environment variables in a string using regex.

    Patterns:
    - ${VAR} -> os.environ.get('VAR', '')
    - ${VAR:-default} -> os.environ.get('VAR') or 'default' (if VAR is empty or None)
    - ${VAR-default} -> os.environ.get('VAR', 'default') (if VAR is None)
    """

    def replace_var(match):
        var_expr = match.group(1)

        # Check for default value syntax
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            value = os.environ.get(var_name)
            return value if value else default
        elif "-" in var_expr:
            var_name, default = var_expr.split("-", 1)
            return os.environ.get(var_name, default)
        else:
            # Direct substitution
            return os.environ.get(var_expr, "")

    # Pattern matches ${...} where ... can contain any characters except }
    pattern = r"\$\{([^}]+)\}"
    return re.sub(pattern, replace_var, text)


class ContainerConfig(BaseModel):
    """Configuration constants for container execution"""

    MANIFEST_PATH: ClassVar[str] = "/app/manifest.yaml"

    # Defaults for docker run() kwargs
    DEFAULT_RUN_PARAMS: ClassVar[Dict[str, Any]] = {
        "detach": True,
        "remove": False,
        "network_mode": "host",
        "mem_limit": "2g",
        "cpu_period": 100000,
        "cpu_quota": 200000,
        "cap_drop": ["ALL"],
    }

    timeout_seconds: int = Field(
        default=300, description="Maximum container execution time in seconds."
    )

    stream_logs: bool = Field(
        default=False,
        description="Whether to stream container logs during execution.",
    )
    cleanup_on_finish: bool = Field(
        default=True,
        description="Whether to cleanup containers on finish.",
    )
    cleanup_force: bool = Field(
        default=True,
        description="Force cleanup even if graceful stop fails.",
    )

    run_params: Dict[str, Any] = Field(
        default_factory=lambda: dict(ContainerConfig.DEFAULT_RUN_PARAMS),
        description="Docker run() parameters (merged with defaults).",
    )

    # ---------- Loaders ----------
    @classmethod
    def load_from_yaml(cls, path: str) -> "ContainerConfig":
        """Create a NEW instance from YAML (MANIFEST_PATH stays constant)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Container config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Apply environment variable interpolation
        data = interpolate_env_vars(data)

        # Merge YAML run_params over defaults; if absent, use defaults
        merged_run_params = dict(cls.DEFAULT_RUN_PARAMS)
        yaml_run_params = data.get("run_params")
        if isinstance(yaml_run_params, dict):
            merged_run_params.update(yaml_run_params)

        return cls(
            timeout_seconds=data.get("timeout_seconds", 300),
            stream_logs=data.get("stream_logs", False),
            cleanup_on_finish=data.get("cleanup_on_finish", True),
            cleanup_force=data.get("cleanup_force", True),
            run_params=merged_run_params,
        )

    @classmethod
    def with_streaming(cls, enabled: bool) -> "ContainerConfig":
        """
        Create a new config using all default values,
        but override `stream_logs` with the given bool.
        """
        return cls(stream_logs=bool(enabled))

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
    ) -> "ContainerConfig":
        """
        Create a new config with default scalars and default run_params,
        overriding only the provided run_params arguments.
        Extra docker kwargs can be passed via **extra.
        """
        params: Dict[str, Any] = dict(cls.DEFAULT_RUN_PARAMS)
        overrides = {
            k: v
            for k, v in {
                "detach": detach,
                "remove": remove,
                "network_mode": network_mode,
                "mem_limit": mem_limit,
                "cpu_period": cpu_period,
                "cpu_quota": cpu_quota,
            }.items()
            if v is not None
        }
        params.update(overrides)
        params.update(extra)  # allow additional docker kwargs
        return cls(run_params=params)


@dataclass
class ExecutorConfig:
    """Configuration for test executor behavior."""

    DEFAULT_CONCURRENT_TESTS: int = 3
    MAX_FAILURES_DISPLAYED: int = 3
    PROGRESS_UPDATE_INTERVAL: int = 4


def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file with environment variable interpolation.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed configuration dictionary with environment variables interpolated
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Apply environment variable interpolation
    return interpolate_env_vars(data)


def merge_defaults_into_suite(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge `test_suite_default` values into each entry of `test_suite`.

    Args:
        config: The parsed config dictionary

    Returns:
        Config with defaults merged into `test_suite`
    """
    if "test_suite_default" not in config:
        return config

    default = config["test_suite_default"]

    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    new_suite = []
    for test in config["test_suite"]:
        merged_test = deep_merge(default, test)
        new_suite.append(merged_test)

    config["test_suite"] = new_suite
    return config


def save_results_to_file(results: Dict[str, Any], output_path: str) -> None:
    """
    Save execution results to a JSON file.

    Args:
        results: Results dictionary to save
        output_path: Path to output JSON file
    """
    import json

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def save_container_results_to_file(
    container_results: List[Dict[str, Any]], logs_dir: str, logs_filename: str
) -> str:
    """
    Save container results to a JSON file.

    Args:
        container_results: Container results dictionary to save
        logs_dir: Path to the logs directory
        logs_filename: Name of the file to store the container results
    """
    import json

    logs_path = f"{logs_dir}/{logs_filename}"

    with open(logs_path, "w") as f:
        json.dump(container_results, f, indent=2)
    return logs_path


class ExecutionMode(str, Enum):
    """
    Enumeration representing the supported execution modes.
    """

    END_TO_END = "end_to_end"
    TESTS_ONLY = "tests_only"
    EVALUATE_ONLY = "evaluate_only"
    VALIDATE_ONLY = "validate_only"
