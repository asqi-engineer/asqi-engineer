import copy
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import yaml
from pydantic import BaseModel, Field


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
    Load and parse a YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Parsed configuration dictionary
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


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
