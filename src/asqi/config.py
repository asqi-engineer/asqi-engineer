import copy
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

import yaml


@dataclass
class ContainerConfig:
    """Configuration constants for container execution."""

    TIMEOUT_SECONDS: int = 300
    MANIFEST_PATH: str = "/app/manifest.yaml"
    STREAM_LOGS: bool = False

    CLEANUP_ON_FINISH: bool = True
    CLEANUP_FORCE: bool = True

    # Defaults for docker run() kwargs
    DEFAULT_RUN_PARAMS: ClassVar[Dict[str, Any]] = {
        "detach": True,
        "remove": False,
        "network_mode": "host",
        "mem_limit": "2g",
        "cpu_period": 100000,
        "cpu_quota": 200000,
    }

    # Effective params after loading (starts as defaults)
    RUN_PARAMS: ClassVar[Dict[str, Any]] = DEFAULT_RUN_PARAMS.copy()

    @classmethod
    def load_from_yaml(cls, path: str) -> None:
        """Load timeout, manifest_path, and run_params from YAML; merge with defaults."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Container config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Scalars (optional; keep defaults if missing)
        if "timeout" in data:
            timeout = data["timeout"]
            if not isinstance(timeout, int) or timeout < 0:
                raise ValueError("'timeout' must be a non-negative integer.")
            cls.TIMEOUT_SECONDS = timeout

        if "manifest_path" in data:
            manifest_path = data["manifest_path"]
            if not isinstance(manifest_path, str) or not manifest_path:
                raise ValueError("'manifest_path' must be a non-empty string.")
            cls.MANIFEST_PATH = manifest_path

        if "stream_logs" in data:
            stream_logs = data["stream_logs"]
            if not isinstance(stream_logs, bool):
                raise ValueError("'stream_logs' must be a boolean.")
            cls.STREAM_LOGS = stream_logs

        if "cleanup_on_finish" in data:
            cleanup = data["cleanup_on_finish"]
            if not isinstance(cleanup, bool):
                raise ValueError("'cleanup_on_finish' must be a boolean.")
            cls.CLEANUP_ON_FINISH = cleanup

        if "cleanup_force" in data:
            cleanup_force = data["cleanup_force"]
            if not isinstance(cleanup_force, bool):
                raise ValueError("'cleanup_force' must be a boolean.")
            cls.CLEANUP_FORCE = cleanup_force

        # run_params: merge YAML over defaults; drop None values
        raw_params = data.get("run_params", {})
        if not isinstance(raw_params, dict):
            raise ValueError("'run_params' must be a mapping in YAML.")
        filtered = {k: v for k, v in raw_params.items() if v is not None}
        cls.RUN_PARAMS = {**cls.DEFAULT_RUN_PARAMS, **filtered}

    # -------------------------
    # Scalar setters
    # -------------------------
    @classmethod
    def set_timeout(cls, timeout: int) -> None:
        if not isinstance(timeout, int) or timeout < 0:
            raise ValueError("'timeout' must be a non-negative integer.")
        cls.TIMEOUT_SECONDS = timeout

    @classmethod
    def set_manifest_path(cls, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("'manifest_path' must be a non-empty string.")
        cls.MANIFEST_PATH = path

    @classmethod
    def set_stream_logs(cls, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("'stream_logs' must be a boolean.")
        cls.STREAM_LOGS = enable

    @classmethod
    def set_cleanup_on_finish(cls, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("'cleanup_on_finish' must be a boolean.")
        cls.CLEANUP_ON_FINISH = enable

    @classmethod
    def set_cleanup_force(cls, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("'cleanup_force' must be a boolean.")
        cls.CLEANUP_FORCE = enable

    # -------------------------
    # Run params setters
    # -------------------------
    @classmethod
    def set_run_param(cls, key: str, value: Any) -> None:
        """Set or override a single run parameter."""
        cls.RUN_PARAMS[key] = value

    @classmethod
    def set_run_params(cls, **kwargs) -> None:
        """Update multiple run parameters at once."""
        # Start fresh from defaults and update
        merged = {**cls.DEFAULT_RUN_PARAMS, **cls.RUN_PARAMS, **kwargs}
        cls.RUN_PARAMS = merged


container_config = ContainerConfig()


@dataclass
class ExecutorConfig:
    """Configuration for test executor behavior."""

    CONCURRENT_TESTS: int = 3
    MAX_FAILURES_DISPLAYED: int = 3
    PROGRESS_UPDATE_INTERVAL: int = 4  # Update every 25% for fallback progress


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
