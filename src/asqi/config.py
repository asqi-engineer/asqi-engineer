import copy
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

import yaml


@dataclass
class ContainerConfig:
    """Configuration constants for container execution."""

    # Scalars
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
    RUN_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: dict(ContainerConfig.DEFAULT_RUN_PARAMS)
    )

    # ---------- Load from YAML ----------
    def load_from_yaml(self, path: str) -> None:
        """Load timeout, manifest_path, stream/cleanup flags, and run_params; merge with defaults."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Container config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Scalars
        if "timeout" in data:
            self.set_timeout(data["timeout"])
        if "manifest_path" in data:
            self.set_manifest_path(data["manifest_path"])
        if "stream_logs" in data:
            self.set_stream_logs(data["stream_logs"])
        if "cleanup_on_finish" in data:
            self.set_cleanup_on_finish(data["cleanup_on_finish"])
        if "cleanup_force" in data:
            self.set_cleanup_force(data["cleanup_force"])

        # run_params: merge YAML over current (which started from defaults)
        raw_params = data.get("run_params", {})
        if not isinstance(raw_params, dict):
            raise ValueError("'run_params' must be a mapping in YAML.")
        self.set_run_params(**{k: v for k, v in raw_params.items() if v is not None})

    # ---------- Scalar setters ----------
    def set_timeout(self, timeout: int) -> None:
        if not isinstance(timeout, int) or timeout < 0:
            raise ValueError("'timeout' must be a non-negative integer.")
        self.TIMEOUT_SECONDS = timeout

    def set_manifest_path(self, path: str) -> None:
        if not isinstance(path, str) or not path:
            raise ValueError("'manifest_path' must be a non-empty string.")
        self.MANIFEST_PATH = path

    def set_stream_logs(self, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("'stream_logs' must be a boolean.")
        self.STREAM_LOGS = enable

    def set_cleanup_on_finish(self, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("'cleanup_on_finish' must be a boolean.")
        self.CLEANUP_ON_FINISH = enable

    def set_cleanup_force(self, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("'cleanup_force' must be a boolean.")
        self.CLEANUP_FORCE = enable

    # ---------- Run params setters ----------
    def set_run_param(self, key: str, value: Any) -> None:
        """Set or override a single docker run() parameter."""
        self.RUN_PARAMS[key] = value

    def set_run_params(self, **kwargs) -> None:
        """Update multiple run params; merge into current dict."""
        self.RUN_PARAMS.update(kwargs)


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
