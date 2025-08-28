import copy
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class ContainerConfig:
    """Configuration constants for container execution."""

    TIMEOUT_SECONDS: int = 300
    MEMORY_LIMIT: str = "2g"
    CPU_QUOTA: int = 200000
    CPU_PERIOD: int = 100000
    MANIFEST_PATH: str = "/app/manifest.yaml"


@dataclass
class ExecutorConfig:
    """Configuration for test executor behavior."""

    DEFAULT_CONCURRENT_TESTS: int = 3
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
