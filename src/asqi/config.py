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
