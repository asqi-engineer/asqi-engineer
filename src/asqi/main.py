import argparse
import glob
import os
import sys
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError

from asqi.schemas import Manifest, SuiteConfig, SUTsConfig
from asqi.validator import validate_test_plan


class ConfigError(Exception):
    pass


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file, raising a ConfigError on failure."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"File not found at '{file_path}'")
    except yaml.YAMLError as e:
        raise ConfigError(f"Could not parse YAML file '{file_path}': {e}")


def load_and_validate_plan(
    suite_path: str, suts_path: str, manifests_path: str
) -> Dict[str, Any]:
    """
    Performs all validation and returns a structured result.
    This function is pure and does not print or exit.

    Returns:
        A dictionary, e.g., {"status": "success", "errors": []} or
        {"status": "failure", "errors": ["error message"]}.
    """
    errors: List[str] = []
    try:
        suts_data = load_yaml_file(suts_path)
        suts_config = SUTsConfig(**suts_data)

        suite_data = load_yaml_file(suite_path)
        suite_config = SuiteConfig(**suite_data)

        # Load manifests - currently just loads locally. TODO: obtain from registry
        manifests: Dict[str, Manifest] = {}
        manifest_files = glob.glob(
            os.path.join(manifests_path, "**/manifest.yaml"), recursive=True
        )

        for manifest_path in manifest_files:
            manifest_data = load_yaml_file(manifest_path)
            if not manifest_data:
                errors.append(
                    f"Warning: Manifest file at '{manifest_path}' is empty or invalid. Skipping."
                )
                continue

            manifest = Manifest(**manifest_data)
            if manifest.image_name in manifests:
                # If two manifests have the same image_name, we currently just overwrite and keep the last one.
                pass
            manifests[manifest.image_name] = manifest

    except (ConfigError, ValidationError) as e:
        errors.append(str(e))
        return {"status": "failure", "errors": errors}

    validation_errors = validate_test_plan(suite_config, suts_config, manifests)
    if validation_errors:
        return {"status": "failure", "errors": validation_errors}

    return {"status": "success", "errors": []}


def main():
    """Main CLI entrypoint. Handles argument parsing, calling logic, and printing results."""
    parser = argparse.ArgumentParser(
        description="A durable test executor for AI systems."
    )
    parser.add_argument(
        "--suite-file", required=True, help="Path to the test suite YAML file."
    )
    parser.add_argument(
        "--suts-file", required=True, help="Path to the SUTs YAML file."
    )
    parser.add_argument(
        "--manifests-dir",
        required=True,
        help="Path to dir with test container manifests.",
    )

    args = parser.parse_args()

    print("--- Running Verification ---")

    result = load_and_validate_plan(
        suite_path=args.suite_file,
        suts_path=args.suts_file,
        manifests_path=args.manifests_dir,
    )

    if result["status"] == "failure":
        print("\n❌ Test Plan Validation Failed:", file=sys.stderr)
        for error in result["errors"]:
            for line in str(error).splitlines():
                print(f"  - {line}", file=sys.stderr)
        sys.exit(1)

    print("\n✨ Success! The test plan is valid.")


if __name__ == "__main__":
    main()
