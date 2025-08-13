import argparse
import glob
import os
import sys
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError

from asqi.schemas import Manifest, SuiteConfig, SUTsConfig
from asqi.validation import validate_test_plan


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
        help="Path to dir with test container manifests (for validation only).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the test suite using DBOS workflow (requires Docker).",
    )
    parser.add_argument(
        "--output-file", help="Path to save execution results JSON file."
    )

    args = parser.parse_args()

    if args.execute:
        # Execute the test suite
        print("--- 🚀 Executing Test Suite ---")

        try:
            from asqi.workflow import DBOS, start_test_execution

            # Launch DBOS if not already launched
            try:
                DBOS.launch()
            except Exception as e:
                print(f"Error launching DBOS: {e}")

            workflow_id = start_test_execution(
                suite_path=args.suite_file,
                suts_path=args.suts_file,
                output_path=args.output_file,
            )

            print(f"\n✨ Execution completed! Workflow ID: {workflow_id}")

        except ImportError:
            print(
                "❌ Error: DBOS workflow dependencies not available.", file=sys.stderr
            )
            print("Install with: pip install dbos", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Execution failed: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Validate the test plan
        if not args.manifests_dir:
            print("❌ Error: --manifests-dir required for validation.", file=sys.stderr)
            sys.exit(1)

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
        print("💡 Add --execute flag to run the test suite.")


if __name__ == "__main__":
    main()
