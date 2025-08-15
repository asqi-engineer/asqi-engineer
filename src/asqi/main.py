import argparse
import glob
import os
import sys
from typing import Any, Dict, List

import yaml
from pydantic import ValidationError

from asqi.schemas import Manifest, ScoreCard, SuiteConfig, SUTsConfig
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


def load_score_card_file(score_card_path: str) -> Dict[str, Any]:
    """Load and validate grading score card configuration."""
    try:
        score_card_data = load_yaml_file(score_card_path)
        # Validate score card structure
        _score_card = ScoreCard(**score_card_data)
        return score_card_data
    except ValidationError as e:
        raise ConfigError(
            f"Invalid score card configuration in '{score_card_path}': {e}"
        )
    except Exception as e:
        raise ConfigError(f"Failed to load score card file '{score_card_path}': {e}")


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
    parser.add_argument("--suite-file", help="Path to the test suite YAML file.")
    parser.add_argument("--suts-file", help="Path to the SUTs YAML file.")
    parser.add_argument(
        "--manifests-dir",
        help="Path to dir with test container manifests (for validation only).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the complete end-to-end workflow: tests + score cards (requires Docker).",
    )
    parser.add_argument(
        "--execute-tests",
        action="store_true",
        help="Execute only the test suite, skip score card evaluation (requires Docker).",
    )
    parser.add_argument(
        "--evaluate-score-cards",
        action="store_true",
        help="Evaluate score cards against existing test results from JSON file.",
    )
    parser.add_argument(
        "--output-file", help="Path to save execution results JSON file."
    )
    parser.add_argument(
        "--input-file",
        help="Path to JSON file with existing test results (for --evaluate-score-cards).",
    )
    parser.add_argument(
        "--score-card-file", help="Path to grading score card YAML file (optional)."
    )

    args = parser.parse_args()

    # Validate execution mode arguments
    execution_modes = [args.execute, args.execute_tests, args.evaluate_score_cards]
    if sum(execution_modes) > 1:
        print("❌ Error: Cannot specify multiple execution modes.", file=sys.stderr)
        sys.exit(1)

    # Validate required arguments for each mode
    if args.execute or args.execute_tests:
        if not args.suite_file:
            print(
                "❌ Error: --suite-file required for test execution.", file=sys.stderr
            )
            sys.exit(1)
        if not args.suts_file:
            print("❌ Error: --suts-file required for test execution.", file=sys.stderr)
            sys.exit(1)
    elif args.evaluate_score_cards:
        if not args.input_file:
            print(
                "❌ Error: --input-file required for score card evaluation.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.score_card_file:
            print(
                "❌ Error: --score-card-file required for score card evaluation.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Validation mode - check required args
        if not args.suite_file:
            print("❌ Error: --suite-file required for validation.", file=sys.stderr)
            sys.exit(1)
        if not args.suts_file:
            print("❌ Error: --suts-file required for validation.", file=sys.stderr)
            sys.exit(1)
        if not args.manifests_dir:
            print("❌ Error: --manifests-dir required for validation.", file=sys.stderr)
            sys.exit(1)

    if args.execute or args.execute_tests:
        # Execute the test suite
        print("--- 🚀 Executing Test Suite ---")

        try:
            from asqi.workflow import DBOS, start_test_execution

            # Launch DBOS if not already launched
            try:
                DBOS.launch()
            except Exception as e:
                print(f"Error launching DBOS: {e}")

            # Determine execution mode
            execution_mode = "tests_only" if args.execute_tests else "end_to_end"

            # Load score card configuration if provided
            score_card_configs = None
            if args.score_card_file:
                try:
                    score_card_config = load_score_card_file(args.score_card_file)
                    score_card_configs = [score_card_config]
                    print(
                        f"✅ Loaded grading score card: {score_card_config.get('score_card_name', 'unnamed')}"
                    )
                except ConfigError as e:
                    print(f"❌ score card configuration error: {e}", file=sys.stderr)
                    sys.exit(1)

            # Validate score card requirement for end-to-end mode
            if execution_mode == "end_to_end" and not score_card_configs:
                print(
                    "⚠️  Warning: No score card provided for end-to-end execution. Running tests only."
                )

            workflow_id = start_test_execution(
                suite_path=args.suite_file,
                suts_path=args.suts_file,
                output_path=args.output_file,
                score_card_configs=score_card_configs,
                execution_mode=execution_mode,
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

    elif args.evaluate_score_cards:
        # Evaluate score cards against existing test results
        print("--- 📊 Evaluating Score Cards ---")

        try:
            from asqi.workflow import DBOS, start_score_card_evaluation

            # Launch DBOS if not already launched
            try:
                DBOS.launch()
            except Exception as e:
                print(f"Error launching DBOS: {e}")

            # Load score card configuration
            try:
                score_card_config = load_score_card_file(args.score_card_file)
                score_card_configs = [score_card_config]
                print(
                    f"✅ Loaded grading score card: {score_card_config.get('score_card_name', 'unnamed')}"
                )
            except ConfigError as e:
                print(f"❌ score card configuration error: {e}", file=sys.stderr)
                sys.exit(1)

            workflow_id = start_score_card_evaluation(
                input_path=args.input_file,
                score_card_configs=score_card_configs,
                output_path=args.output_file,
            )

            print(f"\n✨ Score card evaluation completed! Workflow ID: {workflow_id}")

        except ImportError:
            print(
                "❌ Error: DBOS workflow dependencies not available.", file=sys.stderr
            )
            print("Install with: pip install dbos", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Score card evaluation failed: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # Validate the test plan
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
