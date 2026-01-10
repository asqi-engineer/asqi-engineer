import atexit
import glob
import os
import signal
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional

import typer
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console

from asqi.config import (
    ContainerConfig,
    ExecutionMode,
    ExecutorConfig,
    interpolate_env_vars,
    merge_defaults_into_suite,
)
from asqi.container_manager import shutdown_containers
from asqi.errors import (
    AuditResponsesRequiredError,
    DuplicateIDError,
    MissingIDFieldError,
)
from asqi.logging_config import configure_logging
from asqi.schemas import (
    AuditResponses,
    Manifest,
    ScoreCard,
    SuiteConfig,
    SystemsConfig,
)
from asqi.validation import validate_ids, validate_test_plan

load_dotenv()
configure_logging()
console = Console()


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file with environment variable interpolation.

    Args:
        file_path: Path to the YAML file to load

    Returns:
        Dictionary containing the parsed YAML data with environment variables interpolated

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the YAML file contains invalid syntax or cannot be parsed
        PermissionError: If the file cannot be read due to permissions
    """
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Apply environment variable interpolation
        return interpolate_env_vars(data)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: '{file_path}'") from e
    except yaml.YAMLError as e:
        raise ValueError(
            f"Invalid YAML syntax in configuration file '{file_path}': {e}"
        ) from e
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied accessing configuration file '{file_path}'"
        ) from e


def load_score_card_file(score_card_path: str) -> Dict[str, Any]:
    """Load and validate grading score card configuration.

    Args:
        score_card_path: Path to the score card YAML file

    Returns:
        Dictionary containing the validated score card configuration

    Raises:
        FileNotFoundError: If the score card file does not exist
        ValueError: If the YAML is invalid or score card schema validation fails
        PermissionError: If the file cannot be read due to permissions
    """
    try:
        score_card_data = load_yaml_file(score_card_path)
        # Validate score card structure - this will raise ValidationError if invalid
        ScoreCard(**score_card_data)
        return score_card_data
    except ValidationError as e:
        raise ValueError(
            f"Invalid score card configuration in '{score_card_path}': {e}"
        ) from e


def load_audit_responses_file(audit_responses_path: str) -> Dict[str, Any]:
    """Load and validate audit responses YAML file."""
    try:
        audit_data = load_yaml_file(audit_responses_path)
        # Validate structure - will raise ValidationError if invalid
        AuditResponses(**audit_data)
        return audit_data
    except ValidationError as e:
        raise ValueError(
            f"Invalid audit responses configuration in '{audit_responses_path}': {e}"
        ) from e


def resolve_audit_options(
    score_card_data: Dict[str, Any],
    audit_responses_path: Optional[str],
    skip_audit_flag: bool,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Handles all validation and preparation for audit indicators.

    Returns:
        - updated_score_card_data
        - audit_responses_data (or None)

    Raises:
        AuditResponsesRequiredError: If audit indicators exist but no responses provided
    """

    indicators = score_card_data.get("indicators", []) or []
    audit_indicators = [ind for ind in indicators if ind.get("type") == "audit"]
    has_audit = bool(audit_indicators)

    audit_responses_data = None

    # If no audit indicators at all
    if not has_audit:
        if audit_responses_path:
            console.print(
                "[yellow]⚠ Audit responses file provided but no 'audit' indicators "
                "found in the score card. It will be ignored.[/yellow]"
            )
        return score_card_data, None

    # Validate conflicting flags
    if audit_responses_path and skip_audit_flag:
        console.print(
            "[red]❌ Cannot use --audit-responses and --skip-audit together.[/red]"
        )
        raise typer.Exit(1)

    # Require at least one override method
    if not audit_responses_path and not skip_audit_flag:
        score_card_name = score_card_data.get("score_card_name", "unnamed")
        raise AuditResponsesRequiredError(
            score_card_name=score_card_name,
            audit_indicators=audit_indicators,
        )

    # Load provided audit responses
    if audit_responses_path:
        try:
            audit_responses_data = load_audit_responses_file(audit_responses_path)
        except (FileNotFoundError, ValueError, PermissionError) as e:
            console.print(f"[red]❌ audit responses configuration error: {e}[/red]")
            raise typer.Exit(1)

    # If skipping audit → remove them from score card
    if skip_audit_flag:
        cleaned_card = dict(score_card_data)
        cleaned_card["indicators"] = [
            ind for ind in indicators if ind.get("type") != "audit"
        ]
        return cleaned_card, None

    return score_card_data, audit_responses_data


def load_and_validate_plan(
    suite_path: str, systems_path: str, manifests_path: str
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
        systems_data = load_yaml_file(systems_path)
        systems_config = SystemsConfig(**systems_data)

        suite_data = load_yaml_file(suite_path)
        suite_data = merge_defaults_into_suite(suite_data)
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

            # Use directory name to derive image name for local validation
            # e.g., "test_containers/mock_tester/manifest.yaml" -> "mock_tester"
            container_dir = os.path.basename(os.path.dirname(manifest_path))

            # Check for duplicate container directories
            if container_dir in manifests:
                # If two manifests have the same container directory, we currently just overwrite and keep the last one.
                pass
            manifests[container_dir] = manifest

    except (FileNotFoundError, ValueError, ValidationError, PermissionError) as e:
        errors.append(str(e))
        return {"status": "failure", "errors": errors}

    validation_errors = validate_test_plan(suite_config, systems_config, manifests)
    if validation_errors:
        return {"status": "failure", "errors": validation_errors}

    return {"status": "success", "errors": []}


app = typer.Typer(help="A test executor for AI systems.")


def version_callback(value: bool):
    """Display version information and exit."""
    if value:
        try:
            pkg_version = version("asqi-engineer")
            # Extract just the version number and build info
            # Format: 0.3.1.dev2+g816449b60.d20251120 -> version 0.3.1.dev2, build g816449b60
            if "+" in pkg_version:
                ver, build = pkg_version.split("+", 1)
                # Extract git hash from build info (e.g., g816449b60.d20251120 -> g816449b60)
                git_hash = build.split(".")[0] if "." in build else build
                typer.echo(f"asqi-engineer version {ver}, build {git_hash}")
            else:
                typer.echo(f"asqi-engineer version {pkg_version}")
        except PackageNotFoundError:
            typer.echo("asqi-engineer version: unknown (not installed)")
        raise typer.Exit()


@app.callback()
def _cli_startup_callback(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Global CLI callback invoked before any subcommand.

    Registers shutdown handlers for container cleanup once per process.
    Using a callback keeps registration in the CLI layer and avoids
    side-effects at import time in libraries or tests.
    """
    # Ensure cleanup on normal interpreter exit
    atexit.register(_handle_shutdown)

    # Handle common termination signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_shutdown)
        except Exception as e:
            console.print(f"\n[red]❌Could not register handler for {sig}: {e}[/red]")


def _validate_unique_ids(*config_paths: str) -> None:
    """
    Validates ID uniqueness within each configuration file.
    Prints status and raises on duplicates.

    Raises:
        DuplicateIDError: If duplicate IDs are found within a file
        MissingIDFieldError: If required ID fields are missing within a file
    """

    console.print("\n[blue]Verifying uniqueness of IDs...[/blue]")
    try:
        validate_ids(*config_paths)
    except DuplicateIDError as e:
        console.print(f"\n[red]❌ Found Duplicated IDs: {e}[/red]")
        raise
    except MissingIDFieldError as e:
        console.print(f"\n[red]❌ Missing required ID field: {e}[/red]")
        raise
    console.print("\n[green]✅ IDs verified[/green]")


def _handle_shutdown(signum=None, frame=None):
    signame = None
    if isinstance(signum, int):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)

    if not signame:
        return

    console.print(
        f"[yellow] Shutdown signal received ({signame}). Cleaning up ...[/yellow]"
    )
    shutdown_containers()

    console.print(
        "[yellow] Containers stopped. Waiting for workflows to complete...[/yellow]"
    )


@app.command("validate", help="Validate test plan configuration without execution.")
def validate(
    test_suite_config: str = typer.Option(
        ..., "--test-suite-config", "-t", help="Path to the test suite YAML file."
    ),
    systems_config: str = typer.Option(
        ..., "--systems-config", "-s", help="Path to the systems YAML file."
    ),
    manifests_dir: str = typer.Option(
        ..., help="Path to dir with test container manifests."
    ),
):
    """Validate test plan configuration without execution."""
    console.print("[blue]--- Running Verification ---[/blue]")

    _validate_unique_ids(test_suite_config)

    result = load_and_validate_plan(
        suite_path=test_suite_config,
        systems_path=systems_config,
        manifests_path=manifests_dir,
    )

    if result["status"] == "failure":
        console.print("\n[red]❌ Test Plan Validation Failed:[/red]")
        for error in result["errors"]:
            for line in str(error).splitlines():
                console.print(f"  [red]- {line}[/red]")
        raise typer.Exit(1)

    console.print("\n[green]✨ Success! The test plan is valid.[/green]")
    console.print(
        "[blue]💡 Use 'execute' or 'execute-tests' commands to run tests.[/blue]"
    )


@app.command()
def execute(
    test_suite_config: str = typer.Option(
        ..., "--test-suite-config", "-t", help="Path to the test suite YAML file."
    ),
    systems_config: str = typer.Option(
        ..., "--systems-config", "-s", help="Path to the systems YAML file."
    ),
    score_card_config: str = typer.Option(
        ..., "--score-card-config", "-r", help="Path to grading score card YAML file."
    ),
    output_file: Optional[str] = typer.Option(
        "output_scorecard.json",
        "--output-file",
        "-o",
        help="Path to save execution results JSON file.",
    ),
    audit_responses: Optional[str] = typer.Option(
        None,
        "--audit-responses",
        "-a",
        help="Path to YAML file with manual audit indicator responses.",
    ),
    skip_audit: bool = typer.Option(
        False,
        "--skip-audit",
        help="Skip 'audit' type indicators if no audit responses are provided.",
    ),
    concurrent_tests: int = typer.Option(
        ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
        "--concurrent-tests",
        "-c",
        min=1,
        max=20,
        help=f"Number of tests to run concurrently (must be between 1 and 20, default: {ExecutorConfig.DEFAULT_CONCURRENT_TESTS})",
    ),
    max_failures: int = typer.Option(
        ExecutorConfig.MAX_FAILURES_DISPLAYED,
        "--max-failures",
        "-m",
        min=1,
        max=10,
        help=f"Maximum number of failures to display (must be between 1 and 10, default: {ExecutorConfig.MAX_FAILURES_DISPLAYED}).",
    ),
    progress_interval: int = typer.Option(
        ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
        "--progress-interval",
        "-p",
        min=1,
        max=10,
        help=f"Progress update interval (must be between 1 and 10, default: {ExecutorConfig.PROGRESS_UPDATE_INTERVAL}).",
    ),
    container_config_file: Optional[str] = typer.Option(
        None,
        "--container-config",
        help="Optional path to container configuration YAML. If not provided, built-in defaults are used.",
    ),
    datasets_config: Optional[str] = typer.Option(
        None,
        "--datasets-config",
        "-d",
        help="Path to the datasets registry YAML file containing reusable dataset definitions.",
    ),
):
    """Execute the complete end-to-end workflow: tests + score cards (requires Docker)."""
    console.print("[blue]--- 🚀 Executing End-to-End Workflow ---[/blue]")

    _validate_unique_ids(test_suite_config, score_card_config)

    try:
        from asqi.workflow import DBOS, start_test_execution

        # Load container configuration
        if container_config_file is not None:
            container_config = ContainerConfig.load_from_yaml(container_config_file)
        else:
            container_config = ContainerConfig()
        # Update ExecutorConfig from CLI args
        executor_config = {
            "concurrent_tests": concurrent_tests,
            "max_failures": max_failures,
            "progress_interval": progress_interval,
        }

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        # Load score card configuration
        try:
            score_card_data = load_score_card_file(score_card_config)
            console.print(
                f"[green]✅ Loaded grading score card: {score_card_data.get('score_card_name', 'unnamed')}[/green]"
            )
        except (FileNotFoundError, ValueError, PermissionError) as e:
            console.print(f"[red]❌ score card configuration error: {e}[/red]")
            raise typer.Exit(1)

        # Handle audit logic
        try:
            score_card_data, audit_responses_data = resolve_audit_options(
                score_card_data=score_card_data,
                audit_responses_path=audit_responses,
                skip_audit_flag=skip_audit,
            )
        except AuditResponsesRequiredError as e:
            console.print(f"[red]❌ {e}[/red]")
            raise typer.Exit(1)

        score_card_configs = [score_card_data]

        workflow_id = start_test_execution(
            suite_path=test_suite_config,
            systems_path=systems_config,
            datasets_config_path=datasets_config,
            output_path=output_file,
            score_card_configs=score_card_configs,
            execution_mode=ExecutionMode.END_TO_END,
            executor_config=executor_config,
            container_config=container_config,
            audit_responses_data=audit_responses_data,
        )

        console.print(
            f"\n[green]✨ Execution completed! Workflow ID: {workflow_id}[/green]"
        )

    except ImportError:
        console.print("[red]❌ Error: DBOS workflow dependencies not available.[/red]")
        console.print("[yellow]Install with: pip install dbos[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Execution failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="execute-tests")
def execute_tests(
    test_suite_config: str = typer.Option(
        ..., "--test-suite-config", "-t", help="Path to the test suite YAML file."
    ),
    systems_config: str = typer.Option(
        ..., "--systems-config", "-s", help="Path to the systems YAML file."
    ),
    output_file: Optional[str] = typer.Option(
        "output.json",
        "--output-file",
        "-o",
        help="Path to save execution results JSON file.",
    ),
    test_ids: Optional[List[str]] = typer.Option(
        None,
        "--test-ids",
        "-tids",
        help="Comma-separated list of test ids to run (matches suite test ids).",
    ),
    concurrent_tests: int = typer.Option(
        ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
        "--concurrent-tests",
        "-c",
        min=1,
        max=20,
        help=f"Number of tests to run concurrently (must be between 1 and 20, default: {ExecutorConfig.DEFAULT_CONCURRENT_TESTS})",
    ),
    max_failures: int = typer.Option(
        ExecutorConfig.MAX_FAILURES_DISPLAYED,
        "--max-failures",
        "-m",
        min=1,
        max=10,
        help=f"Maximum number of failures to display (must be between 1 and 10, default: {ExecutorConfig.MAX_FAILURES_DISPLAYED}).",
    ),
    progress_interval: int = typer.Option(
        ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
        "--progress-interval",
        "-p",
        min=1,
        max=10,
        help=f"Progress update interval (must be between 1 and 10, default: {ExecutorConfig.PROGRESS_UPDATE_INTERVAL}).",
    ),
    container_config_file: Optional[str] = typer.Option(
        None,
        "--container-config",
        help="Optional path to container configuration YAML. If not provided, built-in defaults are used.",
    ),
    datasets_config: Optional[str] = typer.Option(
        None,
        "--datasets-config",
        "-d",
        help="Path to the datasets registry YAML file containing reusable dataset definitions.",
    ),
):
    """Execute only the test suite, skip score card evaluation (requires Docker)."""
    console.print("[blue]--- 🚀 Executing Test Suite ---[/blue]")

    _validate_unique_ids(test_suite_config)

    try:
        from asqi.workflow import DBOS, start_test_execution

        # Load container configuration
        if container_config_file is not None:
            container_config = ContainerConfig.load_from_yaml(container_config_file)
        else:
            container_config = ContainerConfig()

        # Update ExecutorConfig from CLI args
        executor_config = {
            "concurrent_tests": concurrent_tests,
            "max_failures": max_failures,
            "progress_interval": progress_interval,
        }

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        workflow_id = start_test_execution(
            suite_path=test_suite_config,
            systems_path=systems_config,
            datasets_config_path=datasets_config,
            output_path=output_file,
            score_card_configs=None,
            execution_mode=ExecutionMode.TESTS_ONLY,
            test_ids=test_ids,
            executor_config=executor_config,
            container_config=container_config,
        )

        console.print(
            f"\n[green]✨ Test execution completed! Workflow ID: {workflow_id}[/green]"
        )

    except ImportError:
        console.print("[red]❌ Error: DBOS workflow dependencies not available.[/red]")
        console.print("[yellow]Install with: pip install dbos[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Test execution failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="generate-dataset")
def generate_dataset(
    generation_config: str = typer.Option(
        ..., "--generation-config", "-t", help="Path to the Generation YAML file."
    ),
    systems_config: Optional[str] = typer.Option(
        None, "--systems-config", "-s", help="Path to the systems YAML file."
    ),
    output_file: Optional[str] = typer.Option(
        "output.json",
        "--output-file",
        "-o",
        help="Path to save execution results JSON file.",
    ),
    concurrent_tests: int = typer.Option(
        ExecutorConfig.DEFAULT_CONCURRENT_TESTS,
        "--concurrent-tests",
        "-c",
        min=1,
        max=20,
        help=f"Number of tests to run concurrently (must be between 1 and 20, default: {ExecutorConfig.DEFAULT_CONCURRENT_TESTS})",
    ),
    max_failures: int = typer.Option(
        ExecutorConfig.MAX_FAILURES_DISPLAYED,
        "--max-failures",
        "-m",
        min=1,
        max=10,
        help=f"Maximum number of failures to display (must be between 1 and 10, default: {ExecutorConfig.MAX_FAILURES_DISPLAYED}).",
    ),
    progress_interval: int = typer.Option(
        ExecutorConfig.PROGRESS_UPDATE_INTERVAL,
        "--progress-interval",
        "-p",
        min=1,
        max=10,
        help=f"Progress update interval (must be between 1 and 10, default: {ExecutorConfig.PROGRESS_UPDATE_INTERVAL}).",
    ),
    container_config_file: Optional[str] = typer.Option(
        None,
        "--container-config",
        help="Optional path to container configuration YAML. If not provided, built-in defaults are used.",
    ),
    datasets_config: Optional[str] = typer.Option(
        None,
        "--datasets-config",
        "-d",
        help="Path to the datasets registry YAML file containing reusable dataset definitions.",
    ),
):
    """Generate synthetic data using data generation containers."""
    console.print("[blue]--- 🚀 Executing Test Suite ---[/blue]")

    try:
        from asqi.workflow import DBOS, start_data_generation

        # Load container configuration
        if container_config_file is not None:
            container_config = ContainerConfig.load_from_yaml(container_config_file)
        else:
            container_config = ContainerConfig()

        # Update ExecutorConfig from CLI args
        executor_config = {
            "concurrent_tests": concurrent_tests,
            "max_failures": max_failures,
            "progress_interval": progress_interval,
        }

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        workflow_id = start_data_generation(
            generation_config_path=generation_config,
            systems_path=systems_config,
            datasets_config_path=datasets_config,
            container_config=container_config,
            executor_config=executor_config,
            output_path=output_file,
        )

        console.print(
            f"\n[green]✨ Data Generation Completed! Workflow ID: {workflow_id}[/green]"
        )

    except ImportError:
        console.print("[red]❌ Error: DBOS workflow dependencies not available.[/red]")
        console.print("[yellow]Install with: pip install dbos[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Test execution failed: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="evaluate-score-cards")
def evaluate_score_cards(
    input_file: str = typer.Option(
        ..., help="Path to JSON file with existing test results."
    ),
    score_card_config: str = typer.Option(
        ..., "--score-card-config", "-r", help="Path to grading score card YAML file."
    ),
    output_file: Optional[str] = typer.Option(
        "output_scorecard.json",
        "--output-file",
        "-o",
        help="Path to save evaluation results JSON file.",
    ),
    audit_responses: Optional[str] = typer.Option(
        None,
        "--audit-responses",
        "-a",
        help="Path to YAML file with manual audit indicator responses.",
    ),
    skip_audit: bool = typer.Option(
        False,
        "--skip-audit",
        help="Skip 'audit' type indicators if no audit responses are provided.",
    ),
):
    """Evaluate score cards against existing test results from JSON file."""
    console.print("[blue]--- 📊 Evaluating Score Cards ---[/blue]")

    _validate_unique_ids(score_card_config)

    try:
        from asqi.workflow import DBOS, start_score_card_evaluation

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        # Load score card configuration
        try:
            score_card_data = load_score_card_file(score_card_config)
            console.print(
                f"[green]✅ Loaded grading score card: {score_card_data.get('score_card_name', 'unnamed')}[/green]"
            )
        except (FileNotFoundError, ValueError, PermissionError) as e:
            console.print(f"[red]❌ score card configuration error: {e}[/red]")
            raise typer.Exit(1)

        # Handle audit logic
        try:
            score_card_data, audit_responses_data = resolve_audit_options(
                score_card_data=score_card_data,
                audit_responses_path=audit_responses,
                skip_audit_flag=skip_audit,
            )
        except AuditResponsesRequiredError as e:
            console.print(f"[red]❌ {e}[/red]")
            raise typer.Exit(1)

        score_card_configs = [score_card_data]

        workflow_id = start_score_card_evaluation(
            input_path=input_file,
            score_card_configs=score_card_configs,
            audit_responses_data=audit_responses_data,
            output_path=output_file,
        )

        console.print(
            f"\n[green]✨ Score card evaluation completed! Workflow ID: {workflow_id}[/green]"
        )

    except ImportError:
        console.print("[red]❌ Error: DBOS workflow dependencies not available.[/red]")
        console.print("[yellow]Install with: pip install dbos[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Score card evaluation failed: {e}[/red]")
        raise typer.Exit(1)


# Expose the Click object for sphinx_click documentation
typer_click_object = typer.main.get_command(app)

if __name__ == "__main__":
    app()
