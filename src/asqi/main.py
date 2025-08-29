import atexit
import signal
from typing import Any, Dict, List, Optional

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console

from asqi.container_manager import shutdown_containers
from asqi.logging_config import configure_logging
from asqi.schemas import ScoreCard, SuiteConfig, SUTsConfig
from asqi.validation import validate_test_plan
from asqi.workflow import (
    dbos_check_images_availabilty,
    extract_manifest_from_image_step,
)

configure_logging()
console = Console()


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file.

    Args:
        file_path: Path to the YAML file to load

    Returns:
        Dictionary containing the parsed YAML data

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the YAML file contains invalid syntax or cannot be parsed
        PermissionError: If the file cannot be read due to permissions
    """
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
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


def load_and_validate_plan(
    suite_path: str, suts_path: str, show_manifests: bool
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

        # Collect all unique images from test suite
        unique_images = list(set(test.image for test in suite_config.test_suite))

        # Check image availability
        image_availability = dbos_check_images_availabilty(unique_images)

        missing_images = [
            img for img, available in image_availability.items() if not available
        ]
        if missing_images:
            console.print(
                f"[yellow]Warning:[/yellow] {len(missing_images)} images not available locally"
            )

        # Extract manifests from available images
        manifests = {}
        available_images = [
            img for img, available in image_availability.items() if available
        ]

        if available_images:
            for image in available_images:
                manifest = extract_manifest_from_image_step(image)
                if manifest:
                    manifests[image] = manifest

    except (FileNotFoundError, ValueError, ValidationError, PermissionError) as e:
        errors.append(str(e))
        return {"status": "failure", "errors": errors}

    validation_errors = validate_test_plan(suite_config, suts_config, manifests)
    if validation_errors:
        return {"status": "failure", "errors": validation_errors}

    if show_manifests:
        return {
            "status": "success",
            "errors": [],
            "available_images": available_images,
            "manifests": manifests,
        }
    else:
        return {
            "status": "success",
            "errors": [],
        }


def load_and_print_manifest(available_images: List, manifests: Dict[str, Any]):
    if available_images:
        for image in available_images:
            manifest = manifests[image]
            if manifest:
                manifests[image] = manifest
                console.print(
                    f"[blue]-- Showing manifest from image: {image}[/blue] ---"
                )

                try:
                    yaml_str = yaml.dump(manifest.model_dump(), sort_keys=False)
                except AttributeError:
                    yaml_str = yaml.dump(manifest, sort_keys=False)

                console.print(yaml_str)

    else:
        console.print("[red]✗ No available images found.[/red]")


app = typer.Typer(help="A test executor for AI systems.")


@app.callback()
def _cli_startup_callback():
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


@app.command("validate", help="Validate test plan configuration without execution.")
def validate(
    suite_file: str = typer.Option(..., help="Path to the test suite YAML file."),
    suts_file: str = typer.Option(..., help="Path to the SUTs YAML file."),
    show_manifests: bool = typer.Option(
        False, "--show-manifests", help="Print extracted manifests from docker images."
    ),
):
    """Validate test plan configuration without execution."""
    console.print("[blue]--- Running Verification ---[/blue]")

    result = load_and_validate_plan(
        suite_path=suite_file, suts_path=suts_file, show_manifests=show_manifests
    )

    if result["status"] == "failure":
        console.print("\n[red]❌ Test Plan Validation Failed:[/red]")
        for error in result["errors"]:
            for line in str(error).splitlines():
                console.print(f"  [red]- {line}[/red]")
        raise typer.Exit(1)

    console.print("\n[green]✨ Success! The test plan is valid.[/green]")

    if show_manifests:
        load_and_print_manifest(result["available_images"], result["manifests"])

    console.print(
        "[blue]💡 Use 'execute' or 'execute-tests' commands to run tests.[/blue]"
    )


@app.command()
def execute(
    suite_file: str = typer.Option(..., help="Path to the test suite YAML file."),
    suts_file: str = typer.Option(..., help="Path to the SUTs YAML file."),
    score_card_file: str = typer.Option(
        ..., help="Path to grading score card YAML file."
    ),
    output_file: Optional[str] = typer.Option(
        None, help="Path to save execution results JSON file."
    ),
):
    """Execute the complete end-to-end workflow: tests + score cards (requires Docker)."""
    console.print("[blue]--- 🚀 Executing End-to-End Workflow ---[/blue]")

    try:
        from asqi.workflow import DBOS, start_test_execution

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        # Load score card configuration
        score_card_configs = None
        try:
            score_card_config = load_score_card_file(score_card_file)
            score_card_configs = [score_card_config]
            console.print(
                f"[green]✅ Loaded grading score card: {score_card_config.get('score_card_name', 'unnamed')}[/green]"
            )
        except (FileNotFoundError, ValueError, PermissionError) as e:
            console.print(f"[red]❌ score card configuration error: {e}[/red]")
            raise typer.Exit(1)

        workflow_id = start_test_execution(
            suite_path=suite_file,
            suts_path=suts_file,
            output_path=output_file,
            score_card_configs=score_card_configs,
            execution_mode="end_to_end",
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
    suite_file: str = typer.Option(..., help="Path to the test suite YAML file."),
    suts_file: str = typer.Option(..., help="Path to the SUTs YAML file."),
    output_file: Optional[str] = typer.Option(
        None, help="Path to save execution results JSON file."
    ),
    test_names: Optional[List[str]] = typer.Option(
        None,
        "--test-names",
        help="Comma-separated list of test names to run (matches suite test names).",
    ),
):
    """Execute only the test suite, skip score card evaluation (requires Docker)."""
    console.print("[blue]--- 🚀 Executing Test Suite ---[/blue]")

    try:
        from asqi.workflow import DBOS, start_test_execution

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        workflow_id = start_test_execution(
            suite_path=suite_file,
            suts_path=suts_file,
            output_path=output_file,
            score_card_configs=None,
            execution_mode="tests_only",
            test_names=test_names,
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


@app.command(name="evaluate-score-cards")
def evaluate_score_cards(
    input_file: str = typer.Option(
        ..., help="Path to JSON file with existing test results."
    ),
    score_card_file: str = typer.Option(
        ..., help="Path to grading score card YAML file."
    ),
    output_file: Optional[str] = typer.Option(
        None, help="Path to save evaluation results JSON file."
    ),
):
    """Evaluate score cards against existing test results from JSON file."""
    console.print("[blue]--- 📊 Evaluating Score Cards ---[/blue]")

    try:
        from asqi.workflow import DBOS, start_score_card_evaluation

        # Launch DBOS if not already launched
        try:
            DBOS.launch()
        except Exception as e:
            console.print(f"[yellow]Warning: Error launching DBOS: {e}[/yellow]")

        # Load score card configuration
        try:
            score_card_config = load_score_card_file(score_card_file)
            score_card_configs = [score_card_config]
            console.print(
                f"[green]✅ Loaded grading score card: {score_card_config.get('score_card_name', 'unnamed')}[/green]"
            )
        except (FileNotFoundError, ValueError, PermissionError) as e:
            console.print(f"[red]❌ score card configuration error: {e}[/red]")
            raise typer.Exit(1)

        workflow_id = start_score_card_evaluation(
            input_path=input_file,
            score_card_configs=score_card_configs,
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


if __name__ == "__main__":
    app()
