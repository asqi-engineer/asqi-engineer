import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dbos import DBOS, DBOSConfig, Queue
from pydantic import ValidationError
from rich.console import Console

from asqi.config import (
    ContainerConfig,
    ExecutorConfig,
    load_config_file,
    save_results_to_file,
)
from asqi.container_manager import (
    check_images_availability,
    extract_manifest_from_image,
    run_container_with_args,
)
from asqi.output import (
    create_test_execution_progress,
    create_workflow_summary,
    format_execution_summary,
    format_failure_summary,
    parse_container_json_output,
)
from asqi.schemas import Manifest, ScoreCard, SuiteConfig, SUTsConfig
from asqi.validation import (
    create_test_execution_plan,
    validate_execution_inputs,
    validate_score_card_inputs,
    validate_test_execution_inputs,
    validate_workflow_configurations,
)

config: DBOSConfig = {
    "name": "asqi-test-executor",
    "database_url": os.environ.get("DBOS_DATABASE_URL"),
}
DBOS(config=config)

# Initialize Rich console and execution queue
console = Console()
test_queue = Queue("test_execution", concurrency=ExecutorConfig.CONCURRENT_TESTS)


class TestExecutionResult:
    """Represents the result of a single test execution."""

    def __init__(self, test_name: str, sut_name: str, image: str):
        self.test_name = test_name
        self.sut_name = sut_name
        self.image = image
        self.start_time: float = 0
        self.end_time: float = 0
        self.success: bool = False
        self.container_id: str = ""
        self.exit_code: int = -1
        self.container_output: str = ""
        self.test_results: Dict[str, Any] = {}
        self.error_message: str = ""

    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/reporting."""
        return {
            "metadata": {
                "test_name": self.test_name,
                "sut_name": self.sut_name,
                "image": self.image,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "execution_time_seconds": self.execution_time,
                "container_id": self.container_id,
                "exit_code": self.exit_code,
                "timestamp": datetime.now().isoformat(),
                "success": self.success,
            },
            "test_results": self.test_results,
            "error_message": self.error_message,
            "container_output": self.container_output,
        }


@DBOS.step()
def check_image_availability(images: List[str]) -> Dict[str, bool]:
    """Check if all required Docker images are available locally."""
    availability = check_images_availability(images)

    # Log warnings for missing images
    missing_images = [img for img, available in availability.items() if not available]
    if missing_images:
        DBOS.logger.warning(f"Missing images: {missing_images}")

    return availability


@DBOS.step()
def extract_manifest_from_image_step(image: str) -> Optional[Manifest]:
    """Extract and parse manifest.yaml from a Docker image."""
    manifest = extract_manifest_from_image(image, ContainerConfig.MANIFEST_PATH)

    if not manifest:
        DBOS.logger.warning(f"Failed to extract manifest from {image}")

    return manifest


@DBOS.step()
def validate_test_plan(
    suite: SuiteConfig, suts: SUTsConfig, manifests: Dict[str, Manifest]
) -> List[str]:
    """
    DBOS step wrapper for comprehensive test plan validation.

    Delegates to validation.py for the actual validation logic.
    This step exists to provide DBOS durability for validation results.

    Args:
        suite: Test suite configuration (pre-validated)
        suts: SUTs configuration (pre-validated)
        manifests: Available manifests (pre-validated)

    Returns:
        List of validation error messages
    """
    # Delegate to the comprehensive validation function
    return validate_workflow_configurations(suite, suts, manifests)


@DBOS.step()
def execute_single_test(
    test_name: str,
    image: str,
    sut_name: str,
    sut_config: Dict[str, Any],
    test_params: Dict[str, Any],
) -> TestExecutionResult:
    """Execute a single test in a Docker container.

    Focuses solely on test execution. Input validation is handled separately
    in validation.py to follow single responsibility principle.

    Args:
        test_name: Name of the test to execute (pre-validated)
        image: Docker image to run (pre-validated)
        sut_name: Name of the system under test (pre-validated)
        sut_config: Configuration for the SUT (pre-validated)
        test_params: Parameters for the test (pre-validated)

    Returns:
        TestExecutionResult containing execution metadata and results

    Raises:
        ValueError: If inputs fail validation or JSON output cannot be parsed
        RuntimeError: If container execution fails
    """
    result = TestExecutionResult(test_name, sut_name, image)

    try:
        validate_test_execution_inputs(
            test_name, image, sut_name, sut_config, test_params
        )
    except ValueError as e:
        result.error_message = str(e)
        result.success = False
        return result

    # Prepare command line arguments
    try:
        sut_config_json = json.dumps(sut_config)
        test_params_json = json.dumps(test_params)
        command_args = [
            "--sut-config",
            sut_config_json,
            "--test-params",
            test_params_json,
        ]
    except (TypeError, ValueError) as e:
        result.error_message = f"Failed to serialize configuration to JSON: {e}"
        result.success = False
        return result

    # Prepare environment variables from SUT config
    container_env = {}
    if "api_key_env" in sut_config:
        api_key_env = sut_config["api_key_env"]
        if api_key_env in os.environ:
            container_env[api_key_env] = os.environ[api_key_env]
        else:
            DBOS.logger.warning(
                f"Environment variable {api_key_env} not found in host environment"
            )

    # Execute container
    result.start_time = time.time()

    container_result = run_container_with_args(
        image=image,
        args=command_args,
        timeout_seconds=ContainerConfig.TIMEOUT_SECONDS,
        memory_limit=ContainerConfig.MEMORY_LIMIT,
        cpu_quota=ContainerConfig.CPU_QUOTA,
        cpu_period=ContainerConfig.CPU_PERIOD,
        environment=container_env,
        stream_logs=True,
    )

    result.end_time = time.time()
    result.container_id = container_result["container_id"]
    result.exit_code = container_result["exit_code"]
    result.container_output = container_result["output"]
    result.error_message = container_result["error"]

    # Parse JSON output from container
    if container_result["success"]:
        try:
            result.test_results = parse_container_json_output(result.container_output)
            result.success = result.test_results.get("success", False)
        except ValueError as e:
            result.error_message = (
                f"Failed to parse JSON output from test '{test_name}': {e}"
            )
            result.success = False
            DBOS.logger.error(
                f"JSON parsing failed for test {test_name}: {result.container_output[:200]}..."
            )
    else:
        result.success = False

    # Log failures for debugging
    if not result.success:
        DBOS.logger.error(f"Test failed: {test_name} - {result.error_message}")

    return result


@DBOS.step()
def evaluate_score_card(
    test_results: List[TestExecutionResult], score_card_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Evaluate score cards against test execution results."""
    from asqi.score_card_engine import ScoreCardEngine

    if not score_card_configs:
        return []

    score_card_engine = ScoreCardEngine()
    all_evaluations = []

    for score_card_config in score_card_configs:
        try:
            # Parse score card configuration
            score_card = ScoreCard(**score_card_config)

            # Evaluate score card against test results
            score_card_evaluations = score_card_engine.evaluate_scorecard(
                test_results, score_card
            )

            # Add score card name to each evaluation
            for evaluation in score_card_evaluations:
                evaluation["score_card_name"] = score_card.score_card_name

            all_evaluations.extend(score_card_evaluations)

            DBOS.logger.info(
                f"Evaluated score card '{score_card.score_card_name}' with {len(score_card_evaluations)} individual evaluations"
            )

        except ValidationError as e:
            error_result = {
                "score_card_name": score_card_config.get("score_card_name", "unknown"),
                "error": f"Score card validation failed: {e}",
                "indicator_name": "SCORE_CARD_VALIDATION_ERROR",
                "test_name": "N/A",
                "sut_name": "N/A",
                "outcome": None,
                "metric_value": None,
            }
            all_evaluations.append(error_result)
            DBOS.logger.error(f"Score card validation failed: {e}")
        except (KeyError, AttributeError, TypeError) as e:
            error_result = {
                "score_card_name": score_card_config.get("score_card_name", "unknown"),
                "error": f"Score card evaluation error: {e}",
                "indicator_name": "SCORE_CARD_EVALUATION_ERROR",
                "test_name": "N/A",
                "sut_name": "N/A",
                "outcome": None,
                "metric_value": None,
            }
            all_evaluations.append(error_result)
            DBOS.logger.error(f"Score card evaluation error: {e}")

    return all_evaluations


@DBOS.workflow()
def run_test_suite_workflow(
    suite_config: Dict[str, Any],
    suts_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a test suite with DBOS durability (tests only, no score card evaluation).

    This workflow:
    1. Validates image availability and extracts manifests
    2. Performs cross-validation of tests, SUTs, and manifests
    3. Executes tests concurrently with progress tracking
    4. Aggregates results with detailed error reporting

    Args:
        suite_config: Serialized SuiteConfig containing test definitions
        suts_config: Serialized SUTsConfig containing SUT configurations

    Returns:
        Execution summary with metadata and individual test results (no score cards)
    """
    workflow_start_time = time.time()

    # Parse configurations
    try:
        suite = SuiteConfig(**suite_config)
        suts = SUTsConfig(**suts_config)
    except ValidationError as e:
        error_msg = f"Configuration validation failed: {e}"
        DBOS.logger.error(error_msg)
        return {
            "summary": create_workflow_summary(
                suite_name="unknown",
                workflow_id=DBOS.workflow_id or "",
                status="CONFIG_ERROR",
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                execution_time=time.time() - workflow_start_time,
                error=error_msg,
            ),
            "results": [],
        }
    except (TypeError, AttributeError) as e:
        error_msg = f"Configuration structure error: {e}"
        DBOS.logger.error(error_msg)
        return {
            "summary": create_workflow_summary(
                suite_name="unknown",
                workflow_id=DBOS.workflow_id or "",
                status="CONFIG_ERROR",
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                execution_time=time.time() - workflow_start_time,
                error=error_msg,
            ),
            "results": [],
        }

    console.print(f"\n[bold blue]Executing Test Suite:[/bold blue] {suite.suite_name}")

    # Collect all unique images from test suite
    unique_images = list(set(test.image for test in suite.test_suite))
    DBOS.logger.info(f"Found {len(unique_images)} unique images")

    # Check image availability
    with console.status("[bold blue]Checking image availability...", spinner="dots"):
        image_availability = check_image_availability(unique_images)

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
        with console.status("[bold blue]Extracting manifests...", spinner="dots"):
            for image in available_images:
                manifest = extract_manifest_from_image_step(image)
                if manifest:
                    manifests[image] = manifest

    # Validate test plan
    with console.status("[bold blue]Validating test plan...", spinner="dots"):
        validation_errors = validate_test_plan(suite, suts, manifests)

    if validation_errors:
        console.print("[red]Validation failed:[/red]")
        for error in validation_errors[: ExecutorConfig.MAX_FAILURES_DISPLAYED]:
            console.print(f"  • {error}")
        if len(validation_errors) > ExecutorConfig.MAX_FAILURES_DISPLAYED:
            remaining = len(validation_errors) - ExecutorConfig.MAX_FAILURES_DISPLAYED
            console.print(f"  • ... and {remaining} more errors")

        DBOS.logger.error(f"Validation failed with {len(validation_errors)} errors")
        return {
            "summary": create_workflow_summary(
                suite_name=suite.suite_name,
                workflow_id=DBOS.workflow_id or "",
                status="VALIDATION_FAILED",
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                execution_time=time.time() - workflow_start_time,
                validation_errors=validation_errors,
            ),
            "results": [],
        }

    # Prepare test execution plan
    test_execution_plan = create_test_execution_plan(suite, suts, image_availability)
    test_count = len(test_execution_plan)

    if test_count == 0:
        console.print("[yellow]No tests to execute[/yellow]")
        return {
            "summary": create_workflow_summary(
                suite_name=suite.suite_name,
                workflow_id=DBOS.workflow_id or "",
                status="NO_TESTS",
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                execution_time=time.time() - workflow_start_time,
            ),
            "results": [],
        }

    # Execute tests concurrently
    console.print(f"\n[bold]Running {test_count} tests...[/bold]")

    try:
        with create_test_execution_progress(console) as progress:
            task = progress.add_task("Executing tests", total=test_count)
            # Enqueue all tests for concurrent execution
            test_handles = []
            for test_plan in test_execution_plan:
                handle = test_queue.enqueue(
                    execute_single_test,
                    test_plan["test_name"],
                    test_plan["image"],
                    test_plan["sut_name"],
                    test_plan["sut_config"],
                    test_plan["test_params"],
                )
                test_handles.append(handle)

            # Collect results as they complete
            all_results = []
            for handle in test_handles:
                result = handle.get_result()
                all_results.append(result)
                try:
                    progress.advance(task)
                except (AttributeError, RuntimeError) as e:
                    DBOS.logger.warning(f"Progress update failed: {e}")

    except (ImportError, AttributeError) as e:
        # Fallback to simple execution without progress bar if Rich components fail
        DBOS.logger.warning(
            f"Progress bar unavailable, falling back to simple execution: {e}"
        )
        console.print("[yellow]Running tests without progress bar...[/yellow]")

        # Enqueue all tests for concurrent execution
        test_handles = []
        for test_plan in test_execution_plan:
            handle = test_queue.enqueue(
                execute_single_test,
                test_plan["test_name"],
                test_plan["image"],
                test_plan["sut_name"],
                test_plan["sut_config"],
                test_plan["test_params"],
            )
            test_handles.append(handle)

        # Collect results as they complete
        all_results = []
        for i, handle in enumerate(test_handles, 1):
            result = handle.get_result()
            all_results.append(result)
            if (
                i % max(1, test_count // ExecutorConfig.PROGRESS_UPDATE_INTERVAL) == 0
                or i == test_count
            ):
                console.print(f"[dim]Completed {i}/{test_count} tests[/dim]")

    workflow_end_time = time.time()

    # Generate summary
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.success)
    failed_tests = total_tests - successful_tests

    summary = create_workflow_summary(
        suite_name=suite.suite_name,
        workflow_id=DBOS.workflow_id or "",
        status="COMPLETED",
        total_tests=total_tests,
        successful_tests=successful_tests,
        failed_tests=failed_tests,
        execution_time=workflow_end_time - workflow_start_time,
        images_checked=len(unique_images),
        manifests_extracted=len(manifests),
        validation_errors=[],
    )

    # Display results
    status_color, message = format_execution_summary(
        total_tests, successful_tests, failed_tests, summary["total_execution_time"]
    )
    console.print(f"\n[{status_color}]Results:[/{status_color}] {message}")

    # Show failed tests if any
    if failed_tests > 0:
        failed_results = [r for r in all_results if not r.success]
        format_failure_summary(
            failed_results, console, ExecutorConfig.MAX_FAILURES_DISPLAYED
        )

    DBOS.logger.info(
        f"Workflow completed: {successful_tests}/{total_tests} tests passed"
    )

    return {
        "summary": summary,
        "results": [result.to_dict() for result in all_results],
    }


@DBOS.step()
def convert_test_results_to_objects(
    test_results_data: Dict[str, Any],
) -> List[TestExecutionResult]:
    """Convert test results data back to TestExecutionResult objects."""
    test_results = []
    for result_dict in test_results_data.get("results", []):
        metadata = result_dict["metadata"]
        result = TestExecutionResult(
            metadata["test_name"], metadata["sut_name"], metadata["image"]
        )
        result.start_time = metadata["start_time"]
        result.end_time = metadata["end_time"]
        result.success = metadata["success"]
        result.container_id = metadata["container_id"]
        result.exit_code = metadata["exit_code"]
        result.container_output = result_dict["container_output"]
        result.test_results = result_dict["test_results"]
        result.error_message = result_dict["error_message"]
        test_results.append(result)
    return test_results


@DBOS.step()
def add_score_cards_to_results(
    test_results_data: Dict[str, Any], score_card_evaluation: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Add score card evaluation results to test results data."""
    # Restructure score card evaluation results
    score_card = None
    if score_card_evaluation:
        # Group evaluations by score card name
        score_cards_by_name = {}
        for evaluation in score_card_evaluation:
            score_card_name = evaluation.get("score_card_name", "unknown")
            if score_card_name not in score_cards_by_name:
                score_cards_by_name[score_card_name] = []
            # Remove score_card_name from individual assessment since it's now at parent level
            assessment = {k: v for k, v in evaluation.items() if k != "score_card_name"}
            score_cards_by_name[score_card_name].append(assessment)

        # If only one score card, use single object structure
        if len(score_cards_by_name) == 1:
            score_card_name, assessments = next(iter(score_cards_by_name.items()))
            score_card = {
                "score_card_name": score_card_name,
                "total_evaluations": len(assessments),
                "assessments": assessments,
            }
        else:
            # Multiple score cards - create array of score card objects
            score_card = []
            for score_card_name, assessments in score_cards_by_name.items():
                score_card.append(
                    {
                        "score_card_name": score_card_name,
                        "total_evaluations": len(assessments),
                        "assessments": assessments,
                    }
                )

    # Create updated results with score card data
    updated_results = test_results_data.copy()
    updated_results["score_card"] = score_card
    return updated_results


@DBOS.workflow()
def evaluate_score_cards_workflow(
    test_results_data: Dict[str, Any],
    score_card_configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate score cards against existing test results.

    Args:
        test_results_data: Test execution results containing 'results' field
        score_card_configs: List of score card configurations to evaluate

    Returns:
        Updated results with score card evaluation data
    """
    # 1. Convert test results back to TestExecutionResult objects
    test_results = convert_test_results_to_objects(test_results_data)

    # 2. Evaluate score cards using existing step
    console.print("\n[bold blue]Evaluating score cards...[/bold blue]")
    score_card_evaluation = evaluate_score_card(test_results, score_card_configs)

    # 3. Add score card results to test data
    return add_score_cards_to_results(test_results_data, score_card_evaluation)


@DBOS.workflow()
def run_end_to_end_workflow(
    suite_config: Dict[str, Any],
    suts_config: Dict[str, Any],
    score_card_configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute a complete end-to-end workflow: test execution + score card evaluation.

    Args:
        suite_config: Serialized SuiteConfig containing test definitions
        suts_config: Serialized SUTsConfig containing SUT configurations
        score_card_configs: List of score card configurations to evaluate

    Returns:
        Complete execution results with test results and score card evaluations
    """
    test_results = run_test_suite_workflow(suite_config, suts_config)
    final_results = evaluate_score_cards_workflow(test_results, score_card_configs)

    return final_results


@DBOS.step()
def save_results_to_file_step(results: Dict[str, Any], output_path: str) -> None:
    """Save execution results to a JSON file."""
    try:
        save_results_to_file(results, output_path)
        console.print(f"Results saved to [bold]{output_path}[/bold]")
        DBOS.logger.info(f"Results saved to {output_path}")
    except (IOError, OSError, PermissionError) as e:
        console.print(f"[red]Failed to save results:[/red] {e}")
        DBOS.logger.error(f"Failed to save results to {output_path}: {e}")
    except (TypeError, ValueError) as e:
        console.print(f"[red]Invalid results data for saving:[/red] {e}")
        DBOS.logger.error(f"Invalid results data cannot be saved to {output_path}: {e}")


def start_test_execution(
    suite_path: str,
    suts_path: str,
    output_path: Optional[str] = None,
    score_card_configs: Optional[List[Dict[str, Any]]] = None,
    execution_mode: str = "end_to_end",
) -> str:
    """
    Orchestrate test suite execution workflow.

    Handles input validation, configuration loading, and workflow delegation.
    Actual execution logic is handled by dedicated workflow functions.

    Args:
        suite_path: Path to test suite YAML file
        suts_path: Path to SUTs YAML file
        output_path: Optional path to save results JSON file
        score_card_configs: Optional list of score card configurations to evaluate
        execution_mode: "tests_only" or "end_to_end"

    Returns:
        Workflow ID for tracking execution

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If configuration files don't exist
        PermissionError: If configuration files cannot be read
    """
    validate_execution_inputs(suite_path, suts_path, execution_mode, output_path)

    try:
        # Load configurations
        suite_config = load_config_file(suite_path)
        suts_config = load_config_file(suts_path)

        # Start appropriate workflow based on execution mode
        if execution_mode == "tests_only":
            handle = DBOS.start_workflow(
                run_test_suite_workflow, suite_config, suts_config
            )
        elif execution_mode == "end_to_end":
            if not score_card_configs:
                # Fall back to tests only if no score cards provided
                handle = DBOS.start_workflow(
                    run_test_suite_workflow, suite_config, suts_config
                )
            else:
                handle = DBOS.start_workflow(
                    run_end_to_end_workflow,
                    suite_config,
                    suts_config,
                    score_card_configs,
                )
        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        if output_path:
            results = handle.get_result()
            save_results_to_file_step(results, output_path)
        else:
            handle.get_result()

        return handle.get_workflow_id()

    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found:[/red] {e}")
        raise


def start_score_card_evaluation(
    input_path: str,
    score_card_configs: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """
    Orchestrate score card evaluation workflow.

    Handles input validation, data loading, and workflow delegation.
    Actual evaluation logic is handled by dedicated workflow functions.

    Args:
        input_path: Path to JSON file containing test execution results
        score_card_configs: List of score card configurations to evaluate
        output_path: Optional path to save updated results JSON file

    Returns:
        Workflow ID for tracking execution

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input file contains invalid JSON
        PermissionError: If input file cannot be read
    """
    validate_score_card_inputs(input_path, score_card_configs, output_path)

    try:
        with open(input_path, "r") as f:
            test_results_data = json.load(f)

        handle = DBOS.start_workflow(
            evaluate_score_cards_workflow, test_results_data, score_card_configs
        )

        # Wait for completion and optionally save results
        if output_path:
            results = handle.get_result()
            save_results_to_file_step(results, output_path)
        else:
            handle.get_result()  # Wait for completion

        return handle.get_workflow_id()

    except FileNotFoundError as e:
        console.print(f"[red]Input file not found:[/red] {e}")
        raise
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in input file:[/red] {e}")
        raise
    except (IOError, OSError) as e:
        console.print(f"[red]Failed to read input file:[/red] {e}")
        raise
    except (ValidationError, ValueError) as e:
        console.print(f"[red]Invalid configuration or data:[/red] {e}")
        raise
    except RuntimeError as e:
        console.print(f"[red]Workflow execution failed:[/red] {e}")
        raise


if __name__ == "__main__":
    DBOS.launch()
