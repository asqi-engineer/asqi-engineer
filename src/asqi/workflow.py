import json
import os
import time
import uuid
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dbos import DBOS, DBOSConfig, Queue
from dotenv import dotenv_values, load_dotenv
from pydantic import ValidationError
from rich.console import Console

from asqi.config import (
    ContainerConfig,
    load_config_file,
    merge_defaults_into_suite,
    save_container_results_to_file,
    save_results_to_file,
)
from asqi.container_manager import (
    INPUT_MOUNT_PATH,
    OUTPUT_MOUNT_PATH,
    check_images_availability,
    extract_manifest_from_image,
    pull_images,
    run_container_with_args,
)
from asqi.output import (
    create_test_execution_progress,
    create_workflow_summary,
    format_execution_summary,
    format_failure_summary,
    parse_container_json_output,
)
from asqi.schemas import Manifest, ScoreCard, SuiteConfig, SystemsConfig
from asqi.validation import (
    create_test_execution_plan,
    validate_execution_inputs,
    validate_score_card_inputs,
    validate_test_execution_inputs,
    validate_test_volumes,
    validate_workflow_configurations,
)

load_dotenv()
oltp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
system_database_url = os.environ.get("DBOS_DATABASE_URL")
if not system_database_url:
    raise ValueError(
        "Database URL must be provided through DBOS_DATABASE_URL environment variable"
    )

config: DBOSConfig = {
    "name": "asqi-test-executor",
    "system_database_url": system_database_url,
    "enable_otlp": True,
}
if oltp_endpoint:
    config["otlp_traces_endpoints"] = [oltp_endpoint]
DBOS(config=config)

# Initialize Rich console and execution queue
console = Console()


def _get_docker_socket_path(env_vars: dict[str, str]) -> str:
    """
    Get the local Docker socket path from the environment.

    Extracts the socket path from DOCKER_HOST environment variable if set,
    otherwise defaults to the standard Linux path.

    Returns:
        Path to the Docker socket on the host system
    """
    docker_host = env_vars.get("DOCKER_HOST", "")
    if docker_host:
        # Remove 'unix://' prefix in Unix socket format: unix:///path/to/socket
        return docker_host.removeprefix("unix://")
    # else, default to standard Linux path
    return "/var/run/docker.sock"


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

    def metadata_dict(self) -> Dict[str, Any]:
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
        }

    def container_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/reporting."""
        return {
            "test_name": self.test_name,
            "test_results": self.test_results,
            "error_message": self.error_message,
            "container_output": self.container_output,
        }


@DBOS.step()
def dbos_check_images_availability(images: List[str]) -> Dict[str, bool]:
    """Check if all required Docker images are available locally."""
    return check_images_availability(images)


@DBOS.step()
def dbos_pull_images(images: List[str]):
    """Pull missing Docker images from registries."""
    return pull_images(images)


@DBOS.step()
def extract_manifest_from_image_step(image: str) -> Optional[Manifest]:
    """Extract and parse manifest.yaml from a Docker image."""
    manifest = extract_manifest_from_image(image, ContainerConfig.MANIFEST_PATH)

    if not manifest:
        DBOS.logger.warning(f"Failed to extract manifest from {image}")

    return manifest


@DBOS.step()
def validate_test_plan(
    suite: SuiteConfig, systems: SystemsConfig, manifests: Dict[str, Manifest]
) -> List[str]:
    """
    DBOS step wrapper for comprehensive test plan validation.

    Delegates to validation.py for the actual validation logic.
    This step exists to provide DBOS durability for validation results.

    Args:
        suite: Test suite configuration (pre-validated)
        systems: systems configuration (pre-validated)
        manifests: Available manifests (pre-validated)

    Returns:
        List of validation error messages
    """
    # Delegate to the comprehensive validation function
    return validate_workflow_configurations(suite, systems, manifests)


@DBOS.step()
def execute_single_test(
    test_name: str,
    image: str,
    sut_name: str,
    systems_params: Dict[str, Any],
    test_params: Dict[str, Any],
    container_config: ContainerConfig,
) -> TestExecutionResult:
    """Execute a single test in a Docker container.

    Focuses solely on test execution. Input validation is handled separately
    in validation.py to follow single responsibility principle.

    Args:
        test_name: Name of the test to execute (pre-validated)
        image: Docker image to run (pre-validated)
        sut_name: Name of the system under test (pre-validated)
        systems_params: Dictionary containing system_under_test and other systems (pre-validated)
        test_params: Parameters for the test (pre-validated)
        container_config: Container execution configurations

    Returns:
        TestExecutionResult containing execution metadata and results

    Raises:
        ValueError: If inputs fail validation or JSON output cannot be parsed
        RuntimeError: If container execution fails
    """
    result = TestExecutionResult(test_name, sut_name, image)

    # Extract system_under_test for validation and environment handling
    sut_params = systems_params.get("system_under_test", {})

    try:
        validate_test_execution_inputs(
            test_name, image, sut_name, sut_params, test_params
        )
    except ValueError as e:
        result.error_message = str(e)
        result.success = False
        return result

    # No global fallbacks - users must explicitly specify credentials
    # Build unified systems_params without any automatic fallbacks
    systems_params_with_fallbacks = systems_params.copy()

    # Load environment variables and merge into system parameters if env_file specified
    sut_params = systems_params_with_fallbacks["system_under_test"]
    if "env_file" in sut_params:
        env_file_path = sut_params["env_file"]
        if os.path.exists(env_file_path):
            try:
                env_vars = dotenv_values(env_file_path)
                # Add BASE_URL and API_KEY from env_file to system parameters if not already present
                if "base_url" not in sut_params and "BASE_URL" in env_vars:
                    sut_params["base_url"] = env_vars["BASE_URL"]
                if "api_key" not in sut_params and "API_KEY" in env_vars:
                    sut_params["api_key"] = env_vars["API_KEY"]
                DBOS.logger.info(f"Loaded environment variables from {env_file_path}")
            except Exception as e:
                DBOS.logger.warning(
                    f"Failed to load environment file {env_file_path}: {e}"
                )
        else:
            DBOS.logger.warning(f"Specified environment file {env_file_path} not found")

    # Prepare command line arguments
    try:
        systems_params_json = json.dumps(systems_params_with_fallbacks)
        test_params_json = json.dumps(test_params)
        command_args = [
            "--systems-params",
            systems_params_json,
            "--test-params",
            test_params_json,
        ]
    except (TypeError, ValueError) as e:
        result.error_message = f"Failed to serialize configuration to JSON: {e}"
        result.success = False
        return result

    # Prepare environment variables from system_under_test params (reuse loaded env_vars)
    container_env = {
        "OUTPUT_MOUNT_PATH": str(OUTPUT_MOUNT_PATH),
        "INPUT_MOUNT_PATH": str(INPUT_MOUNT_PATH),
    }

    # Load environment variables only if explicitly specified via env_file
    if "env_file" in sut_params:
        env_file_path = sut_params["env_file"]
        if os.path.exists(env_file_path):
            try:
                env_vars = dotenv_values(env_file_path)
                # Filter out None values to ensure all env vars are strings
                filtered_env_vars = {k: v for k, v in env_vars.items() if v is not None}
                container_env.update(filtered_env_vars)
                DBOS.logger.info(f"Loaded environment variables from {env_file_path}")
            except Exception as e:
                DBOS.logger.warning(
                    f"Failed to load environment file {env_file_path}: {e}"
                )
        else:
            DBOS.logger.warning(f"Specified environment file {env_file_path} not found")

    # Pass through explicit API key if specified
    if "api_key" in sut_params:
        container_env["API_KEY"] = sut_params["api_key"]
        DBOS.logger.info("Using direct API key for authentication")

    # Extract manifest to check for host access requirements
    manifest = None
    try:
        manifest = extract_manifest_from_image(image, ContainerConfig.MANIFEST_PATH)
    except Exception as e:
        # Log warning but continue - manifest extraction failure shouldn't stop test execution
        DBOS.logger.warning(f"Failed to extract manifest from {image}: {e}")

    # Configure Docker-in-Docker for containers that require host access
    if manifest and manifest.host_access:
        docker_socket_path = _get_docker_socket_path(env_vars=container_env)
        container_config.run_params.update(
            {
                "cap_drop": ["ALL"],
                "cap_add": ["SYS_ADMIN"],
                "volumes": {
                    docker_socket_path: {
                        "bind": "/var/run/docker.sock",
                        "mode": "rw",
                    }
                },
            }
        )
        # Remove env variable DOCKER_HOST to avoid container looking for host path inside container
        del container_env["DOCKER_HOST"]
        DBOS.logger.info(
            f"Configured Docker-in-Docker for test: {test_name} (image: {image}) using host socket: {docker_socket_path}"
        )

    # Execute container
    result.start_time = time.time()

    # Generate container name: {sut}-{test}-{short_uuid}
    truncated_sut = sut_name.lower().replace(" ", "_")[:25]
    truncated_test = test_name.lower().replace(" ", "_")[:25]
    prefix = f"{truncated_sut}-{truncated_test}"
    container_name = f"{prefix}-{str(uuid.uuid4())[:8]}"

    container_result = run_container_with_args(
        image=image,
        args=command_args,
        environment=container_env,
        container_config=container_config,
        name=container_name,
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
        except (KeyError, AttributeError, TypeError, ValueError) as e:
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
    systems_config: Dict[str, Any],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute a test suite with DBOS durability (tests only, no score card evaluation).

    This workflow:
    1. Validates image availability and extracts manifests
    2. Performs cross-validation of tests, systems, and manifests
    3. Executes tests concurrently with progress tracking
    4. Aggregates results with detailed error reporting

    Args:
        suite_config: Serialized SuiteConfig containing test definitions
        systems_config: Serialized SystemsConfig containing system configurations
        executor_config: Execution parameters controlling concurrency and reporting
        container_config: Container execution configurations

    Returns:
        Execution summary with metadata and individual test results (no score cards) and container results
    """
    workflow_start_time = time.time()

    # unique per-workflow execution
    queue_name = f"test_execution_{DBOS.workflow_id}"

    test_queue = Queue(queue_name, concurrency=executor_config["concurrent_tests"])

    # Parse configurations
    try:
        suite = SuiteConfig(**suite_config)
        systems = SystemsConfig(**systems_config)
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
        }, []
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
        }, []

    try:
        validate_test_volumes(suite)

    except ValueError as e:
        error_msg = f"Volume validation failed: {e}"
        DBOS.logger.error(error_msg)
        return {
            "summary": create_workflow_summary(
                suite_name=suite.suite_name,
                workflow_id=DBOS.workflow_id or "",
                status="VALIDATION_FAILED",
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                execution_time=time.time() - workflow_start_time,
                error=error_msg,
            ),
            "results": [],
        }, []

    console.print(f"\n[bold blue]Executing Test Suite:[/bold blue] {suite.suite_name}")

    # Collect all unique images from test suite
    unique_images = list(set(test.image for test in suite.test_suite))
    DBOS.logger.info(f"Found {len(unique_images)} unique images")

    # Check image availability
    with console.status("[bold blue]Checking image availability...", spinner="dots"):
        image_availability = dbos_check_images_availability(unique_images)

    # Try to pull missing images from Docker Hub/registries
    missing_images = [
        img for img, available in image_availability.items() if not available
    ]
    if missing_images:
        console.print(
            f"[yellow]Warning:[/yellow] {len(missing_images)} images not available locally"
        )
        with console.status(
            "[bold blue]Pulling missing images from registry...", spinner="dots"
        ):
            dbos_pull_images(missing_images)

    # Extract manifests from available images (post-pull)
    manifests = {}

    # After pulling, we need to check availability again to include newly pulled images
    if missing_images:
        updated_image_availability = dbos_check_images_availability(unique_images)
        image_availability.update(updated_image_availability)

    # Now get all available images including ones that were just pulled
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
        validation_errors = validate_test_plan(suite, systems, manifests)

    if validation_errors:
        console.print("[red]Validation failed:[/red]")
        for error in validation_errors[: executor_config["max_failures"]]:
            console.print(f"  • {error}")
        if len(validation_errors) > executor_config["max_failures"]:
            remaining = len(validation_errors) - executor_config["max_failures"]
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
        }, []

    # Prepare test execution plan
    test_execution_plan = create_test_execution_plan(suite, systems, image_availability)
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
        }, []

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
                    test_plan["systems_params"],
                    test_plan["test_params"],
                    container_config,
                )
                test_handles.append((handle, test_plan))

            # Collect results as they complete
            all_results = []
            for handle, test_plan in test_handles:
                try:
                    result = handle.get_result()
                except Exception as e:  # Gracefully handle DBOS/HTTP timeouts per test
                    DBOS.logger.error(
                        f"Test execution handle failed for {test_plan['test_name']} (image: {test_plan['image']}): {e}"
                    )
                    # Synthesize a failed TestExecutionResult with timeout semantics
                    result = TestExecutionResult(
                        test_plan["test_name"],
                        test_plan["sut_name"],
                        test_plan["image"],
                    )
                    now = time.time()
                    result.start_time = now
                    result.end_time = now
                    result.exit_code = 137  # convention for forced termination/timeout
                    result.success = False
                    result.error_message = f"Test execution failed: {e}"
                    result.container_output = ""
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
                test_plan["systems_params"],
                test_plan["test_params"],
                container_config,
            )
            test_handles.append((handle, test_plan))

        # Collect results as they complete
        all_results = []
        progress_interval = max(1, test_count // executor_config["progress_interval"])
        for i, (handle, test_plan) in enumerate(test_handles, 1):
            try:
                result = handle.get_result()
            except Exception as e:
                DBOS.logger.error(
                    f"Test execution handle failed for {test_plan['test_name']} (image: {test_plan['image']}): {e}"
                )
                result = TestExecutionResult(
                    test_plan["test_name"], test_plan["sut_name"], test_plan["image"]
                )
                now = time.time()
                result.start_time = now
                result.end_time = now
                result.exit_code = 137
                result.success = False
                result.error_message = f"Test execution failed: {e}"
                result.container_output = ""
            all_results.append(result)
            if i % progress_interval == 0 or i == test_count:
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
        format_failure_summary(failed_results, console, executor_config["max_failures"])

    DBOS.logger.info(
        f"Workflow completed: {successful_tests}/{total_tests} tests passed"
    )

    return {
        "summary": summary,
        "results": [result.metadata_dict() for result in all_results],
    }, [result.container_dict() for result in all_results]


@DBOS.step()
def convert_test_results_to_objects(
    test_results_data: Dict[str, Any],
    test_container_data: List[Dict[str, Any]],
) -> List[TestExecutionResult]:
    """
    Convert test results data back to TestExecutionResult objects.

    Args:
        test_results_data: Test execution results
        test_container_data: Test container results containing 'results' field

    Returns:
        List of TestExecutionResult objects
    """
    test_results = []
    test_results_list = test_results_data.get("results", [])

    for id, result_dict in enumerate(test_results_list):
        metadata = result_dict["metadata"]
        result = TestExecutionResult(
            metadata["test_name"], metadata["sut_name"], metadata["image"]
        )
        result.start_time = metadata["start_time"]
        result.end_time = metadata["end_time"]
        result.success = metadata["success"]
        result.container_id = metadata["container_id"]
        result.exit_code = metadata["exit_code"]
        # case where the logs file was moved and test_container_data is empty
        if id < len(test_container_data):
            result.container_output = test_container_data[id]["container_output"]
            result.test_results = test_container_data[id]["test_results"]
            result.error_message = test_container_data[id]["error_message"]
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
    test_container_data: List[Dict[str, Any]],
    score_card_configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate score cards against existing test results.

    Args:
        test_results_data: Test execution results
        test_container_data: Test container results containing 'results' field
        score_card_configs: List of score card configurations to evaluate

    Returns:
        Updated results with score card evaluation data
    """
    # 1. Convert test results back to TestExecutionResult objects
    test_results = convert_test_results_to_objects(
        test_results_data, test_container_data
    )

    # 2. Evaluate score cards using existing step
    console.print("\n[bold blue]Evaluating score cards...[/bold blue]")
    score_card_evaluation = evaluate_score_card(test_results, score_card_configs)

    # 3. Add score card results to test data
    return add_score_cards_to_results(test_results_data, score_card_evaluation)


@DBOS.workflow()
def run_end_to_end_workflow(
    suite_config: Dict[str, Any],
    systems_config: Dict[str, Any],
    score_card_configs: List[Dict[str, Any]],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute a complete end-to-end workflow: test execution + score card evaluation.

    Args:
        suite_config: Serialized SuiteConfig containing test definitions
        systems_config: Serialized SystemsConfig containing system configurations
        score_card_configs: List of score card configurations to evaluate
        executor_config: Execution parameters controlling concurrency and reporting
        container_config: Container execution configurations

    Returns:
        Complete execution results with test results, score card evaluations and container results
    """
    test_results, container_results = run_test_suite_workflow(
        suite_config, systems_config, executor_config, container_config
    )

    final_results = evaluate_score_cards_workflow(
        test_results, container_results, score_card_configs
    )

    return final_results, container_results


def save_results_to_file_step(results: Dict[str, Any], output_path: str) -> None:
    """Save execution results to a JSON file."""
    try:
        save_results_to_file(results, output_path)
        console.print(f"Results saved to [bold]{output_path}[/bold]")
    except (IOError, OSError, PermissionError) as e:
        console.print(f"[red]Failed to save results:[/red] {e}")
    except (TypeError, ValueError) as e:
        console.print(f"[red]Invalid results data for saving:[/red] {e}")


def save_container_results_to_file_step(
    container_results: List[Dict[str, Any]], output_path: str
) -> None:
    """Save container results to a JSON file."""
    logs_dir = os.getenv("LOGS_PATH", "logs")
    try:
        logs_filename = Path(output_path).name
        if not logs_filename:
            raise ValueError(f"Invalid logs file name: {output_path}")

        Path(logs_dir).mkdir(exist_ok=True)

        logs_path = save_container_results_to_file(
            container_results, logs_dir, logs_filename
        )
        console.print(f"Container results saved to [bold]{logs_path}[/bold]")
    except (IOError, OSError, PermissionError) as e:
        console.print(f"[red]Failed to save container results:[/red] {e}")
    except (TypeError, ValueError) as e:
        console.print(f"[red]Invalid container results data for saving:[/red] {e}")


def start_test_execution(
    suite_path: str,
    systems_path: str,
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
    output_path: Optional[str] = None,
    score_card_configs: Optional[List[Dict[str, Any]]] = None,
    execution_mode: str = "end_to_end",
    test_names: Optional[List[str]] = None,
) -> str:
    """
    Orchestrate test suite execution workflow.

    Handles input validation, configuration loading, and workflow delegation.
    Actual execution logic is handled by dedicated workflow functions.

    Args:
        suite_path: Path to test suite YAML file
        systems_path: Path to systems YAML file
        executor_config: Executor configuration dictionary. Expected keys:
            - "concurrent_tests": int, number of concurrent tests
            - "max_failures": int, max number of failures to display
            - "progress_interval": int, interval for progress updates
        container_config: Container execution configurations
        output_path: Optional path to save results JSON file
        score_card_configs: Optional list of score card configurations to evaluate
        execution_mode: "tests_only" or "end_to_end"
        test_names: Optional list of test names to filter from suite

    Returns:
        Workflow ID for tracking execution

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If configuration files don't exist
        PermissionError: If configuration files cannot be read
    """
    validate_execution_inputs(suite_path, systems_path, execution_mode, output_path)

    try:
        # Load configurations
        suite_config = merge_defaults_into_suite(load_config_file(suite_path))
        systems_config = load_config_file(systems_path)

        # if test_names provided, filter suite_config
        if test_names:
            # Parse test names: handle both repeated flags and comma-separated values
            parsed_test_names = []
            for item in test_names:
                parsed_test_names.extend(
                    [name.strip() for name in item.split(",") if name.strip()]
                )

            original_tests = suite_config.get("test_suite", [])
            available_tests = [t.get("name") for t in original_tests]

            # map lowercase → original name
            available_map = {name.lower(): name for name in available_tests}
            # set of normalized requested names
            requested_set = {name.lower() for name in parsed_test_names}

            missing = requested_set - set(available_map.keys())
            if missing:
                msg_lines = []
                for m in missing:
                    # use original user input instead of lowercase
                    user_input = next(
                        (n for n in parsed_test_names if n.lower() == m), m
                    )
                    suggestions = get_close_matches(m, available_map.keys(), n=1)
                    if suggestions:
                        suggestion = available_map[suggestions[0]]
                        msg_lines.append(
                            f"❌ Test not found: {user_input}\n   Did you mean: {suggestion}"
                        )
                    else:
                        msg_lines.append(f"❌ Test not found: {user_input}")
                raise ValueError("\n".join(msg_lines))

            # filter using lowercase
            suite_config["test_suite"] = [
                t for t in original_tests if t.get("name").lower() in requested_set
            ]

        # Start appropriate workflow based on execution mode
        if execution_mode == "tests_only":
            handle = DBOS.start_workflow(
                run_test_suite_workflow,
                suite_config,
                systems_config,
                executor_config,
                container_config,
            )
        elif execution_mode == "end_to_end":
            if not score_card_configs:
                # Fall back to tests only if no score cards provided
                handle = DBOS.start_workflow(
                    run_test_suite_workflow,
                    suite_config,
                    systems_config,
                    executor_config,
                    container_config,
                )
            else:
                handle = DBOS.start_workflow(
                    run_end_to_end_workflow,
                    suite_config,
                    systems_config,
                    score_card_configs,
                    executor_config,
                    container_config,
                )
        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        if output_path:
            results, container_results = handle.get_result()
            save_results_to_file_step(results, output_path)
            save_container_results_to_file_step(container_results, output_path)
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

        logs_dir = Path(os.getenv("LOGS_PATH", "logs"))
        container_path = logs_dir / input_path
        if container_path.exists():
            with open(container_path, "r") as f:
                test_container_data = json.load(f)
        else:
            test_container_data = []

        handle = DBOS.start_workflow(
            evaluate_score_cards_workflow,
            test_results_data,
            test_container_data,
            score_card_configs,
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
