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
    ExecutionMode,
    interpolate_env_vars,
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
from asqi.errors import ReportValidationError
from asqi.output import (
    create_test_execution_progress,
    create_workflow_summary,
    extract_container_json_output_fields,
    format_execution_summary,
    format_failure_summary,
    parse_container_json_output,
    translate_report_paths,
)
from asqi.schemas import (
    AuditResponses,
    Manifest,
    ScoreCard,
    SuiteConfig,
    SystemsConfig,
)
from asqi.validation import (
    build_env_var_error_message,
    create_test_execution_plan,
    validate_execution_inputs,
    validate_indicator_display_reports,
    validate_score_card_inputs,
    validate_test_execution_inputs,
    validate_test_volumes,
    validate_workflow_configurations,
    verify_score_card_reports,
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
}
if oltp_endpoint:
    config["enable_otlp"] = True
    config["otlp_traces_endpoints"] = [oltp_endpoint]
    config["otlp_logs_endpoints"] = [oltp_endpoint]
DBOS(config=config)

# Initialize Rich console and execution queue
console = Console()


def _get_available_images(
    unique_images: List[str],
) -> Tuple[List[str], Dict[str, bool]]:
    """
    Get the list of available Docker images, pulling missing ones as needed.

    Args:
        unique_images: List of docker image names

    Returns:
        List of valid available images and a dictionary of their availability status

    """
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

    # After pulling, we need to check availability again to include newly pulled images
    if missing_images:
        updated_image_availability = dbos_check_images_availability(unique_images)
        image_availability.update(updated_image_availability)

    # Now get all available images including ones that were just pulled
    available_images = [
        img for img, available in image_availability.items() if available
    ]
    return available_images, image_availability


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

    def __init__(self, test_name: str, test_id: str, sut_name: str, image: str):
        self.test_id = test_id
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
        self.generated_reports: List[Dict[str, Any]] = []

    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def result_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/reporting."""
        return {
            "metadata": {
                "test_id": self.test_id,
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
            "generated_reports": self.generated_reports,
        }

    def container_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/reporting."""
        return {
            "test_id": self.test_id,
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
def extract_manifests_step(images: List[str]) -> Dict[str, Manifest]:
    """
    Extract and parse manifests from a list of Docker images.

    Args:
        images: List of Docker image names

    Returns:
        Dictionary mapping Image name to Manifest
    """
    manifests = {}

    with console.status("[bold blue]Extracting manifests...", spinner="dots"):
        for image in images:
            manifest = extract_manifest_from_image(image, ContainerConfig.MANIFEST_PATH)
            if manifest:
                manifests[image] = manifest
            else:
                DBOS.logger.warning(f"Failed to extract manifest from {image}")

    return manifests


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
def validate_indicator_display_reports_step(
    suite: SuiteConfig, manifests: Dict[str, Manifest], score_cards: List[ScoreCard]
) -> List[str]:
    """
    DBOS step wrapper for comprehensive score card report validation.

    Delegates to validation.py for the actual validation logic.
    This step exists to provide DBOS durability for validation results.

    Args:
        suite: Test suite configuration (pre-validated)
        manifests: Available manifests (pre-validated)
        score_cards: List of score card configurations (pre-validated)

    Returns:
        List of validation error messages
    """
    test_id_to_image = {test.id: test.image for test in suite.test_suite}

    return validate_indicator_display_reports(manifests, score_cards, test_id_to_image)


@DBOS.step()
def execute_single_test(
    test_name: str,
    test_id: str,
    image: str,
    sut_name: str,
    systems_params: Dict[str, Any],
    test_params: Dict[str, Any],
    container_config: ContainerConfig,
    env_file: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
) -> TestExecutionResult:
    """Execute a single test in a Docker container.

    Focuses solely on test execution. Input validation is handled separately
    in validation.py to follow single responsibility principle.

    Args:
        test_name: Name of the test to execute (pre-validated)
        test_id: Unique ID of the test to execute (pre-validated)
        image: Docker image to run (pre-validated)
        sut_name: Name of the system under test (pre-validated)
        systems_params: Dictionary containing system_under_test and other systems (pre-validated)
        test_params: Parameters for the test (pre-validated)
        container_config: Container execution configurations
        env_file: Optional path to .env file for test-level environment variables
        environment: Optional dictionary of environment variables for the test

    Returns:
        TestExecutionResult containing execution metadata and results

    Raises:
        ValueError: If inputs fail validation or JSON output cannot be parsed
        RuntimeError: If container execution fails
    """
    result = TestExecutionResult(test_name, test_id, sut_name, image)

    # Extract system_under_test for validation and environment handling
    sut_params = systems_params.get("system_under_test", {})

    try:
        validate_test_execution_inputs(
            test_id, image, sut_name, sut_params, test_params
        )
    except ValueError as e:
        result.error_message = str(e)
        result.success = False
        return result

    # No global fallbacks - users must explicitly specify credentials
    # Build unified systems_params without any automatic fallbacks
    systems_params_with_fallbacks = systems_params.copy()

    # Load environment variables and merge into system parameters if env_file specified
    sut_params = systems_params_with_fallbacks.get("system_under_test", {})
    if "env_file" in sut_params and sut_params["env_file"]:
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

    # Prepare environment variables - multi-level priority:
    # Base paths, System-level env_file, Test-level env_file, Test-level environment dict
    container_env = {
        "OUTPUT_MOUNT_PATH": str(OUTPUT_MOUNT_PATH),
        "INPUT_MOUNT_PATH": str(INPUT_MOUNT_PATH),
    }

    # Load environment variables from system_under_test env_file (backward compatibility)
    if "env_file" in sut_params and sut_params["env_file"]:
        env_file_path = sut_params["env_file"]
        if os.path.exists(env_file_path):
            try:
                env_vars = dotenv_values(env_file_path)
                # Filter out None values to ensure all env vars are strings
                filtered_env_vars = {k: v for k, v in env_vars.items() if v is not None}
                container_env.update(filtered_env_vars)
                DBOS.logger.info(
                    f"Loaded environment variables from system-level env_file: {env_file_path}"
                )
            except Exception as e:
                DBOS.logger.warning(
                    f"Failed to load system-level environment file {env_file_path}: {e}"
                )
        else:
            DBOS.logger.warning(
                f"System-level environment file {env_file_path} not found"
            )

    # Load environment variables from test-level env_file
    if env_file:
        if os.path.exists(env_file):
            try:
                test_env_vars = dotenv_values(env_file)
                # Filter out None values
                filtered_test_env_vars = {
                    k: v for k, v in test_env_vars.items() if v is not None
                }
                container_env.update(filtered_test_env_vars)
                DBOS.logger.info(
                    f"Loaded environment variables from test-level env_file: {env_file}"
                )
            except Exception as e:
                DBOS.logger.warning(
                    f"Failed to load test-level environment file {env_file}: {e}"
                )
        else:
            DBOS.logger.warning(f"Test-level environment file {env_file} not found")

    # Merge test-level environment dict (with interpolation support)
    if environment:
        interpolated_env = interpolate_env_vars(environment)
        for key, value in interpolated_env.items():
            container_env[key] = value
            if environment.get(key) != value:
                DBOS.logger.info(f"Interpolated environment variable: {key}")
            else:
                DBOS.logger.info(f"Set environment variable: {key}")

    # Pass through explicit API key if specified
    if "api_key" in sut_params:
        container_env["API_KEY"] = sut_params["api_key"]
        DBOS.logger.info("Using direct API key for authentication")

    # Extract manifest to check for host access requirements and validate environment variables
    manifest = None
    try:
        manifest = extract_manifest_from_image(image, ContainerConfig.MANIFEST_PATH)
    except Exception as e:
        # Log warning but continue - manifest extraction failure shouldn't stop test execution
        DBOS.logger.warning(f"Failed to extract manifest from {image}: {e}")

    # Validate environment variables against manifest requirements
    if manifest and manifest.environment_variables:
        missing_required = []
        missing_optional = []

        for env_var in manifest.environment_variables:
            if env_var.name not in container_env:
                if env_var.required:
                    missing_required.append(env_var)
                else:
                    missing_optional.append(env_var)

        # Fail immediately if required environment variables are missing
        if missing_required:
            error_msg = build_env_var_error_message(missing_required, test_name, image)
            result.error_message = error_msg
            result.success = False
            return result

        # Log warnings for optional missing environment variables
        if missing_optional:
            for env_var in missing_optional:
                DBOS.logger.warning(
                    f"Optional environment variable '{env_var.name}' not provided for test '{test_name}'. "
                    f"{env_var.description or 'No description provided.'}"
                )

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
            f"Configured Docker-in-Docker for test id: {test_id} (image: {image}) using host socket: {docker_socket_path}"
        )

    # Execute container
    result.start_time = time.time()

    # Generate container name: {sut}-{test_id}-{short_uuid}
    if not sut_name:
        sut_name = "data_generation"
    truncated_sut = sut_name.lower().replace(" ", "_")[:25]
    truncated_test_id = test_id.lower()[:25]
    prefix = f"{truncated_sut}-{truncated_test_id}"
    container_name = f"{prefix}-{str(uuid.uuid4())[:8]}"

    container_result = run_container_with_args(
        image=image,
        args=command_args,
        environment=container_env,
        container_config=container_config,
        name=container_name,
        workflow_id=DBOS.workflow_id or "",
    )

    result.end_time = time.time()
    result.container_id = container_result["container_id"]
    result.exit_code = container_result["exit_code"]
    result.container_output = container_result["output"]
    result.error_message = container_result["error"]

    test_results = {}
    generated_reports = []

    # Parse JSON output from container
    if container_result["success"]:
        try:
            parsed_container_results = parse_container_json_output(
                result.container_output
            )
            test_results, generated_reports = extract_container_json_output_fields(
                parsed_container_results
            )
            result.test_results = test_results
            result.generated_reports = generated_reports

            host_output_volume = test_params.get("volumes", {}).get("output", "")
            translate_report_paths(generated_reports, host_output_volume)
            result.success = result.test_results.get("success", False)
        except ValueError as e:
            result.error_message = (
                f"Failed to parse JSON output from test id '{test_id}': {e}"
            )
            result.success = False
            DBOS.logger.error(
                f"JSON parsing failed for test id {test_id}: {result.container_output[:200]}..."
            )
    else:
        result.success = False

    # Log failures for debugging
    if not result.success:
        DBOS.logger.error(f"Test failed, id: {test_id} - {result.error_message}")

    return result


@DBOS.step()
def evaluate_score_card(
    test_results: List[TestExecutionResult],
    score_card_configs: List[Dict[str, Any]],
    audit_responses_data: Optional[Dict[str, Any]] = None,
    execution_mode: ExecutionMode = ExecutionMode.END_TO_END,
) -> List[Dict[str, Any]]:
    """Evaluate score cards against test execution results."""
    from asqi.score_card_engine import ScoreCardEngine

    if not score_card_configs:
        DBOS.logger.warning("No score card configurations provided")
        return []

    score_card_engine = ScoreCardEngine()
    all_evaluations = []

    audit_responses = None
    if audit_responses_data is not None:
        try:
            audit_responses = AuditResponses(**audit_responses_data)
        except ValidationError as e:
            DBOS.logger.error(f"Audit responses validation failed: {e}")
            audit_responses = None

    score_cards = []
    for score_card_config in score_card_configs:
        try:
            score_cards.append(ScoreCard(**score_card_config))
        except ValidationError as e:
            error_result = {
                "score_card_name": score_card_config.get("score_card_name", "unknown"),
                "error": f"Score card validation failed: {e}",
                "indicator_id": "score_card_validation_error",
                "indicator_name": "SCORE CARD VALIDATION ERROR",
                "test_name": "N/A",
                "test_id": "N/A",
                "sut_name": "N/A",
                "outcome": None,
                "metric_value": None,
            }
            all_evaluations.append(error_result)
            DBOS.logger.error(f"Score card validation failed: {e}")
            continue

    if not score_cards:
        return all_evaluations

    test_id_to_image, manifests = _resolve_display_reports_inputs(
        test_results, execution_mode
    )

    for score_card in score_cards:
        try:
            # Validate score card display reports (ExecutionMode.END_TO_END is validated before test execution)
            if execution_mode == ExecutionMode.EVALUATE_ONLY:
                validate_display_reports(manifests, score_card, test_id_to_image)

            # Evaluate score card against test results
            score_card_evaluations = score_card_engine.evaluate_scorecard(
                test_results, score_card, audit_responses
            )

            # Add score card name to each evaluation
            for evaluation in score_card_evaluations:
                evaluation["score_card_name"] = score_card.score_card_name

            all_evaluations.extend(score_card_evaluations)

            DBOS.logger.info(
                f"Evaluated score card '{score_card.score_card_name}' with {len(score_card_evaluations)} individual evaluations"
            )

        except (
            KeyError,
            AttributeError,
            TypeError,
            ValueError,
            ReportValidationError,
        ) as e:
            error_result = {
                "score_card_name": score_card.score_card_name,
                "error": f"Score card evaluation error: {e}",
                "indicator_id": "score_card_evaluation_error",
                "indicator_name": "SCORE CARD EVALUATION ERROR",
                "test_name": "N/A",
                "test_id": "N/A",
                "sut_name": "N/A",
                "outcome": None,
                "metric_value": None,
            }
            all_evaluations.append(error_result)
            DBOS.logger.error(f"Score card evaluation error: {e}")

    verify_score_card_reports(all_evaluations)
    return all_evaluations


@DBOS.workflow()
def run_test_suite_workflow(
    suite_config: Dict[str, Any],
    systems_config: Dict[str, Any],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
    score_card_configs: Optional[List[Dict[str, Any]]] = None,
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
        score_card_configs: Optional list of score card configurations

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
        score_cards = []
        for score_card_config in score_card_configs or []:
            score_cards.append(ScoreCard(**score_card_config))
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

    """Get the list of available Docker images for the test suite."""
    unique_images = list(set(test.image for test in suite.test_suite))
    available_images, image_availability = _get_available_images(unique_images)

    # Extract manifests from available images (post-pull)
    manifests = extract_manifests_step(available_images)

    # Validate test plan
    with console.status("[bold blue]Validating test plan...", spinner="dots"):
        validation_errors = validate_test_plan(suite, systems, manifests)

    # Validate score cards reports
    with console.status(
        "[bold blue]Validating indicators display reports...", spinner="dots"
    ):
        validation_errors.extend(
            validate_indicator_display_reports_step(suite, manifests, score_cards)
        )

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
                    test_plan["test_id"],
                    test_plan["image"],
                    test_plan["sut_name"],
                    test_plan["systems_params"],
                    test_plan["test_params"],
                    container_config,
                    test_plan.get("env_file"),
                    test_plan.get("environment"),
                )
                test_handles.append((handle, test_plan))

            # Collect results as they complete
            all_results = []
            for handle, test_plan in test_handles:
                try:
                    result = handle.get_result()
                except Exception as e:  # Gracefully handle DBOS/HTTP timeouts per test
                    DBOS.logger.error(
                        f"Test execution handle failed for {test_plan['test_id']} (image: {test_plan['image']}): {e}"
                    )
                    # Synthesize a failed TestExecutionResult with timeout semantics
                    result = TestExecutionResult(
                        test_plan["test_name"],
                        test_plan["test_id"],
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
                test_plan["test_id"],
                test_plan["image"],
                test_plan["sut_name"],
                test_plan["systems_params"],
                test_plan["test_params"],
                container_config,
                test_plan.get("env_file"),
                test_plan.get("environment"),
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
                    f"Test execution handle failed for {test_plan['test_id']} (image: {test_plan['image']}): {e}"
                )
                result = TestExecutionResult(
                    test_plan["test_name"],
                    test_plan["test_id"],
                    test_plan["sut_name"],
                    test_plan["image"],
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

    validation_errors = validate_test_container_reports(all_results, manifests)

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
        validation_errors=validation_errors,
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
        "results": [result.result_dict() for result in all_results],
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
        test_container_data: Test container results containing container output and error message

    Returns:
        List of TestExecutionResult objects
    """
    test_results = []
    test_results_list = test_results_data.get("results", [])

    for id, result_dict in enumerate(test_results_list):
        metadata = result_dict["metadata"]
        result = TestExecutionResult(
            metadata["test_name"],
            metadata["test_id"],
            metadata["sut_name"],
            metadata["image"],
        )
        result.start_time = metadata["start_time"]
        result.end_time = metadata["end_time"]
        result.success = metadata["success"]
        result.container_id = metadata["container_id"]
        result.exit_code = metadata["exit_code"]
        result.test_results = result_dict["test_results"]
        result.generated_reports = result_dict["generated_reports"]
        # case where the logs file was moved and test_container_data is empty
        if id < len(test_container_data):
            result.container_output = test_container_data[id]["container_output"]
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
    audit_responses_data: Optional[Dict[str, Any]] = None,
    execution_mode: ExecutionMode = ExecutionMode.END_TO_END,
) -> Dict[str, Any]:
    """
    Evaluate score cards against existing test results.

    Args:
        test_results_data: Test execution results
        test_container_data: Test container results containing container output and error message
        score_card_configs: List of score card configurations to evaluate
        audit_responses_data: Optional dict with manual audit responses
        execution_mode: Execution mode, expected modes:
            - `ExecutionMode.EVALUATE_ONLY`
            - `ExecutionMode.END_TO_END`

    Returns:
        Updated results with score card evaluation data
    """
    # 1. Convert test results back to TestExecutionResult objects
    test_results = convert_test_results_to_objects(
        test_results_data, test_container_data
    )

    # 2. Evaluate score cards using existing step
    console.print("\n[bold blue]Evaluating score cards...[/bold blue]")
    score_card_evaluation = evaluate_score_card(
        test_results, score_card_configs, audit_responses_data, execution_mode
    )

    # 3. Add score card results to test data
    return add_score_cards_to_results(test_results_data, score_card_evaluation)


@DBOS.workflow()
def run_end_to_end_workflow(
    suite_config: Dict[str, Any],
    systems_config: Dict[str, Any],
    score_card_configs: List[Dict[str, Any]],
    executor_config: Dict[str, Any],
    container_config: ContainerConfig,
    audit_responses_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Execute a complete end-to-end workflow: test execution + score card evaluation.

    Args:
        suite_config: Serialized SuiteConfig containing test definitions
        systems_config: Serialized SystemsConfig containing system configurations
        score_card_configs: List of score card configurations to evaluate
        executor_config: Execution parameters controlling concurrency and reporting
        container_config: Container execution configurations
        audit_responses_data: Optional dict with manual audit responses

    Returns:
        Complete execution results with test results, score card evaluations and container results
    """
    test_results, container_results = run_test_suite_workflow(
        suite_config,
        systems_config,
        executor_config,
        container_config,
        score_card_configs,
    )

    final_results = evaluate_score_cards_workflow(
        test_results, container_results, score_card_configs, audit_responses_data
    )

    return final_results, container_results


def validate_test_container_reports(
    all_results: List[TestExecutionResult], manifests: Dict[str, Manifest]
) -> List[str]:
    """
    Validates that the reports returned by the test container exactly match the test container manifest `output_reports` definitions.

    Args:
        all_results: List of test execution results
        manifests: Dictionary linking each image to its manifest

    Returns:
        List of validation error messages or empty list if none found
    """
    validation_errors = []

    for result in all_results:
        if not result.success:
            continue

        result_errors = []

        if not isinstance(result.generated_reports, list):
            result_errors.append("'generated_reports' must be a list")
        else:
            for report in result.generated_reports:
                if not isinstance(report, dict):
                    result_errors.append(
                        "Each report in 'generated_reports' must be a dictionary"
                    )
                    continue

                report_name = report.get("report_name")
                report_type = report.get("report_type")
                report_path_str = report.get("report_path")

                if not report_name or not isinstance(report_name, str):
                    result_errors.append("Report missing valid 'report_name' field")
                    continue
                if not report_type or not isinstance(report_type, str):
                    result_errors.append(
                        f"Report name '{report_name}' missing valid 'report_type' field"
                    )
                    continue
                if not report_path_str or not isinstance(report_path_str, str):
                    result_errors.append(
                        f"Report name '{report_name}' missing valid 'report_path' field"
                    )
                    continue
                try:
                    report_path = Path(report_path_str)
                    if not report_path.exists():
                        result_errors.append(
                            f"Report name '{report_name}' does not exist at path '{report_path_str}'"
                        )
                except (TypeError, ValueError, OSError) as error:
                    result_errors.append(
                        f"Report name '{report_name}' file validation error: {str(error)}"
                    )
            if not result_errors:
                if result.image not in manifests:
                    DBOS.logger.warning(
                        f"No manifest found for image '{result.image}', skipping 'generated_reports' validation for test {result.test_id}"
                    )
                else:
                    manifest = manifests[result.image]
                    manifest_reports = {
                        (report.name, report.type)
                        for report in (manifest.output_reports or [])
                    }
                    container_reports = {
                        (report["report_name"], report["report_type"])
                        for report in result.generated_reports
                    }
                    missing_reports = manifest_reports - container_reports
                    extra_reports = container_reports - manifest_reports

                    if missing_reports or extra_reports:
                        missing_reports = sorted(list(missing_reports))
                        extra_reports = sorted(list(extra_reports))
                        result_errors.append(
                            f"Mismatch between manifest and returned 'generated_reports' for image '{result.image}'. Missing expected reports: {missing_reports}. Extra reports: {extra_reports}."
                        )

                        result.test_results["success"] = False
                        result.test_results["error"] = (
                            "Test results invalidated due to 'generated_reports' mismatch"
                        )
                        result.test_results["missing_reports"] = missing_reports
                        result.test_results["extra_reports"] = extra_reports

        if result_errors:
            result.success = False
            error_message = "| ".join(result_errors)
            result.error_message = error_message
            validation_errors.append(f"Test {result.test_id}: {error_message}")
            DBOS.logger.error(f"Test {result.test_id}: {error_message}")

    return validation_errors


def _resolve_display_reports_inputs(
    test_results: List[TestExecutionResult],
    execution_mode: ExecutionMode,
) -> Tuple[Dict[str, str], Dict[str, Manifest]]:
    """
    Prepares input data needed for display report validation.

    Args:
        test_results: List of test execution results
        execution_mode: Execution mode, expected modes:
            - `ExecutionMode.EVALUATE_ONLY`
            - `ExecutionMode.END_TO_END`

    Returns:
        Links test ids to their image names and manifests by image name
    """
    if execution_mode == ExecutionMode.END_TO_END:
        return {}, {}
    unique_images = list(set(result.image for result in test_results))
    available_images, _ = _get_available_images(unique_images)
    manifests = extract_manifests_step(available_images)
    suite_dict = {
        "suite_name": "evaluation_suite",
        "test_suite": [
            {
                "id": result.test_id,
                "name": result.test_name,
                "image": result.image,
                "systems_under_test": [result.sut_name],
            }
            for result in test_results
        ],
    }
    suite = SuiteConfig(**suite_dict)
    test_id_to_image = {test.id: test.image for test in suite.test_suite}
    return test_id_to_image, manifests


def validate_display_reports(
    manifests: Dict[str, Manifest],
    score_card: ScoreCard,
    test_id_to_image: Dict[str, str],
):
    """
    Validate that score card 'display_reports' are defined in the test container manifests and match the expected structure

    Args:
        manifests: Dictionary linking each image to its manifest
        score_card: Score card configuration to validate
        test_id_to_image: Dictionary linking each test id to the image used

    Raises:
        ReportValidationError: If validation fails
    """
    with console.status(
        "[bold blue]Validating indicators display reports...", spinner="dots"
    ):
        validation_errors = validate_indicator_display_reports(
            manifests, [score_card], test_id_to_image
        )
        if validation_errors:
            errors = ", ".join(validation_errors)
            raise ReportValidationError(errors)


def save_results_to_file_step(results: Dict[str, Any], output_path: str) -> None:
    """Save execution results to a JSON file."""
    try:
        save_results_to_file(results, output_path)
        console.print(f"\nResults saved to [bold]{output_path}[/bold]")
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
    audit_responses_data: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    score_card_configs: Optional[List[Dict[str, Any]]] = None,
    execution_mode: ExecutionMode = ExecutionMode.END_TO_END,
    test_ids: Optional[List[str]] = None,
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
        audit_responses_data: Optional dictionary of audit responses data
        output_path: Optional path to save results JSON file
        score_card_configs: Optional list of score card configurations to evaluate
        execution_mode: Execution mode, expected modes:
            - `ExecutionMode.TESTS_ONLY`
            - `ExecutionMode.END_TO_END`
        test_ids: Optional list of test ids to filter from suite

    Returns:
        Workflow ID for tracking execution

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If configuration files don't exist
        PermissionError: If configuration files cannot be read
    """
    validate_execution_inputs(
        suite_path, systems_path, execution_mode, audit_responses_data, output_path
    )

    try:
        # Load configurations
        suite_config = merge_defaults_into_suite(load_config_file(suite_path))
        systems_config = load_config_file(systems_path)

        # if test_ids provided, filter suite_config
        if test_ids:
            # Parse test ids: handle both repeated flags and comma-separated values
            parsed_test_ids = []
            for item in test_ids:
                parsed_test_ids.extend(
                    [name.strip() for name in item.split(",") if name.strip()]
                )

            original_tests = suite_config.get("test_suite", [])
            available_tests = [t.get("name") for t in original_tests]

            # map lowercase → original name
            available_map = {name.lower(): name for name in available_tests}
            # set of normalized requested names
            requested_set = {name.lower() for name in parsed_test_ids}

            missing = requested_set - set(available_map.keys())
            if missing:
                msg_lines = []
                for m in missing:
                    # use original user input instead of lowercase
                    user_input = next((n for n in parsed_test_ids if n.lower() == m), m)
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
        if execution_mode == ExecutionMode.TESTS_ONLY:
            handle = DBOS.start_workflow(
                run_test_suite_workflow,
                suite_config,
                systems_config,
                executor_config,
                container_config,
            )
        elif execution_mode == ExecutionMode.END_TO_END:
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
                    audit_responses_data,
                )
        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        results, container_results = handle.get_result()
        if output_path:
            save_results_to_file_step(results, output_path)
            save_container_results_to_file_step(container_results, output_path)

        return handle.get_workflow_id()

    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found:[/red] {e}")
        raise


def start_score_card_evaluation(
    input_path: str,
    score_card_configs: List[Dict[str, Any]],
    audit_responses_data: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Orchestrate score card evaluation workflow.

    Handles input validation, data loading, and workflow delegation.
    Actual evaluation logic is handled by dedicated workflow functions.

    Args:
        input_path: Path to JSON file containing test execution results
        score_card_configs: List of score card configurations to evaluate
        audit_responses_data : Optional dictionary of audit responses data
        output_path: Optional path to save updated results JSON file

    Returns:
        Workflow ID for tracking execution

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input file contains invalid JSON
        PermissionError: If input file cannot be read
    """
    validate_score_card_inputs(
        input_path, score_card_configs, audit_responses_data, output_path
    )

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
            audit_responses_data,
            ExecutionMode.EVALUATE_ONLY,
        )

        results = handle.get_result()
        if output_path:
            save_results_to_file_step(results, output_path)

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
