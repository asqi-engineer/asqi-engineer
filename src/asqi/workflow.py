import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dbos import DBOS, DBOSConfig, Queue
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
from asqi.schemas import GradingPolicy, Manifest, SuiteConfig, SUTsConfig
from asqi.validation import (
    create_test_execution_plan,
    validate_manifests_against_tests,
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
    """Validate that all tests can be executed with available manifests."""
    return validate_manifests_against_tests(suite, suts, manifests)


@DBOS.step()
def execute_single_test(
    test_name: str,
    image: str,
    sut_name: str,
    sut_config: Dict[str, Any],
    test_params: Dict[str, Any],
) -> TestExecutionResult:
    """Execute a single test in a Docker container."""
    result = TestExecutionResult(test_name, sut_name, image)

    # Prepare command line arguments
    sut_config_json = json.dumps(sut_config)
    test_params_json = json.dumps(test_params)
    command_args = ["--sut-config", sut_config_json, "--test-params", test_params_json]

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
            result.error_message = f"Failed to parse JSON output: {e}"
            result.success = False
    else:
        result.success = False

    # Log failures for debugging
    if not result.success:
        DBOS.logger.error(f"Test failed: {test_name} - {result.error_message}")

    return result


@DBOS.step()
def evaluate_grading_policies(
    test_results: List[TestExecutionResult], policy_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Evaluate grading policies against test execution results."""
    from asqi.policy_engine import PolicyEngine

    if not policy_configs:
        return []

    policy_engine = PolicyEngine()
    all_evaluations = []

    for policy_config in policy_configs:
        try:
            # Parse policy configuration
            policy = GradingPolicy(**policy_config)

            # Evaluate policy against test results
            policy_evaluations = policy_engine.evaluate_policy(test_results, policy)

            # Add policy name to each evaluation
            for evaluation in policy_evaluations:
                evaluation["policy_name"] = policy.policy_name

            all_evaluations.extend(policy_evaluations)

            DBOS.logger.info(
                f"Evaluated policy '{policy.policy_name}' with {len(policy_evaluations)} individual evaluations"
            )

        except Exception as e:
            error_result = {
                "policy_name": policy_config.get("policy_name", "unknown"),
                "error": f"Failed to evaluate policy: {e}",
                "indicator_name": "POLICY_ERROR",
                "test_name": "N/A",
                "sut_name": "N/A",
                "outcome": None,
                "metric_value": None,
            }
            all_evaluations.append(error_result)
            DBOS.logger.error(f"Failed to evaluate policy: {e}")

    return all_evaluations


@DBOS.workflow()
def run_test_suite_workflow(
    suite_config: Dict[str, Any],
    suts_config: Dict[str, Any],
    policy_configs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Execute a complete test suite with DBOS durability.

    This workflow:
    1. Validates image availability and extracts manifests
    2. Performs cross-validation of tests, SUTs, and manifests
    3. Executes tests concurrently with progress tracking
    4. Aggregates results with detailed error reporting

    Args:
        suite_config: Serialized SuiteConfig containing test definitions
        suts_config: Serialized SUTsConfig containing SUT configurations

    Returns:
        Execution summary with metadata and individual test results
    """
    workflow_start_time = time.time()

    # Parse configurations
    try:
        suite = SuiteConfig(**suite_config)
        suts = SUTsConfig(**suts_config)
    except Exception as e:
        DBOS.logger.error(f"Configuration parsing failed: {e}")
        return {
            "summary": create_workflow_summary(
                suite_name="unknown",
                workflow_id=DBOS.workflow_id or "",
                status="CONFIG_ERROR",
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                execution_time=time.time() - workflow_start_time,
                error=str(e),
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
        with create_test_execution_progress(test_count, console) as progress:
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
                except Exception as e:
                    print(f"Warning: Failed to update progress: {e}")

    except Exception:
        # Fallback to simple execution without progress bar
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

    # Evaluate grading policies if provided
    policy_evaluation = []
    if policy_configs:
        console.print("\n[bold blue]Evaluating grading policies...[/bold blue]")
        policy_evaluation = evaluate_grading_policies(all_results, policy_configs)

        if policy_evaluation:
            # Group by policy name for display
            policies_by_name = {}
            for evaluation in policy_evaluation:
                policy_name = evaluation.get("policy_name", "unknown")
                if policy_name not in policies_by_name:
                    policies_by_name[policy_name] = []
                policies_by_name[policy_name].append(evaluation)

            # Display summary for each policy
            for policy_name, evaluations in policies_by_name.items():
                passed = sum(
                    1
                    for e in evaluations
                    if e.get("outcome")
                    and e["outcome"].upper()
                    in ["PASS", "A", "GOOD", "SUCCESS", "EXCELLENT", "FAST"]
                )
                failed = sum(
                    1
                    for e in evaluations
                    if e.get("outcome")
                    and e["outcome"].upper()
                    not in ["PASS", "A", "GOOD", "SUCCESS", "EXCELLENT", "FAST"]
                )
                errors = sum(1 for e in evaluations if e.get("error"))

                if errors > 0:
                    status_color = "red"
                    status_text = (
                        f"ERRORS ({passed} passed, {failed} failed, {errors} errors)"
                    )
                elif failed > 0:
                    status_color = "yellow"
                    status_text = f"MIXED ({passed} passed, {failed} failed)"
                else:
                    status_color = "green"
                    status_text = f"ALL PASSED ({passed} passed)"

                console.print(
                    f"[{status_color}]Policy '{policy_name}': {status_text}[/{status_color}]"
                )

    return {
        "summary": summary,
        "results": [result.to_dict() for result in all_results],
        "policy_evaluation": policy_evaluation,
    }


@DBOS.step()
def save_results_to_file_step(results: Dict[str, Any], output_path: str) -> None:
    """Save execution results to a JSON file."""
    try:
        save_results_to_file(results, output_path)
        console.print(f"Results saved to [bold]{output_path}[/bold]")
        DBOS.logger.info(f"Results saved to {output_path}")
    except Exception as e:
        console.print(f"[red]Failed to save results:[/red] {e}")
        DBOS.logger.error(f"Failed to save results to {output_path}: {e}")


def start_test_execution(
    suite_path: str,
    suts_path: str,
    output_path: Optional[str] = None,
    policy_configs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Start test suite execution and return the workflow ID.

    Args:
        suite_path: Path to test suite YAML file
        suts_path: Path to SUTs YAML file
        output_path: Optional path to save results JSON file
        policy_configs: Optional list of policy configurations to evaluate

    Returns:
        Workflow ID for tracking execution
    """
    try:
        # Load configurations
        suite_config = load_config_file(suite_path)
        suts_config = load_config_file(suts_path)

        # Start workflow
        handle = DBOS.start_workflow(
            run_test_suite_workflow, suite_config, suts_config, policy_configs
        )

        # Wait for completion and optionally save results
        if output_path:
            results = handle.get_result()
            save_results_to_file_step(results, output_path)
        else:
            handle.get_result()  # Wait for completion

        return handle.get_workflow_id()

    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found:[/red] {e}")
        raise
    except Exception as e:
        console.print(f"[red]Failed to start test execution:[/red] {e}")
        raise


if __name__ == "__main__":
    DBOS.launch()
