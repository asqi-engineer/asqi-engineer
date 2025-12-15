import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def parse_container_json_output(output: str) -> Dict[str, Any]:
    """
    Extract JSON from container output with robust parsing.

    Args:
        output: Raw container output string

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found in output or output is empty
    """
    output = output.strip()

    if not output:
        raise ValueError(
            "Empty container output - test container produced no output (check container logs for details)"
        )

    # Try direct parsing first
    if output.startswith("{") and output.endswith("}"):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

    # Strategy: Find the LAST complete JSON object in the output
    # This handles cases where logs are mixed with JSON output
    lines = output.split("\n")

    # Try parsing from each line that starts with { to the end
    # Start from the end and work backwards to find the last valid JSON
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("{"):
            json_str = "\n".join(lines[i:])
            end = json_str.rfind("}")
            if end == -1:
                raise ValueError(f"JSON object not properly closed: {json_str}")
            json_str = json_str[: end + 1]
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"No valid JSON found in container output. Output preview: '{output[:100]}{'...' if len(output) > 100 else ''}'"
    )


def extract_container_json_output_fields(
    container_json_output: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Extract test results and technical reports from container results.

    Args:
        container_results: Parsed JSON dictionary from container output

    Notes:
        Provides backward compatibility for container outputs that do not include the `technical_reports` field

    Returns:
        A dictionary containing the test results and a list of technical reports
    """
    if (
        "test_results" in container_json_output
        and "technical_reports" in container_json_output
    ):
        test_results = container_json_output.get("test_results") or {}
        technical_reports = container_json_output.get("technical_reports") or []
    else:
        test_results = container_json_output
        technical_reports = []
    return test_results, technical_reports


def create_test_execution_progress(console: Console) -> Progress:
    """
    Create a progress bar for test execution tracking.

    Args:
        test_count: Total number of tests to execute
        console: Rich console instance

    Returns:
        Configured Progress instance
    """
    return Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
    )


def format_execution_summary(
    total_tests: int, successful_tests: int, failed_tests: int, execution_time: float
) -> tuple[str, str]:
    """
    Format execution summary with appropriate styling.

    Args:
        total_tests: Total number of tests executed
        successful_tests: Number of successful tests
        failed_tests: Number of failed tests
        execution_time: Total execution time in seconds

    Returns:
        Tuple of (status_color, formatted_message)
    """
    success_rate = successful_tests / total_tests if total_tests > 0 else 0.0

    status_color = (
        "green" if failed_tests == 0 else "yellow" if successful_tests > 0 else "red"
    )

    message = (
        f"{successful_tests}/{total_tests} tests passed "
        f"({success_rate:.0%}) in {execution_time:.1f}s"
    )

    return status_color, message


def format_failure_summary(
    failed_results: List, console: Console, max_displayed: int = 3
) -> None:
    """
    Display summary of failed tests.

    Args:
        failed_results: List of failed test results
        console: Rich console instance
        max_displayed: Maximum number of failures to display
    """
    if not failed_results:
        return

    console.print("\n[red]Failed tests:[/red]")
    for result in failed_results[:max_displayed]:
        error_msg = (
            result.error_message
            or f"Test id '{result.test_id}' returned failure status (exit code: {result.exit_code})"
        )
        console.print(
            f"  • id: {result.test_id} (system under test: {result.sut_name}): {error_msg}"
        )

    if len(failed_results) > max_displayed:
        remaining = len(failed_results) - max_displayed
        console.print(f"  • ... and {remaining} more failures")


def create_workflow_summary(
    suite_name: str,
    workflow_id: str,
    status: str,
    total_tests: int,
    successful_tests: int,
    failed_tests: int,
    execution_time: float,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create standardized workflow summary dictionary.

    Args:
        suite_name: Name of the test suite
        workflow_id: DBOS workflow ID
        status: Execution status
        total_tests: Total number of tests
        successful_tests: Number of successful tests
        failed_tests: Number of failed tests
        execution_time: Total execution time
        **kwargs: Additional summary fields

    Returns:
        Standardized summary dictionary
    """

    summary = {
        "suite_name": suite_name,
        "workflow_id": workflow_id,
        "status": status,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
        "total_execution_time": execution_time,
        "timestamp": datetime.now().isoformat(),
    }

    # Add any additional fields
    summary.update(kwargs)

    return summary
