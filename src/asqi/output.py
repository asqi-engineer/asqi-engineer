import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dbos import DBOS
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from asqi.container_manager import OUTPUT_MOUNT_PATH


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
) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
    """
    Extract test results, generated reports, and generated datasets from container results.

    Args:
        container_results: Parsed JSON dictionary from container output

    Notes:
        Provides backward compatibility for container outputs that do not include
        the `generated_reports` or `generated_datasets` fields

    Returns:
        A tuple containing (test_results, generated_reports, generated_datasets)
    """
    if (
        "test_results" in container_json_output
        and "generated_reports" in container_json_output
    ):
        test_results = container_json_output.get("test_results") or {}
        generated_reports = container_json_output.get("generated_reports") or []
        generated_datasets = container_json_output.get("generated_datasets") or []
    else:
        # Backward compatibility
        test_results = container_json_output
        generated_reports = []
        generated_datasets = []
    return test_results, generated_reports, generated_datasets


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


def _translate_container_path(
    container_path_str: str, host_output_volume: str, item_type: str
) -> str:
    """
    Translate a container path to the host path.

    Args:
        container_path_str: Path inside the container
        host_output_volume: Host output volume path (relative or absolute)
        item_type: Type of item for logging (e.g., "report", "dataset")

    Returns:
        Translated host path (always absolute)
    """
    output_mount_path = Path(OUTPUT_MOUNT_PATH)
    # Convert host_output_volume to absolute path to ensure consistency
    host_output_volume_path = Path(host_output_volume).resolve()
    container_path = Path(container_path_str)

    try:
        relative_path = container_path.relative_to(output_mount_path)
        return str(host_output_volume_path / relative_path)
    except ValueError:
        # Fallback when the path is outside OUTPUT_MOUNT_PATH
        stripped_path = container_path_str.lstrip("/")
        translated_path = str(host_output_volume_path / stripped_path)
        DBOS.logger.warning(
            f"{item_type.capitalize()} path '{container_path}' is outside the container "
            f"output mount path '{OUTPUT_MOUNT_PATH}', using '{translated_path}'"
        )
        return translated_path


def translate_report_paths(generated_reports: list, host_output_volume: str) -> None:
    """
    Translate the test container report path to the host path for each report.

    Args:
        generated_reports: List of generated report dictionaries with `report_path`
        host_output_volume: String path to the host output volume
    """
    if not host_output_volume:
        return

    for report in generated_reports:
        report_path_str = report.get("report_path", "")
        if report_path_str:
            report["report_path"] = _translate_container_path(
                report_path_str, host_output_volume, "report"
            )


def translate_dataset_paths(generated_datasets: list, host_output_volume: str) -> None:
    """
    Translate the container dataset path to the host path for each dataset.

    Args:
        generated_datasets: List of generated dataset dictionaries with `dataset_path`
        host_output_volume: String path to the host output volume
    """
    if not host_output_volume:
        return

    for dataset in generated_datasets:
        dataset_path_str = dataset.get("dataset_path", "")
        if dataset_path_str:
            dataset["dataset_path"] = _translate_container_path(
                dataset_path_str, host_output_volume, "dataset"
            )


def _verify_and_display_output_item(
    path_str: str,
    item_name: str,
    item_context: str,
    item_type: str = "file",
    metadata: Dict[str, Any] | None = None,
) -> bool:
    """
    Verify file exists and display formatted message.

    Args:
        path_str: Path to the file
        item_name: Name of the item (dataset name, report name, etc.)
        item_context: Context identifier (job name, indicator id, etc.)
        item_type: Type description for display (e.g., "dataset", "report")
        metadata: Optional metadata dict to display

    Returns:
        True if file exists, False otherwise
    """
    console = Console()

    try:
        path = Path(path_str)
        if path.exists() and path.is_file():
            # Build metadata string if provided
            metadata_str = ""
            if metadata:
                metadata_parts = [f"{k}: {v}" for k, v in metadata.items()]
                metadata_str = (
                    f" ({', '.join(metadata_parts)})" if metadata_parts else ""
                )

            console.print(
                f"{item_context}: {item_type.capitalize()} [bold]{item_name}[/bold] "
                f"saved to [bold magenta]{path_str}[/bold magenta]{metadata_str}"
            )
            return True
        else:
            console.print(
                f"{item_context}: {item_type.capitalize()} [bold]{item_name}[/bold] "
                f"[red]{path.name}[/red] is missing. Expected path: [bold magenta]{path_str}[/bold magenta]"
            )
            return False
    except (TypeError, ValueError, OSError) as e:
        console.print(
            f"{item_context}: Invalid {item_type} path for [bold]{item_name}[/bold]: "
            f"[red]{path_str}[/red] ({str(e)})"
        )
        return False


def display_score_card_reports(all_evaluations: List[Dict[str, Any]]) -> None:
    """
    Display information about all generated reports referenced in score card evaluations.

    Args:
        all_evaluations: List of score card evaluation results
    """
    if not all_evaluations:
        return

    console = Console()
    console.print("\n[bold blue]Verifying generated reports...[/bold blue]")
    reports_count = 0

    for evaluation in all_evaluations:
        indicator_id = evaluation.get("indicator_id", "")
        report_paths = evaluation.get("report_paths") or []

        for report_path_str in report_paths:
            # Extract report name from path for display
            report_name = Path(report_path_str).name
            context = f"Indicator id [bold]'{indicator_id}'[/bold]"

            if _verify_and_display_output_item(
                report_path_str, report_name, context, "report"
            ):
                reports_count += 1

    if reports_count == 0:
        console.print("No reports were generated")


def display_generated_datasets(all_results: List[Dict[str, Any]]) -> None:
    """
    Display information about all generated datasets from test/generation job results.

    Args:
        all_results: List of test execution or generation job results
    """
    console = Console()
    console.print("\n[bold blue]Verifying generated datasets...[/bold blue]")
    datasets_count = 0

    # Collect all datasets from all results
    for result in all_results:
        generated_datasets = result.get("generated_datasets", [])
        if not generated_datasets:
            continue

        job_name = result.get("metadata", {}).get("test_name") or result.get(
            "metadata", {}
        ).get("job_id", "unknown")

        for dataset in generated_datasets:
            dataset_name = dataset.get("dataset_name", "unnamed")
            dataset_path = dataset.get("dataset_path", "")
            dataset_type = dataset.get("dataset_type", "unknown")

            if dataset_path:
                # Prepare metadata for display
                metadata = {}
                if "num_rows" in dataset:
                    metadata["num_rows"] = f"{dataset['num_rows']} rows"
                if "format" in dataset:
                    metadata["format"] = dataset["format"]

                # Add type to context for clarity
                context = f"Job [bold]'{job_name}'[/bold]"
                type_label = (
                    f"dataset ({dataset_type})"
                    if dataset_type != "unknown"
                    else "dataset"
                )

                if _verify_and_display_output_item(
                    dataset_path, dataset_name, context, type_label, metadata
                ):
                    datasets_count += 1

    if datasets_count == 0:
        console.print("No datasets were generated")
