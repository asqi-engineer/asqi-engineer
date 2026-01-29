import argparse
import copy
import json
import os
import select
import subprocess
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


DEFAULT_DATASET = "hello-world@1.0"


def _redact_systems_params(systems_params: Dict[str, Any]) -> Dict[str, Any]:
    redacted = copy.deepcopy(systems_params)
    sut_cfg = redacted.get("system_under_test")
    if isinstance(sut_cfg, dict) and "api_key" in sut_cfg:
        sut_cfg["api_key"] = "REDACTED"
    return redacted


def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2))


def _error_output(message: str) -> Dict[str, Any]:
    # Must conform to ContainerOutput schema (workflow expects results.success)
    return {
        "results": {"success": False, "error": message, "pass_rate": 0.0},
        "generated_reports": [],
        "generated_datasets": [],
    }


def _calculate_task_metrics(job_dir: Path) -> Dict[str, float]:
    """Calculate all task-level metrics in a single pass by reading each result file once.

    Extracts: tokens, execution time, throughput, and latency metrics.

    Args:
        job_dir: Directory containing trial results

    Returns:
        Dict with avg_tokens_per_task, avg_task_execution_time, avg_throughput_tokens_per_sec, avg_latency_ms_per_token
    """
    try:
        token_counts = []
        execution_times = []
        throughputs = []
        latencies = []

        # Find all result.json files in trial subdirectories - read each once
        if job_dir.exists():
            for trial_dir in job_dir.iterdir():
                if trial_dir.is_dir():
                    result_file = trial_dir / "result.json"
                    if result_file.exists():
                        try:
                            trial_data = json.loads(result_file.read_text())

                            # Extract agent result metrics
                            agent_result = trial_data.get("agent_result", {})
                            n_output_tokens = agent_result.get("n_output_tokens", 0)

                            if (
                                isinstance(n_output_tokens, (int, float))
                                and n_output_tokens > 0
                            ):
                                token_counts.append(n_output_tokens)

                            # Extract execution time metrics
                            agent_exec = trial_data.get("agent_execution", {})
                            started_at = agent_exec.get("started_at")
                            finished_at = agent_exec.get("finished_at")

                            if started_at and finished_at:
                                start_time = datetime.fromisoformat(
                                    started_at.replace("Z", "+00:00")
                                )
                                end_time = datetime.fromisoformat(
                                    finished_at.replace("Z", "+00:00")
                                )
                                exec_time_sec = (end_time - start_time).total_seconds()

                                if exec_time_sec > 0 and n_output_tokens > 0:
                                    execution_times.append(exec_time_sec)

                                    # Calculate throughput: tokens/sec
                                    throughput = n_output_tokens / exec_time_sec
                                    throughputs.append(throughput)

                                    # Calculate latency: ms/token
                                    latency = (exec_time_sec * 1000) / n_output_tokens
                                    latencies.append(latency)
                        except (json.JSONDecodeError, OSError, ValueError):
                            continue

        # Calculate averages
        result = {
            "avg_tokens_per_task": 0.0,
            "avg_task_execution_time": 0.0,
            "avg_throughput_tokens_per_sec": 0.0,
            "avg_latency_ms_per_token": 0.0,
        }

        if token_counts:
            result["avg_tokens_per_task"] = round(
                sum(token_counts) / len(token_counts), 2
            )

        if execution_times:
            result["avg_task_execution_time"] = round(
                sum(execution_times) / len(execution_times), 2
            )

        if throughputs:
            result["avg_throughput_tokens_per_sec"] = round(
                sum(throughputs) / len(throughputs), 2
            )

        if latencies:
            result["avg_latency_ms_per_token"] = round(
                sum(latencies) / len(latencies), 2
            )

        return result

    except Exception as e:
        print(f"WARNING: Error calculating task metrics: {e}", file=sys.stderr)
        return {
            "avg_tokens_per_task": 0.0,
            "avg_task_execution_time": 0.0,
            "avg_throughput_tokens_per_sec": 0.0,
            "avg_latency_ms_per_token": 0.0,
        }


def _calculate_pass_rate(harbor_results: Dict[str, Any]) -> float:
    """Calculate pass rate from Harbor result.json.

    Use the dataset-level `metrics` -> `mean` value(s) from Harbor aggregated
    `result.json` as the pass rate.

    Args:
        harbor_results: Parsed Harbor result.json data

    Returns:
        Float between 0.0 and 1.0 representing pass rate (rounded to 4 decimals)
    """
    try:
        stats = harbor_results.get("stats", {})
        evals = stats.get("evals", {})

        # Use the first available metrics->mean value found in the aggregated
        # result.json and return it as the pass_rate (rounded to 4 decimals).
        for eval_key, eval_data in evals.items():
            metrics = eval_data.get("metrics", [])
            if metrics and isinstance(metrics, list):
                for m in metrics:
                    if isinstance(m, dict) and "mean" in m:
                        try:
                            return round(float(m["mean"]), 4)
                        except (ValueError, TypeError):
                            # Unable to parse this mean — try next
                            continue

        # No usable mean found
        return 0.0

    except Exception as e:
        print(f"WARNING: Error calculating pass_rate: {e}", file=sys.stderr)
        return 0.0


def _job_name(provider: str, model: str, dataset: str) -> str:
    job_hash = uuid.uuid4().hex[:8]
    model_clean = model.replace("/", "_").replace(":", "_")
    return f"{provider}_{model_clean}_{job_hash}"


def _configure_job_dirs(host_output_path: Optional[str]) -> tuple[Path, Path]:
    """Return (jobs_dir_for_harbor_write, jobs_dir_for_container_read)."""
    if host_output_path:
        host_path = Path(host_output_path)

        # If we are in a container with /output mounted, try to bind mount
        # the internal host path to /output to solve Split Brain.
        # Use bind mount instead of symlink to preserve the path structure
        # so that Docker Host sees the correct path, preventing "Mounts denied".
        if Path("/output").exists():
            # Create the destination directory if it doesn't exist
            host_path.mkdir(parents=True, exist_ok=True)

            # Check if already mounted (simple check)
            try:
                # This is a heuristic. Proper check would involve /proc/mounts
                if host_path.exists() and any(host_path.iterdir()):
                    # If it has content, it might be the volume or previous run.
                    # We can't easily distinguish.
                    pass
            except Exception as e:
                print(
                    f"DEBUG: Mount check heuristic failed (non-critical): {e}",
                    file=sys.stderr,
                )

            # Attempt bind mount
            print(
                f"DEBUG: Attempting to bind mount /output to {host_path}",
                file=sys.stderr,
            )
            try:
                # Safe to ignore Bandit S603/S607: using list form (not shell=True)
                # prevents shell injection, and "mount" is a system command passed
                # explicitly without user input manipulation.
                subprocess.run(  # noqa: S603,S607
                    ["mount", "--bind", "/output", str(host_path)],
                    check=True,
                    capture_output=True,
                )
                print("DEBUG: Bind mount successful", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(
                    f"WARNING: Bind mount failed: {e.stderr.decode().strip()}. Falling back to symlink (may fail verification)",
                    file=sys.stderr,
                )
                # Fallback to symlink if bind mount fails (e.g. unprivileged)
                if not host_path.exists() or (
                    host_path.is_dir() and not any(host_path.iterdir())
                ):
                    try:
                        if host_path.exists():
                            host_path.rmdir()
                        host_path.symlink_to("/output")
                        print(
                            f"DEBUG: Symlinked {host_path} -> /output", file=sys.stderr
                        )
                    except Exception as ex:
                        print(
                            f"WARNING: Fallback symlink failed: {ex}", file=sys.stderr
                        )

        # Harbor must write to the host path for nested docker mounts.
        jobs_dir_for_harbor = Path(host_output_path) / "harbor"
        jobs_dir_for_harbor.mkdir(parents=True, exist_ok=True)

        # But *this* container should read via the mounted output path.
        if Path("/output").exists():
            jobs_dir_for_read = Path("/output") / "harbor"
            jobs_dir_for_read.mkdir(parents=True, exist_ok=True)
        else:
            jobs_dir_for_read = jobs_dir_for_harbor

        print(
            f"DEBUG: Using host output path for Docker-in-Docker: {jobs_dir_for_harbor}",
            file=sys.stderr,
        )
        return jobs_dir_for_harbor, jobs_dir_for_read

    # If we have a mounted output directory, use it directly to avoid copying logic issues
    # and race conditions with file flushing.
    if Path("/output").exists():
        jobs_dir = Path("/output") / "harbor"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        return jobs_dir, jobs_dir

    # Fallback to local directory (will be copied to output mount later)
    jobs_dir = Path("/app/jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir, jobs_dir


def _configure_provider_env(
    provider: str, api_key: str, base_url: Optional[str]
) -> Dict[str, str]:
    env = os.environ.copy()
    match provider:
        case "codex":
            if base_url:
                env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = api_key
        case "qwen-coder":
            if not base_url:
                raise ValueError("Missing base_url for qwen-coder provider")
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = api_key
        case "claude-code":
            if base_url:
                env["ANTHROPIC_BASE_URL"] = base_url
                env["ANTHROPIC_AUTH_TOKEN"] = api_key
                env["ANTHROPIC_API_KEY"] = api_key
            else:
                env["ANTHROPIC_API_KEY"] = api_key
        case "opencode":
            env["OPENAI_API_KEY"] = api_key
        case "goose":
            if base_url:
                env["OPENAI_HOST"] = base_url
            env["OPENAI_API_KEY"] = api_key
        case "aider":
            if base_url:
                env["OPENAI_API_BASE"] = base_url
            env["OPENAI_API_KEY"] = api_key
        case "cline-cli":
            if base_url:
                env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_KEY"] = api_key
        case _:
            raise ValueError(f"Unsupported provider: {provider}")
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Harbor evaluation test container with batch support"
    )
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        sut_params = systems_params.get("system_under_test", {})
        if not sut_params:
            raise ValueError("Missing system_under_test in systems_params")

        sut_type = sut_params.get("type")
        if sut_type != "agent_cli":
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        job_dir = run_harbor(sut_params=sut_params, test_params=test_params)

        # Parse Harbor results - read aggregated result.json once
        success = True
        pass_rate = 0.0
        n_total_trials = 0
        n_errors = 0
        avg_tokens_per_task = 0.0
        avg_task_execution_time = 0.0
        avg_throughput_tokens_per_sec = 0.0
        avg_latency_ms_per_token = 0.0

        if job_dir:
            result_json_path = job_dir / "result.json"
            if result_json_path.exists():
                try:
                    # Read aggregated result.json once
                    harbor_data = json.loads(result_json_path.read_text())

                    # Extract all aggregated metrics from single read
                    n_total_trials = harbor_data.get("n_total_trials", 0)
                    stats = harbor_data.get("stats", {})
                    n_errors = stats.get("n_errors", 0)

                    # Calculate pass_rate from aggregated data
                    pass_rate = _calculate_pass_rate(harbor_data)

                    # Calculate all task-level metrics in single pass
                    task_metrics = _calculate_task_metrics(job_dir)
                    avg_tokens_per_task = task_metrics["avg_tokens_per_task"]
                    avg_task_execution_time = task_metrics["avg_task_execution_time"]
                    avg_throughput_tokens_per_sec = task_metrics[
                        "avg_throughput_tokens_per_sec"
                    ]
                    avg_latency_ms_per_token = task_metrics["avg_latency_ms_per_token"]

                    # If there are errors, mark as partial success
                    if n_errors > 0:
                        success = True  # Harbor still ran, just had some failures

                    print(f"DEBUG: Calculated pass_rate: {pass_rate}", file=sys.stderr)
                    print(
                        f"DEBUG: Total trials: {n_total_trials}, Errors: {n_errors}",
                        file=sys.stderr,
                    )
                    print(
                        f"DEBUG: Average tokens per task: {avg_tokens_per_task}",
                        file=sys.stderr,
                    )
                    print(
                        f"DEBUG: Average task execution time: {avg_task_execution_time}s",
                        file=sys.stderr,
                    )
                    print(
                        f"DEBUG: Average throughput: {avg_throughput_tokens_per_sec} tokens/sec",
                        file=sys.stderr,
                    )
                    print(
                        f"DEBUG: Average latency: {avg_latency_ms_per_token} ms/token",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"WARNING: Failed to parse result.json: {e}", file=sys.stderr)
                    success = False
            else:
                success = False
                print(
                    f"WARNING: No result.json found at {result_json_path}",
                    file=sys.stderr,
                )
        else:
            success = False
            print("WARNING: No job directory returned", file=sys.stderr)

        # Prepare result with harbor metrics
        test_result = {
            "success": success,
            "pass_rate": pass_rate,
            "n_total_trials": n_total_trials,
            "n_errors": n_errors,
            "avg_tokens_per_task": avg_tokens_per_task,
            "avg_task_execution_time": avg_task_execution_time,
            "avg_throughput_tokens_per_sec": avg_throughput_tokens_per_sec,
            "avg_latency_ms_per_token": avg_latency_ms_per_token,
        }

        # Save output.json to output volume
        try:
            output_mount_path = Path(os.environ.get("OUTPUT_MOUNT_PATH", "/output"))
            output_json_path = output_mount_path / "output.json"
            with open(output_json_path, "w") as f:
                json.dump(test_result, f, indent=2)
            print(f"DEBUG: Output saved to {output_json_path}", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not save output.json: {e}", file=sys.stderr)

        _print_json(test_result)
        sys.exit(0 if success else 1)

    except json.JSONDecodeError as e:
        _print_json(_error_output(f"Invalid JSON in arguments: {e}"))
        sys.exit(1)
    except subprocess.TimeoutExpired:
        _print_json(_error_output("Harbor execution timed out"))
        sys.exit(1)
    except Exception as e:
        _print_json(_error_output(f"Unexpected error: {e}"))
        sys.exit(1)


def run_harbor(sut_params, test_params):
    """Run Harbor with support for single task or multiple tasks via job config.

    Supports:
    - Single task: harbor run -d <dataset> -a <agent> -m <model> -t <task>
    - Multiple tasks: Passed via job config file (recommended for batch evaluation)
    """
    provider = sut_params.get("provider")
    model = sut_params.get("model")
    api_key = sut_params.get("api_key")
    base_url = sut_params.get("base_url")

    if not provider:
        raise ValueError("Missing provider in systems_params")
    if not model:
        raise ValueError("Missing model in systems_params")
    if not api_key:
        raise ValueError("Missing api_key in systems_params")

    dataset = test_params.get("dataset", DEFAULT_DATASET)
    tasks = test_params.get("tasks")
    job_name = _job_name(provider, model, dataset)

    jobs_dir_for_harbor, jobs_dir_for_read = _configure_job_dirs(
        os.environ.get("HOST_OUTPUT_PATH")
    )

    # Build harbor command
    harbor_cmd = [
        "harbor",
        "run",
        "--agent",
        provider,
        "--model",
        model,
        "--env",
        "docker",
        "--dataset",
        dataset,
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir_for_harbor),
    ]

    # Add task(s)
    if tasks:
        if isinstance(tasks, list):
            # Multiple tasks - add each one
            for task in tasks:
                harbor_cmd.extend(["-t", task])
        else:
            # Single task as string
            harbor_cmd.extend(["-t", tasks])

    env = _configure_provider_env(provider=provider, api_key=api_key, base_url=base_url)

    print(f"DEBUG: About to run command: {' '.join(harbor_cmd)}", file=sys.stderr)

    # Run harbor with real-time output and capture
    process = subprocess.Popen(
        harbor_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0,  # Unbuffered for real-time character output
        universal_newlines=True,
    )

    stdout_content = []
    stderr_content = []

    # handle streaming progress bar
    def handle_output():
        while process.poll() is None:
            ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

            for stream in ready:
                if stream == process.stdout and process.stdout is not None:
                    char = process.stdout.read(1)
                    if char:
                        stdout_content.append(char)
                        sys.stdout.write(char)
                        sys.stdout.flush()
                elif stream == process.stderr and process.stderr is not None:
                    char = process.stderr.read(1)
                    if char:
                        stderr_content.append(char)
                        sys.stderr.write(char)
                        sys.stderr.flush()

        # Read any remaining output
        remaining_stdout = process.stdout.read() if process.stdout is not None else ""
        remaining_stderr = process.stderr.read() if process.stderr is not None else ""

        if remaining_stdout:
            stdout_content.append(remaining_stdout)
            sys.stdout.write(remaining_stdout)
            sys.stdout.flush()

        if remaining_stderr:
            stderr_content.append(remaining_stderr)
            sys.stderr.write(remaining_stderr)
            sys.stderr.flush()

    handle_output()
    process.wait()

    # Use the job name we created to find results
    job_specific_dir = jobs_dir_for_harbor / job_name

    # Debug: List contents of job directory
    if job_specific_dir and job_specific_dir.exists():
        print(f"DEBUG: Contents of {job_specific_dir}:", file=sys.stderr)
        for item in sorted(job_specific_dir.iterdir()):
            print(f"  - {item.name}", file=sys.stderr)

    return job_specific_dir


if __name__ == "__main__":
    main()
