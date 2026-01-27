import argparse
import copy
import json
import os
import select
import subprocess
import sys
import uuid
from pathlib import Path
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
        "results": {"success": False, "error": message},
        "generated_reports": [],
        "generated_datasets": [],
    }


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
            except Exception:
                pass

            # Attempt bind mount
            print(
                f"DEBUG: Attempting to bind mount /output to {host_path}",
                file=sys.stderr,
            )
            try:
                subprocess.run(
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
    parser = argparse.ArgumentParser(description="Garak test container")
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

        results_payload: Dict[str, Any] = {"success": True}

        if job_dir:
            result_json_path = job_dir / "result.json"
            if result_json_path.exists():
                try:
                    harbor_data = json.loads(result_json_path.read_text())
                    results_payload["evaluation_results"] = harbor_data
                except Exception as e:
                    print(f"WARNING: Failed to parse result.json: {e}", file=sys.stderr)

        test_result: Dict[str, Any] = {
            "results": results_payload,
            "agent_config": _redact_systems_params(systems_params),
            "test_params": test_params,
        }

        _print_json(test_result)
        sys.exit(0)

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
    harbor_cmd = ["harbor", "run"]
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
    task = test_params.get("task", None)
    job_name = _job_name(provider, model, dataset)

    jobs_dir_for_harbor, jobs_dir_for_read = _configure_job_dirs(
        os.environ.get("HOST_OUTPUT_PATH")
    )
    harbor_cmd.extend(
        [
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
    )
    if task:
        harbor_cmd.extend(["-t", task])

    env = _configure_provider_env(provider=provider, api_key=api_key, base_url=base_url)

    print(f"DEBUG: About to run command: {' '.join(harbor_cmd)}", file=sys.stderr)

    # --- run command: start ---

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

    # --- run command: end ---

    # Copy results to output mount only if not using Docker-in-Docker
    # (with Docker-in-Docker, results are already in the host path)
    job_specific_dir = jobs_dir_for_harbor / job_name

    # Debug: List contents of job directory
    if job_specific_dir.exists():
        print(f"DEBUG: Contents of {job_specific_dir}:", file=sys.stderr)
        for item in sorted(job_specific_dir.iterdir()):
            print(f"  - {item.name}", file=sys.stderr)

    return job_specific_dir


if __name__ == "__main__":
    main()
