import argparse
import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    """Garak test container entrypoint that interfaces with the ASQI executor."""
    parser = argparse.ArgumentParser(description="Garak test container")
    parser.add_argument(
        "--sut-config", required=True, help="SUT configuration as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        sut_config = json.loads(args.sut_config)
        test_params = json.loads(args.test_params)

        # Validate SUT type
        sut_type = sut_config.get("type")
        if sut_type not in ["llm_api"]:
            raise ValueError(f"Unsupported SUT type: {sut_type}")

        # Extract test parameters
        probes = test_params.get("probes", ["blank"])
        generations = test_params.get("generations", 10)
        config_file = test_params.get("config_file")

        # Create temporary directory for garak outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Build garak command
            garak_cmd = ["garak"]

            # Configure model based on SUT type and config
            if sut_type == "llm_api":
                provider = sut_config.get("config", {}).get("provider", "openai")
                model = sut_config.get("config", {}).get("model", "gpt-3.5-turbo")

                if provider == "openai":
                    garak_cmd.extend(["--model_type", "openai", "--model_name", model])
                elif provider == "huggingface":
                    garak_cmd.extend(
                        ["--model_type", "huggingface", "--model_name", model]
                    )
                else:
                    # Default to treating as a generic API endpoint
                    garak_cmd.extend(["--model_type", "rest", "--model_name", model])

            # Add probes - garak expects comma-separated list
            probe_list = ",".join(probes)
            garak_cmd.extend(["--probes", probe_list])

            # Add generations parameter
            garak_cmd.extend(["--generations", str(generations)])

            # Add parallelism to speed up execution
            garak_cmd.extend(["--parallel_attempts", "8"])

            # Try to limit probe attempts for testing
            if generations == 1:  # If this is a quick test, limit attempts
                garak_cmd.extend(["--probe_options", '{"max_prompts": 10}'])

            # Set report prefix to control output location
            report_prefix = str(output_dir / "garak_report")
            garak_cmd.extend(["--report_prefix", report_prefix])

            # Add custom config file if provided
            if config_file:
                # Assume config_file is provided as a JSON string that we write to a temp file
                config_path = output_dir / "garak_config.yaml"
                with open(config_path, "w") as f:
                    if isinstance(config_file, str):
                        f.write(config_file)
                    else:
                        import yaml

                        yaml.dump(config_file, f)
                garak_cmd.extend(["--config", str(config_path)])

            # Use current environment API keys
            env = os.environ.copy()

            print(
                f"DEBUG: About to run command: {' '.join(garak_cmd)}", file=sys.stderr
            )

            # Run garak with real-time output and capture
            process = subprocess.Popen(
                garak_cmd,
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
                    ready, _, _ = select.select(
                        [process.stdout, process.stderr], [], [], 0.1
                    )

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
                remaining_stdout = (
                    process.stdout.read() if process.stdout is not None else ""
                )
                remaining_stderr = (
                    process.stderr.read() if process.stderr is not None else ""
                )

                if remaining_stdout:
                    stdout_content.append(remaining_stdout)
                    sys.stdout.write(remaining_stdout)
                    sys.stdout.flush()

                if remaining_stderr:
                    stderr_content.append(remaining_stderr)
                    sys.stderr.write(remaining_stderr)
                    sys.stderr.flush()

            handle_output()

            # Wait for process to complete
            process.wait()

            # Create result object
            result = subprocess.CompletedProcess(
                args=garak_cmd,
                returncode=process.returncode,
                stdout="".join(stdout_content),
                stderr="".join(stderr_content),
            )

            # Parse garak results
            success = result.returncode == 0

            # Try to extract metrics from garak output files
            score = 0.0
            vulnerabilities_found = 0
            total_attempts = generations * len(probes)

            # Look for garak report files (should have report_prefix pattern)
            report_files = list(output_dir.glob("garak_report*.jsonl"))
            if not report_files:
                # Fallback: look for any jsonl files
                report_files = list(output_dir.glob("*.jsonl"))

            if report_files:
                try:
                    # Parse the first report file found
                    with open(report_files[0], "r") as f:
                        lines = f.readlines()
                        passed_count = 0
                        failed_count = 0

                        # Count passes and failures from all entries
                        for line in lines:
                            try:
                                entry = json.loads(line.strip())
                                # Garak typically uses 'passed' field or status indicators
                                if (
                                    entry.get("passed", False)
                                    or entry.get("status") == "PASS"
                                ):
                                    passed_count += 1
                                elif (
                                    entry.get("failed", False)
                                    or entry.get("status") == "FAIL"
                                ):
                                    failed_count += 1
                                elif "result" in entry:
                                    # Some garak outputs use result field
                                    if entry["result"] in ["PASS", "passed", True]:
                                        passed_count += 1
                                    elif entry["result"] in ["FAIL", "failed", False]:
                                        failed_count += 1
                            except json.JSONDecodeError:
                                continue

                        total = passed_count + failed_count
                        if total > 0:
                            score = passed_count / total
                            vulnerabilities_found = failed_count
                        else:
                            # Nothing was run - this indicates an error
                            success = False
                            score = 0.0

                except (FileNotFoundError, KeyError, IndexError):
                    # If we can't parse results, treat as failure
                    success = False
                    score = 0.0

            else:
                # No report files generated - this indicates failure
                success = False
                score = 0.0

            # Prepare result
            test_result = {
                "success": success,
                "score": score,
                "vulnerabilities_found": vulnerabilities_found,
                "total_attempts": total_attempts,
                "probes_used": probes,
                "generations": generations,
                "sut_type": sut_type,
                "garak_stdout": result.stdout[:1000],  # Truncate to avoid huge outputs
                "garak_stderr": result.stderr[:1000] if result.stderr else "",
            }

            if not success:
                test_result["error"] = f"Garak execution failed: {result.stderr}"

            # Output results as JSON
            print(json.dumps(test_result, indent=2))

            # Exit with appropriate code
            sys.exit(0 if success else 1)

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except subprocess.TimeoutExpired:
        error_result = {
            "success": False,
            "error": "Garak execution timed out",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
