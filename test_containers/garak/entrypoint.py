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
        generations = test_params.get("generations", 1)
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
            process.wait()
            result = subprocess.CompletedProcess(
                args=garak_cmd,
                returncode=process.returncode,
                stdout="".join(stdout_content),
                stderr="".join(stderr_content),
            )

            # Parse garak results
            success = result.returncode == 0

            score = 0.0
            vulnerabilities_found = 0
            total_attempts = 0
            probe_results = {}

            if success:
                # Find the correct report file using the report_prefix
                report_files = list(
                    output_dir.glob(f"{Path(report_prefix).name}*.report.jsonl")
                )

                if report_files:
                    try:
                        report_file = report_files[0]
                        eval_entries = []
                        _run_info = None

                        with open(report_file, "r") as f:
                            for line in f:
                                try:
                                    entry = json.loads(line.strip())
                                    entry_type = entry.get("entry_type")

                                    if entry_type == "eval":
                                        eval_entries.append(entry)
                                    elif entry_type == "start_run setup":
                                        _run_info = entry

                                except json.JSONDecodeError:
                                    continue

                        # Calculate results from eval entries
                        if eval_entries:
                            total_passed = 0
                            total_attempts = 0

                            for eval_entry in eval_entries:
                                probe_name = eval_entry.get("probe", "unknown")
                                detector_name = eval_entry.get("detector", "unknown")
                                passed = eval_entry.get("passed", 0)
                                total = eval_entry.get("total", 0)

                                # Store individual probe results
                                if probe_name not in probe_results:
                                    probe_results[probe_name] = {}

                                probe_results[probe_name][detector_name] = {
                                    "passed": passed,
                                    "total": total,
                                    "score": passed / total if total > 0 else 0.0,
                                }

                                total_passed += passed
                                total_attempts += total

                            # Calculate overall score
                            if total_attempts > 0:
                                score = total_passed / total_attempts
                                vulnerabilities_found = total_attempts - total_passed

                            print(
                                f"DEBUG: Found {len(eval_entries)} eval entries",
                                file=sys.stderr,
                            )
                            print(
                                f"DEBUG: Total attempts: {total_attempts}, Total passed: {total_passed}",
                                file=sys.stderr,
                            )

                        else:
                            # No eval entries found - this indicates an error
                            success = False
                            score = 0.0
                            print(
                                "DEBUG: No eval entries found in report",
                                file=sys.stderr,
                            )

                    except (FileNotFoundError, KeyError, IndexError) as e:
                        # Can't parse results, treat as failure
                        success = False
                        score = 0.0
                        print(f"DEBUG: Error parsing report file: {e}", file=sys.stderr)

                else:
                    # No report files generated - this indicates failure
                    success = False
                    score = 0.0
                    print("DEBUG: No report files found", file=sys.stderr)

            else:
                # Garak execution failed
                success = False
                score = 0.0
                print(
                    f"DEBUG: Garak execution failed with return code: {result.returncode}",
                    file=sys.stderr,
                )

            # Prepare result
            test_result = {
                "success": success,
                "score": score,
                "vulnerabilities_found": vulnerabilities_found,
                "total_attempts": total_attempts,
                "probes_used": probes,
                "generations": generations,
                "sut_type": sut_type,
                "probe_results": probe_results,
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
