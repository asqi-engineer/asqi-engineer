import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add the llmperf directory to sys.path to import token_benchmark_ray
sys.path.insert(0, "/app/llmperf")


def _export_openai_env(base_url: str, api_key: str) -> None:
    """Set environment variables expected by llmperf for OpenAI-compatible APIs."""
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key


def _derive_params(
    systems_params: Dict[str, Any], test_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Derive llmperf benchmark parameters with sensible defaults and overrides.

    Expected structure for systems_params:
      { "system_under_test": { "type": "llm_api", "base_url": str, "model": str, "api_key": str } }

    test_params may optionally override benchmark knobs using the same names as llmperf CLI:
      mean_input_tokens, stddev_input_tokens, mean_output_tokens, stddev_output_tokens,
      max_num_completed_requests, timeout, num_concurrent_requests, additional_sampling_params.
    """
    sut = systems_params.get("system_under_test", {})
    if not sut:
        raise ValueError("Missing system_under_test in systems_params")

    if sut.get("type") not in ["llm_api"]:
        raise ValueError(f"Unsupported system_under_test type: {sut.get('type')}")

    base_url = sut.get("base_url")
    api_key = (
        sut.get("api_key")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    model = sut.get("model")

    if not base_url or not model:
        raise ValueError("system_under_test requires 'base_url' and 'model'")
    if not api_key:
        raise ValueError("No API key found for system_under_test")

    # Defaults per user request
    params = {
        "llm_api": "openai",
        "model": model,
        "mean_input_tokens": int(test_params.get("mean_input_tokens", 550)),
        "stddev_input_tokens": int(test_params.get("stddev_input_tokens", 150)),
        "mean_output_tokens": int(test_params.get("mean_output_tokens", 150)),
        "stddev_output_tokens": int(test_params.get("stddev_output_tokens", 10)),
        "max_num_completed_requests": int(
            test_params.get("max_num_completed_requests", 2)
        ),
        "timeout": int(test_params.get("timeout", 600)),
        "num_concurrent_requests": int(test_params.get("num_concurrent_requests", 1)),
        "results_dir": str(test_params.get("results_dir", "")),
        "additional_sampling_params": test_params.get(
            "additional_sampling_params", "{}"
        ),
    }

    # Validate and normalize additional_sampling_params
    if isinstance(params["additional_sampling_params"], dict):
        params["additional_sampling_params"] = json.dumps(
            params["additional_sampling_params"]
        )  # to str
    elif isinstance(params["additional_sampling_params"], str):
        try:
            json.loads(params["additional_sampling_params"])  # must be valid JSON
        except Exception as e:
            raise ValueError(f"additional_sampling_params must be valid JSON: {e}")
    else:
        raise ValueError("additional_sampling_params must be a JSON string or dict")

    return {
        "base_url": base_url,
        "api_key": api_key,
        **params,
    }


def _ensure_results_dir(results_dir: str) -> str:
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        return results_dir
    # Default to mounted output path
    out_root = os.environ.get("OUTPUT_MOUNT_PATH", "/tmp")
    out = Path(out_root) / "llmperf"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _run_benchmark_and_collect(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLMPerf token benchmark in-process, then load summary JSON."""
    results_dir = _ensure_results_dir(params["results_dir"])

    try:
        # Import the function from the installed llmperf package
        from token_benchmark_ray import run_token_benchmark  # type: ignore
        import ray  # type: ignore

        # Propagate environment into Ray workers (OPENAI_* are already set)
        env_vars = dict(os.environ)
        try:
            ray.init(runtime_env={"env_vars": env_vars})  # type: ignore
        except Exception:
            # Ray may already be initialized; ignore
            pass

        run_token_benchmark(
            llm_api=params["llm_api"],
            model=params["model"],
            test_timeout_s=params["timeout"],
            max_num_completed_requests=params["max_num_completed_requests"],
            num_concurrent_requests=params["num_concurrent_requests"],
            mean_input_tokens=params["mean_input_tokens"],
            stddev_input_tokens=params["stddev_input_tokens"],
            mean_output_tokens=params["mean_output_tokens"],
            stddev_output_tokens=params["stddev_output_tokens"],
            additional_sampling_params=params["additional_sampling_params"],
            results_dir=results_dir,
            user_metadata={},
        )
    except Exception as e:
        # Fail fast if llmperf is not importable or runtime is missing
        raise RuntimeError(f"llmperf runtime not available: {e}")

    # Find the latest *_summary.json file emitted
    summary_files = sorted(
        glob.glob(os.path.join(results_dir, "*_summary.json")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not summary_files:
        raise FileNotFoundError(f"No summary file produced in {results_dir}")

    with open(summary_files[0], "r") as f:
        summary = json.load(f)

    return {
        "success": True,
        "results_dir": results_dir,
        "summary_file": summary_files[0],
        "summary": summary,
    }


def run_llmperf(systems_params: Dict[str, Any], test_params: Dict[str, Any]):
    # Derive settings and export OpenAI-compatible env vars
    cfg = _derive_params(systems_params, test_params)
    _export_openai_env(cfg["base_url"], cfg["api_key"])

    # Run benchmark and collect summary
    return _run_benchmark_and_collect(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="LLMPerf Token Benchmark Test Container"
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
        result = run_llmperf(systems_params, test_params)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(
            json.dumps({"success": False, "error": f"Unexpected error: {e}"}, indent=2)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
