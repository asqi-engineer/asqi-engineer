import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import ray  # type: ignore
from llmperf import common_metrics  # type: ignore
from llmperf.models import RequestConfig  # type: ignore
from llmperf.utils import (  # type: ignore
    flatten_dict,
    randomly_sample_sonnet_lines_prompt,
)
from openai_client import OpenAISDKClient

sys.path.insert(0, "/app/llmperf")


def _derive_params(
    systems_params: Dict[str, Any], test_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Derive llmperf benchmark parameters with sensible defaults and overrides.

    Expected structure for systems_params:
      { "system_under_test": { "type": "llm_api", "base_url": str, "model": str, "api_key": str } }

    test_params may optionally override benchmark knobs using the same names as llmperf CLI:
      mean_input_tokens, stddev_input_tokens, mean_output_tokens, stddev_output_tokens,
      max_num_completed_requests, timeout, num_concurrent_requests.
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

    # Extract metadata for tracking (injected by ASQI workflow)
    metadata = test_params.get("metadata", {})

    # Defaults per user request
    params = {
        "llm_api": "openai",
        "model": model,
        "sut_name": sut.get("name"),
        "mean_input_tokens": int(test_params.get("mean_input_tokens", 550)),
        "stddev_input_tokens": int(test_params.get("stddev_input_tokens", 150)),
        "mean_output_tokens": int(test_params.get("mean_output_tokens", 150)),
        "stddev_output_tokens": int(test_params.get("stddev_output_tokens", 10)),
        "max_num_completed_requests": int(
            test_params.get("max_num_completed_requests", 1)
        ),
        "timeout": int(test_params.get("timeout", 600)),
        "num_concurrent_requests": int(test_params.get("num_concurrent_requests", 1)),
    }

    return {
        "base_url": base_url,
        "api_key": api_key,
        "metadata": metadata,
        **params,
    }


def _ensure_results_dir(subfolder: str = "") -> str:
    # Default to mounted output path
    out_root = os.environ.get("OUTPUT_MOUNT_PATH", tempfile.gettempdir())
    out = Path(out_root) / subfolder if subfolder else Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _run_benchmark_and_collect(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLMPerf token benchmark using custom OpenAI SDK client with metadata support."""
    # Generate unique folder name: llmperf_<8digit_hash>
    hash_input = json.dumps(
        {k: v for k, v in params.items() if k not in ["base_url", "api_key"]},
        sort_keys=True,
    )
    hash_str = hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[:8]
    folder_name = f"llmperf_{hash_str}"
    results_dir = _ensure_results_dir(folder_name)

    try:
        env_vars = {
            "OPENAI_API_BASE": params["base_url"],
            "OPENAI_API_KEY": params["api_key"],
        }
        if not ray.is_initialized():
            ray.init(runtime_env={"env_vars": env_vars})

        metadata = params.get("metadata", {})
        model = params["model"]
        max_requests = params["max_num_completed_requests"]
        num_concurrent = params["num_concurrent_requests"]
        mean_input = params["mean_input_tokens"]
        stddev_input = params["stddev_input_tokens"]
        mean_output = params["mean_output_tokens"]
        stddev_output = params["stddev_output_tokens"]
        timeout = params["timeout"]

        # Generate prompts with varying lengths
        prompts = []
        num_output_tokens_list = []
        for _ in range(max_requests):
            num_input = max(1, int(random.gauss(mean_input, stddev_input)))
            num_output = max(1, int(random.gauss(mean_output, stddev_output)))
            prompt = randomly_sample_sonnet_lines_prompt(num_input)
            prompts.append(prompt)
            num_output_tokens_list.append(num_output)

        # Create client actors with metadata
        clients = [
            OpenAISDKClient.remote(metadata=metadata) for _ in range(num_concurrent)
        ]

        # Submit requests
        start_time = time.monotonic()
        pending_refs: List[Any] = []
        results: List[Dict[str, Any]] = []
        request_idx = 0

        while request_idx < max_requests or pending_refs:
            # Check timeout
            if time.monotonic() - start_time > timeout:
                print(f"Timeout reached after {timeout}s")
                break

            # Submit new requests up to concurrency limit
            while len(pending_refs) < num_concurrent and request_idx < max_requests:
                client = clients[request_idx % num_concurrent]
                request_config = RequestConfig(
                    model=model,
                    prompt=prompts[request_idx],
                    sampling_params={"max_tokens": num_output_tokens_list[request_idx]},
                    llm_api="openai",
                    metadata={},
                )
                ref = client.llm_request.remote(request_config)
                pending_refs.append(ref)
                request_idx += 1

            # Wait for any request to complete
            if pending_refs:
                done_refs, pending_refs = ray.wait(
                    pending_refs, num_returns=1, timeout=1.0
                )
                for ref in done_refs:
                    try:
                        metrics, generated_text, req_config = ray.get(ref)
                        results.append(
                            {
                                "metrics": metrics,
                                "generated_text": generated_text,
                                "request_config": req_config,
                            }
                        )
                    except Exception as e:
                        print(f"Request failed: {e}")

        end_time = time.monotonic()
        total_time = end_time - start_time

        # Calculate summary statistics using llmperf format
        if len(results) == 0:
            raise RuntimeError("No requests completed successfully")

        # Build metrics list for pandas DataFrame (same as llmperf)
        metrics_list = [r["metrics"] for r in results]
        df = pd.DataFrame(metrics_list)
        df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

        # Calculate statistics for each metric (matching llmperf's metrics_summary)
        ret = {}
        for key in [
            common_metrics.INTER_TOKEN_LAT,
            common_metrics.TTFT,
            common_metrics.E2E_LAT,
            common_metrics.REQ_OUTPUT_THROUGHPUT,
            common_metrics.NUM_INPUT_TOKENS,
            common_metrics.NUM_OUTPUT_TOKENS,
        ]:
            ret[key] = {}
            series = pd.Series(df_without_errored_req[key].tolist()).dropna()
            if len(series) > 0:
                quantiles = series.quantile(
                    [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
                ).to_dict()
                quantiles_reformatted = {
                    f"p{int(q * 100)}": v for q, v in quantiles.items()
                }
                ret[key]["quantiles"] = quantiles_reformatted
                ret[key]["mean"] = series.mean()
                ret[key]["min"] = series.min()
                ret[key]["max"] = series.max()
                ret[key]["stddev"] = series.std()
            else:
                ret[key]["quantiles"] = {}
                ret[key]["mean"] = 0
                ret[key]["min"] = 0
                ret[key]["max"] = 0
                ret[key]["stddev"] = 0

        # Additional aggregate metrics (matching llmperf)
        ret[common_metrics.NUM_REQ_STARTED] = len(metrics_list)

        error_codes = df[common_metrics.ERROR_CODE].dropna()
        num_errors = len(error_codes)
        ret[common_metrics.ERROR_RATE] = (
            num_errors / len(metrics_list) if len(metrics_list) else 0
        )
        ret[common_metrics.NUM_ERRORS] = num_errors
        ret[common_metrics.ERROR_CODE_FREQ] = (
            str(dict(error_codes.value_counts())) if num_errors else "{}"
        )

        overall_output_throughput = (
            df_without_errored_req[common_metrics.NUM_OUTPUT_TOKENS].sum() / total_time
            if total_time > 0
            else 0
        )
        ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

        num_completed_requests = len(df_without_errored_req)
        ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
        ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = (
            num_completed_requests / total_time * 60 if total_time > 0 else 0
        )

        # Build metadata structure (matching llmperf's LLMPerfResults)
        summary = {
            "model": model,
            "mean_input_tokens": mean_input,
            "stddev_input_tokens": stddev_input,
            "mean_output_tokens": mean_output,
            "stddev_output_tokens": stddev_output,
            "num_concurrent_requests": num_concurrent,
            "results": ret,
        }

        # Flatten to produce keys like results_ttft_s_mean
        flattened_summary = flatten_dict(summary)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            return obj

        flattened_summary = {k: convert_numpy(v) for k, v in flattened_summary.items()}

        # Save summary (flattened format matching llmperf output)
        summary_path = os.path.join(
            results_dir, f"{model.replace('/', '_')}_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(flattened_summary, f, indent=2)

        # Save individual results
        individual_path = os.path.join(
            results_dir, f"{model.replace('/', '_')}_individual.json"
        )
        with open(individual_path, "w") as f:
            json.dump(
                [
                    {
                        "ttft": r["metrics"].get(common_metrics.TTFT, 0),
                        "e2e_latency": r["metrics"].get(common_metrics.E2E_LAT, 0),
                        "output_throughput": r["metrics"].get(
                            common_metrics.REQ_OUTPUT_THROUGHPUT, 0
                        ),
                        "output_tokens": r["metrics"].get(
                            common_metrics.NUM_OUTPUT_TOKENS, 0
                        ),
                        "input_tokens": r["metrics"].get(
                            common_metrics.NUM_INPUT_TOKENS, 0
                        ),
                        "error_code": r["metrics"].get(common_metrics.ERROR_CODE),
                        "error_msg": r["metrics"].get(common_metrics.ERROR_MSG, ""),
                    }
                    for r in results
                ],
                f,
                indent=2,
            )

    except Exception as e:
        raise RuntimeError(f"Benchmark failed: {e}")

    return {
        "success": True,
        "results_dir": results_dir,
        "metrics": flattened_summary,
    }


def run_llmperf(systems_params: Dict[str, Any], test_params: Dict[str, Any]):
    # Derive settings
    cfg = _derive_params(systems_params, test_params)

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
