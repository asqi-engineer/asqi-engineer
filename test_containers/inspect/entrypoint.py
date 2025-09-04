import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Try to import inspect log utilities (may not be available in all environments)
try:
    from inspect_ai.log import read_eval_log
    INSPECT_LOG_AVAILABLE = True
except ImportError:
    INSPECT_LOG_AVAILABLE = False

def main():
    """Main entrypoint for Inspect evaluation."""
    parser = argparse.ArgumentParser(description="Inspect evaluation container")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        # Extract system_under_test
        sut_params = systems_params.get("system_under_test", {})
        if not sut_params:
            raise ValueError("Missing system_under_test in systems_params")

        # Validate SUT type
        sut_type = sut_params.get("type")
        if sut_type not in ["llm_api"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        # Extract SUT parameters
        base_url = sut_params.get("base_url")
        api_key = sut_params.get("api_key")
        model = sut_params.get("model")

        if not base_url or not api_key or not model:
            raise ValueError("Missing required parameters: base_url, api_key, or model")

        # Extract test parameters
        evaluation = test_params.get("evaluation", "xstest")
        evaluation_params = test_params.get("evaluation_params", {})
        limit = test_params.get("limit", 10)

        # Set environment variables for inspect
        # Inspect requires model names in the format "provider/model" or "openai-api/provider/model"
        parsed = urlparse(base_url)
        netloc = parsed.netloc
        parts = netloc.split('.')
        provider = parts[-2] if len(parts) >= 2 else 'unknown'
        
        if provider == 'openai':
            full_model = f"openai/{model}"
        else:
            full_model = f"openai-api/{provider}/{model}"
            provider_upper = provider.upper().replace('-', '_')
        
        # Always set OPENAI_API_KEY and OPENAI_BASE_URL for compatibility
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        
        # Also set provider-specific environment variables
        provider_upper = provider.upper().replace('-', '_')
        os.environ[f"{provider_upper}_API_KEY"] = api_key
        os.environ[f"{provider_upper}_BASE_URL"] = base_url
        
        os.environ["INSPECT_EVAL_MODEL"] = full_model

        # Import inspect modules
        from inspect_ai import eval as inspect_eval

        # Define available evaluations with their import paths and functions
        EVALUATION_REGISTRY = {
            # Basic evaluations
            "xstest": ("inspect_evals.xstest", "xstest"),  # done
            "truthfulqa": ("inspect_evals.truthfulqa", "truthfulqa"),  # done
            "boolq": ("inspect_evals.boolq", "boolq"), # done       
            
            # MMLU variants
            "mmlu": ("inspect_evals.mmlu", "mmlu_0_shot"), # done
            "mmlu_0_shot": ("inspect_evals.mmlu", "mmlu_0_shot"), # done
            "mmlu_5_shot": ("inspect_evals.mmlu", "mmlu_5_shot"), # done
            
            # Code evaluations
            "humaneval": ("inspect_evals.humaneval", "humaneval"), # done
            "gsm8k": ("inspect_evals.gsm8k", "gsm8k"),  # done
            "mbpp": ("inspect_evals.mbpp", "mbpp"), # done
            "ifeval": ("inspect_evals.ifeval", "ifeval"), # done
            
            # Commonsense reasoning
            "hellaswag": ("inspect_evals.hellaswag", "hellaswag"),  # done
            "winogrande": ("inspect_evals.winogrande", "winogrande"), # done
            "piqa": ("inspect_evals.piqa", "piqa"),  # done
            "commonsense_qa": ("inspect_evals.commonsense_qa", "commonsense_qa"),  # done
            
            # Science and knowledge
            "arc": ("inspect_evals.arc", "arc_challenge"),  # done
            "arc_easy": ("inspect_evals.arc", "arc_easy"), # done
            "arc_challenge": ("inspect_evals.arc", "arc_challenge"), # done
            "sciq": ("inspect_evals.sciknoweval", "sciknoweval"), # TODO
            "medqa": ("inspect_evals.medqa", "medqa"),  # done
            "pubmedqa": ("inspect_evals.pubmedqa", "pubmedqa"), # done
            
            # Mathematics
            "math": ("inspect_evals.math", "math"),  # done
            "mathvista": ("inspect_evals.mathvista", "mathvista"), # done
            "aime2024": ("inspect_evals.aime2024", "aime2024"), # done
            "mgsm": ("inspect_evals.mgsm", "mgsm"), # done
            
            # Reading comprehension
            "drop": ("inspect_evals.drop", "drop"),  # done
            "race_h": ("inspect_evals.race_h", "race_h"), # done
            
            # Safety and alignment
            "bbq": ("inspect_evals.bbq", "bbq"),  # done
            "bold": ("inspect_evals.bold", "bold"), # TODO missing torch
            "strong_reject": ("inspect_evals.strong_reject", "strong_reject"),  # done
            "sycophancy": ("inspect_evals.sycophancy", "sycophancy"), # done
            
            # Agent and tool use
            "agent_bench_os": ("inspect_evals.agent_bench", "agent_bench_os"), # TODO DinD
            
            # Multilingual
            "mmlu_pro": ("inspect_evals.mmlu_pro", "mmlu_pro"),  # done
            "paws": ("inspect_evals.paws", "paws"), # done
            "lingoly": ("inspect_evals.lingoly", "lingoly"), # TODO
            
            # Programming and software engineering
            "ds1000": ("inspect_evals.ds1000", "ds1000"), # TODO DinD
            "swe_bench": ("inspect_evals.swe_bench", "swe_bench"), # TODO extra deps
            "class_eval": ("inspect_evals.class_eval", "class_eval"), # TODO DinD
            
            # Multi-modal
            "mmiu": ("inspect_evals.mmiu", "mmiu"), # TODO
            "mmmu_multiple_choice": ("inspect_evals.mmmu", "mmmu_multiple_choice"), # TODO
            "mmmu_open": ("inspect_evals.mmmu", "mmmu_open"), # TODO
            
            # Long context and reasoning
            "infinite_bench_code_debug": ("inspect_evals.infinite_bench", "infinite_bench_code_debug"), # TODO 
            "infinite_bench_math_calc": ("inspect_evals.infinite_bench", "infinite_bench_math_calc"), # done
            "infinite_bench_passkey": ("inspect_evals.infinite_bench", "infinite_bench_passkey"), # done
            
            # Specialized domains
            "chembench": ("inspect_evals.chembench", "chembench"), # done
            "healthbench": ("inspect_evals.healthbench", "healthbench"), # done
            "cybench": ("inspect_evals.cybench", "cybench"), # TODO simple
            "gpqa_diamond": ("inspect_evals.gpqa", "gpqa_diamond"),  # done
        }

        # Get evaluation function
        if evaluation not in EVALUATION_REGISTRY:
            available_evals = ", ".join(sorted(EVALUATION_REGISTRY.keys()))
            raise ValueError(f"Unsupported evaluation: {evaluation}. Supported evaluations: {available_evals}")

        module_name, func_name = EVALUATION_REGISTRY[evaluation]
        
        # Dynamically import the evaluation function
        try:
            module = __import__(module_name, fromlist=[func_name])
            task_func = getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import evaluation '{evaluation}' from {module_name}.{func_name}: {e}")

        # Create a temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the task with evaluation parameters
            task = task_func(**evaluation_params)

            # Run evaluation with limit
            print(f"DEBUG: Starting evaluation with limit={limit}", file=sys.stderr)
            log = inspect_eval(
                task,
                limit=limit,
                log_dir=temp_dir
            )[0]  # eval returns a list, get first result
            print(f"DEBUG: Evaluation completed, log.location = {log.location}", file=sys.stderr)

            metrics = {}
            try:
                from inspect_ai.log import read_eval_log
                eval_log = read_eval_log(log.location)
                
                # Get metrics from the log's results
                if (hasattr(eval_log, 'results') and eval_log.results and 
                    hasattr(eval_log.results, 'scores') and eval_log.results.scores):
                    eval_score = eval_log.results.scores[0]
                    if hasattr(eval_score, 'metrics') and eval_score.metrics:
                        for metric_name, metric_obj in eval_score.metrics.items():
                            if hasattr(metric_obj, 'value'):
                                metrics[metric_name] = metric_obj.value
            except (ImportError, Exception) as e:
                # Fallback to the old method if reading the log file fails
                print(f"DEBUG: Failed to read eval log: {e}", file=sys.stderr)
                pass

            # Extract results from the log
            results = log.results
            if results is None:
                # Try to read from the eval log file even if results is None
                try:
                    from inspect_ai.log import read_eval_log
                    eval_log = read_eval_log(log.location)
                    if hasattr(eval_log, 'results') and eval_log.results:
                        results = eval_log.results
                except:
                    pass
                
                if results is None:
                    raise ValueError("Evaluation failed - no results available")

            # Handle different result structures
            if hasattr(results, 'scores') and results.scores is not None:
                scores = results.scores
            elif isinstance(results, list) and len(results) > 0:
                # If results is a list, get the first result
                scores = results[0].scores if hasattr(results[0], 'scores') else None
            else:
                scores = None
                print("DEBUG: No scores found", file=sys.stderr)

            # Count response types from individual sample scores if available
            total_samples = 0

            if scores is not None and hasattr(scores, 'samples') and scores.samples:
                for sample_score in scores.samples:
                    if hasattr(sample_score, 'score'):
                        total_samples += 1

            # If we don't have sample-level scores, try to get total_samples from other sources
            if total_samples == 0:
                # Try to get sample count from the log or results
                if hasattr(log, 'samples') and log.samples:
                    total_samples = len(log.samples)
                elif not isinstance(results, list) and hasattr(results, 'samples') and results.samples:
                    total_samples = len(results.samples)
                else:
                    # Default to the limit if we can't determine the actual count
                    total_samples = limit

            # Prepare result
            result = {
                "success": True,
                "evaluation": evaluation,
                "evaluation_params": evaluation_params,
                "total_samples": total_samples,
                "model": model,
                "metrics": metrics
            }

            # Output results as JSON
            print(json.dumps(result, indent=2))
            sys.exit(0)

    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON decode error: {e}", file=sys.stderr)
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "evaluation": "unknown",
            "total_samples": 0,
            "metrics": {}
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}", file=sys.stderr)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr)
        error_result = {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "evaluation": "unknown",
            "total_samples": 0,
            "metrics": {}
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
