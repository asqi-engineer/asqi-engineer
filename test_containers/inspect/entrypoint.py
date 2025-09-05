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
            "agie_aqua_rat": ("inspect_evals.agieval", "agie_aqua_rat"), # done
            "agie_logiqa_en": ("inspect_evals.agieval", "agie_logiqa_en"), # done
            "agie_lsat_ar": ("inspect_evals.agieval", "agie_lsat_ar"), # done
            "agie_lsat_lr": ("inspect_evals.agieval", "agie_lsat_lr"), # done
            "agie_lsat_rc": ("inspect_evals.agieval", "agie_lsat_rc"), # done
            "agie_math": ("inspect_evals.agieval", "agie_math"), # done
            "agie_sat_en": ("inspect_evals.agieval", "agie_sat_en"), # done
            "agie_sat_en_without_passage": ("inspect_evals.agieval", "agie_sat_en_without_passage"), # done
            "agie_sat_math": ("inspect_evals.agieval", "agie_sat_math"), #done
            "aime2024": ("inspect_evals.aime2024", "aime2024"), # done
            "agent_bench_os": ("inspect_evals.agent_bench", "agent_bench_os"), # TODO DinD
            "agentdojo": ("inspect_evals.agentdojo", "agentdojo"), # done
            "agentharm": ("inspect_evals.agentharm", "agentharm"), # done
            "agentharm_benign": ("inspect_evals.agentharm", "agentharm_benign"), # done
            "agentic_misalignment": ("inspect_evals.agentic_misalignment", "agentic_misalignment"), # TODO
            "air_bench": ("inspect_evals.air_bench", "air_bench"), # done
            "apps": ("inspect_evals.apps", "apps"), # TODO
            "arc": ("inspect_evals.arc", "arc_challenge"),  # done
            "arc_challenge": ("inspect_evals.arc", "arc_challenge"), # done
            "arc_easy": ("inspect_evals.arc", "arc_easy"), # done
            "assistant_bench_closed_book_one_shot": ("inspect_evals.assistant_bench", "assistant_bench_closed_book_one_shot"), # done
            "assistant_bench_closed_book_zero_shot": ("inspect_evals.assistant_bench", "assistant_bench_closed_book_zero_shot"), # done
            "assistant_bench_web_browser": ("inspect_evals.assistant_bench", "assistant_bench_web_browser"), # TODO
            "assistant_bench_web_search_one_shot": ("inspect_evals.assistant_bench", "assistant_bench_web_search_one_shot"), # TODO
            "assistant_bench_web_search_zero_shot": ("inspect_evals.assistant_bench", "assistant_bench_web_search_zero_shot"), # TODO
            "bbeh": ("inspect_evals.bbeh", "bbeh"), # done
            "bbeh_mini": ("inspect_evals.bbeh", "bbeh_mini"), # done
            "bbh": ("inspect_evals.bbh", "bbh"), # done
            "bbq": ("inspect_evals.bbq", "bbq"),  # done
            "bigcodebench": ("inspect_evals.bigcodebench", "bigcodebench"), # TODO
            "bold": ("inspect_evals.bold", "bold"), # done
            "boolq": ("inspect_evals.boolq", "boolq"), # done
            "browse_comp": ("inspect_evals.browse_comp", "browse_comp"), # done
            "chembench": ("inspect_evals.chembench", "chembench"), # done
            "class_eval": ("inspect_evals.class_eval", "class_eval"), # TODO DinD
            "commonsense_qa": ("inspect_evals.commonsense_qa", "commonsense_qa"),  # done
            "core_bench": ("inspect_evals.core_bench", "core_bench"), # TODO
            "cybermetric_80": ("inspect_evals.cybermetric", "cybermetric_80"), # done
            "cybermetric_500": ("inspect_evals.cybermetric", "cybermetric_500"), # TODO
            "cybermetric_2000": ("inspect_evals.cybermetric", "cybermetric_2000"), #done
            "cybermetric_10000": ("inspect_evals.cybermetric", "cybermetric_10000"), # done
            "cybench": ("inspect_evals.cybench", "cybench"), # TODO simple
            "cyse2_interpreter_abuse": ("inspect_evals.cyberseceval_2", "cyse2_interpreter_abuse"), # done
            "cyse2_prompt_injection": ("inspect_evals.cyberseceval_2", "cyse2_prompt_injection"), # done
            "cyse2_vulnerability_exploit": ("inspect_evals.cyberseceval_2", "cyse2_vulnerability_exploit"), # done
            "visual_prompt_injection": ("inspect_evals.cyberseceval_3", "visual_prompt_injection"), # TODO
            "docvqa": ("inspect_evals.docvqa", "docvqa"), # TODO
            "drop": ("inspect_evals.drop", "drop"),  # done
            "ds1000": ("inspect_evals.ds1000", "ds1000"), # done
            "gaia": ("inspect_evals.gaia", "gaia"), # TODO: 401 error
            "gaia_level1": ("inspect_evals.gaia", "gaia_level1"), # TODO: 401 error
            "gaia_level2": ("inspect_evals.gaia", "gaia_level2"), # TODO: 401 error
            "gaia_level3": ("inspect_evals.gaia", "gaia_level3"), # TODO: 401 error
            "gdm_approved_directories": ("inspect_evals.gdm_capabilities", "gdm_approved_directories"),
            "gdm_calculator_improvement": ("inspect_evals.gdm_capabilities", "gdm_calculator_improvement"),
            "gdm_classifier_evasion": ("inspect_evals.gdm_capabilities", "gdm_classifier_evasion"),
            "gdm_context_length_mod_instrumental_only": ("inspect_evals.gdm_capabilities", "gdm_context_length_mod_instrumental_only"),
            "gdm_context_length_mod_irreversibility_only": ("inspect_evals.gdm_capabilities", "gdm_context_length_mod_irreversibility_only"),
            "gdm_cover_your_tracks": ("inspect_evals.gdm_capabilities", "gdm_cover_your_tracks"),
            "gdm_database_tool": ("inspect_evals.gdm_capabilities", "gdm_database_tool"),
            "gdm_in_house_ctf": ("inspect_evals.gdm_capabilities", "gdm_in_house_ctf"),
            "gdm_intercode_ctf": ("inspect_evals.gdm_capabilities", "gdm_intercode_ctf"),
            "gdm_latency_calculator": ("inspect_evals.gdm_capabilities", "gdm_latency_calculator"),
            "gdm_max_messages_calculator": ("inspect_evals.gdm_capabilities", "gdm_max_messages_calculator"),
            "gdm_max_tokens": ("inspect_evals.gdm_capabilities", "gdm_max_tokens"),
            "gdm_oversight_frequency": ("inspect_evals.gdm_capabilities", "gdm_oversight_frequency"),
            "gdm_oversight_pattern": ("inspect_evals.gdm_capabilities", "gdm_oversight_pattern"),
            "gdm_read_logs": ("inspect_evals.gdm_capabilities", "gdm_read_logs"),
            "gdm_strategic_rule_breaking": ("inspect_evals.gdm_capabilities", "gdm_strategic_rule_breaking"),
            "gdm_turn_off_filters": ("inspect_evals.gdm_capabilities", "gdm_turn_off_filters"),
            "gpqa_diamond": ("inspect_evals.gpqa", "gpqa_diamond"),  # done
            "gsm8k": ("inspect_evals.gsm8k", "gsm8k"),  # done
            "healthbench": ("inspect_evals.healthbench", "healthbench"), # done
            "hellaswag": ("inspect_evals.hellaswag", "hellaswag"),  # done
            "hle": ("inspect_evals.hle", "hle"), # TODO
            "humaneval": ("inspect_evals.humaneval", "humaneval"), # done
            "ifeval": ("inspect_evals.ifeval", "ifeval"), # done
            "infinite_bench_code_debug": ("inspect_evals.infinite_bench", "infinite_bench_code_debug"), # done
            "infinite_bench_code_run": ("inspect_evals.infinite_bench", "infinite_bench_code_run"), # done
            "infinite_bench_kv_retrieval": ("inspect_evals.infinite_bench", "infinite_bench_kv_retrieval"), # done
            "infinite_bench_longbook_choice_eng": ("inspect_evals.infinite_bench", "infinite_bench_longbook_choice_eng"), # done
            "infinite_bench_longdialogue_qa_eng": ("inspect_evals.infinite_bench", "infinite_bench_longdialogue_qa_eng"), # done
            "infinite_bench_math_calc": ("inspect_evals.infinite_bench", "infinite_bench_math_calc"), # done
            "infinite_bench_math_find": ("inspect_evals.infinite_bench", "infinite_bench_math_find"), # done
            "infinite_bench_number_string": ("inspect_evals.infinite_bench", "infinite_bench_number_string"), # done
            "infinite_bench_passkey": ("inspect_evals.infinite_bench", "infinite_bench_passkey"), # done
            "lab_bench_cloning_scenarios": ("inspect_evals.lab_bench", "lab_bench_cloning_scenarios"), # done
            "lab_bench_dbqa": ("inspect_evals.lab_bench", "lab_bench_dbqa"), # done 
            "lab_bench_figqa": ("inspect_evals.lab_bench", "lab_bench_figqa"), # done
            "lab_bench_litqa": ("inspect_evals.lab_bench", "lab_bench_litqa"), # done
            "lab_bench_protocolqa": ("inspect_evals.lab_bench", "lab_bench_protocolqa"), # done
            "lab_bench_seqqa": ("inspect_evals.lab_bench", "lab_bench_seqqa"), # done
            "lingoly": ("inspect_evals.lingoly", "lingoly"), # TODO
            "lingoly_too": ("inspect_evals.lingoly", "lingoly_too"), # TODO: maximum token length exceeded for nova lite
            "livebench": ("inspect_evals.livebench", "livebench"), # done
            "mask": ("inspect_evals.mask", "mask"), # TODO
            "math": ("inspect_evals.math", "math"),  # done
            "mathvista": ("inspect_evals.mathvista", "mathvista"), # done
            "mbpp": ("inspect_evals.mbpp", "mbpp"), # done
            "medqa": ("inspect_evals.medqa", "medqa"),  # done
            "mgsm": ("inspect_evals.mgsm", "mgsm"), # done
            "mle_bench": ("inspect_evals.mle_bench", "mle_bench"),
            "mle_bench_full": ("inspect_evals.mle_bench", "mle_bench_full"),
            "mle_bench_lite": ("inspect_evals.mle_bench", "mle_bench_lite"),
            "mmlu": ("inspect_evals.mmlu", "mmlu_0_shot"), # done
            "mmlu_0_shot": ("inspect_evals.mmlu", "mmlu_0_shot"), # done
            "mmlu_5_shot": ("inspect_evals.mmlu", "mmlu_5_shot"), # done
            "mmlu_pro": ("inspect_evals.mmlu_pro", "mmlu_pro"),  # done
            "mmiu": ("inspect_evals.mmiu", "mmiu"), # TODO
            "mmmu_multiple_choice": ("inspect_evals.mmmu", "mmmu_multiple_choice"), # done
            "mmmu_open": ("inspect_evals.mmmu", "mmmu_open"), # done
            "musr": ("inspect_evals.musr", "musr"), # done
            "niah": ("inspect_evals.niah", "niah"), # done
            "onet_m6": ("inspect_evals.onet", "onet_m6"),
            "osworld": ("inspect_evals.osworld", "osworld"),
            "paws": ("inspect_evals.paws", "paws"), # done
            "personality_BFI": ("inspect_evals.personality", "personality_BFI"),
            "personality_TRAIT": ("inspect_evals.personality", "personality_TRAIT"),
            "pre_flight": ("inspect_evals.pre_flight", "pre_flight"),
            "piqa": ("inspect_evals.piqa", "piqa"),  # done
            "pubmedqa": ("inspect_evals.pubmedqa", "pubmedqa"), # done
            "race_h": ("inspect_evals.race_h", "race_h"), # done
            "sciq": ("inspect_evals.sciknoweval", "sciknoweval"), # TODO
            "scicode": ("inspect_evals.scicode", "scicode"),
            "sec_qa_v1": ("inspect_evals.sec_qa", "sec_qa_v1"), # done
            "sec_qa_v1_5_shot": ("inspect_evals.sec_qa", "sec_qa_v1_5_shot"), # done
            "sec_qa_v2": ("inspect_evals.sec_qa", "sec_qa_v2"), # done
            "sec_qa_v2_5_shot": ("inspect_evals.sec_qa", "sec_qa_v2_5_shot"), # done
            "sevenllm_mcq_en": ("inspect_evals.sevenllm", "sevenllm_mcq_en"), # done
            "sevenllm_mcq_zh": ("inspect_evals.sevenllm", "sevenllm_mcq_zh"), # done
            "sevenllm_qa_en": ("inspect_evals.sevenllm", "sevenllm_qa_en"), # done
            "sevenllm_qa_zh": ("inspect_evals.sevenllm", "sevenllm_qa_zh"),
            "simpleqa": ("inspect_evals.simpleqa", "simpleqa"), # done
            "sosbench": ("inspect_evals.sosbench", "sosbench"),
            "squad": ("inspect_evals.squad", "squad"),
            "stereoset": ("inspect_evals.stereoset", "stereoset"), # done
            "strong_reject": ("inspect_evals.strong_reject", "strong_reject"),  # done
            "swe_bench": ("inspect_evals.swe_bench", "swe_bench"), # TODO extra deps
            "swe_bench_verified_mini": ("inspect_evals.swe_bench", "swe_bench_verified_mini"), # TODO extra deps
            "sycophancy": ("inspect_evals.sycophancy", "sycophancy"), # done
            "truthfulqa": ("inspect_evals.truthfulqa", "truthfulqa"),  # done
            "usaco": ("inspect_evals.usaco", "usaco"),
            "vstar_bench_attribute_recognition": ("inspect_evals.vstar_bench", "vstar_bench_attribute_recognition"),
            "vstar_bench_spatial_relationship_reasoning": ("inspect_evals.vstar_bench", "vstar_bench_spatial_relationship_reasoning"),
            "winogrande": ("inspect_evals.winogrande", "winogrande"), # done
            "wmdp_bio": ("inspect_evals.wmdp", "wmdp_bio"), # done
            "wmdp_chem": ("inspect_evals.wmdp", "wmdp_chem"), # done
            "wmdp_cyber": ("inspect_evals.wmdp", "wmdp_cyber"), # done
            "worldsense": ("inspect_evals.worldsense", "worldsense"), # done
            "writingbench": ("inspect_evals.writingbench", "writingbench"),
            "xstest": ("inspect_evals.xstest", "xstest"),  # done
            "zerobench": ("inspect_evals.zerobench", "zerobench"), # done
            "zerobench_subquestions": ("inspect_evals.zerobench", "zerobench_subquestions"), # done
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
