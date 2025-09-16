import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Import inspect modules
from inspect_ai import eval
from inspect_ai.log import read_eval_log

EVALUATION_REGISTRY = {
    ## Coding
    # ❌ APPS
    "apps": ("inspect_evals.apps", False), # TODO
    # ✅ AgentBench
    "agent_bench_os": ("inspect_evals.agent_bench", True), 
    # ❌ BigCodeBench
    "bigcodebench": ("inspect_evals.bigcodebench", False), # TODO: very large
    # ❌ CORE-Bench
    "core_bench": ("inspect_evals.core_bench", False), # TODO
    # ✅ ClassEval
    "class_eval": ("inspect_evals.class_eval", True), 
    # ✅ DS-1000
    "ds1000": ("inspect_evals.ds1000", True), 
    # ✅ HumanEval
    "humaneval": ("inspect_evals.humaneval", True), 
    # ✅ MBPP
    "mbpp": ("inspect_evals.mbpp", True), 
    # ❌ MLE-bench
    "mle_bench": ("inspect_evals.mle_bench", False), # TODO: Kaggle Auth required
    "mle_bench_full": ("inspect_evals.mle_bench", False), # TODO: Kaggle Auth required
    "mle_bench_lite": ("inspect_evals.mle_bench", False), # TODO: Kaggle Auth required
    # ❌ SWE-bench
    "swe_bench": ("inspect_evals.swe_bench", False), # TODO extra deps
    "swe_bench_verified_mini": ("inspect_evals.swe_bench", False), # TODO extra deps
    # ❌ SciCode
    "scicode": ("inspect_evals.scicode", False), # TODO
    # ✅ USACO
    "usaco": ("inspect_evals.usaco", True), # TODO

    ## Assistants
    # ❌ AssistantBench
    "assistant_bench_closed_book_one_shot": ("inspect_evals.assistant_bench", True), 
    "assistant_bench_closed_book_zero_shot": ("inspect_evals.assistant_bench", True), 
    "assistant_bench_web_browser": ("inspect_evals.assistant_bench", False), # TODO: Requires GOOGLE_CSE_API_KEY
    "assistant_bench_web_search_one_shot": ("inspect_evals.assistant_bench", False), # TODO: Requires GOOGLE_CSE_API_KEY
    "assistant_bench_web_search_zero_shot": ("inspect_evals.assistant_bench", False), # TODO: Requires GOOGLE_CSE_API_KEY
    # ✅ BrowseComp
    "browse_comp": ("inspect_evals.browse_comp", True), 
    # ❌ GAIA
    "gaia": ("inspect_evals.gaia", True), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level1": ("inspect_evals.gaia", True), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level2": ("inspect_evals.gaia", True), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level3": ("inspect_evals.gaia", True), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    # ❌ Mind2Web
    "mind2web": ("inspect_evals.mind2web", False), # TODO
    # ❌ OSWorld
    "osworld": ("inspect_evals.osworld", False), # TODO
    # ✅ Sycophancy
    "sycophancy": ("inspect_evals.sycophancy", True), 

    ## Cybersecurity
    # ❌ CYBERSECEVAL 3
    "visual_prompt_injection": ("inspect_evals.cyberseceval_3", False), # TODO
    # ❌ Catastrophic Cyber Capabilities Benchmark (3CB)
    "threecb": ("inspect_evals.threecb", False), # TODO
    # ❌ Cybench
    "cybench": ("inspect_evals.cybench", False), # TODO simple
    # ✅ CyberMetric
    "cybermetric_80": ("inspect_evals.cybermetric", True), 
    "cybermetric_500": ("inspect_evals.cybermetric", True), 
    "cybermetric_2000": ("inspect_evals.cybermetric", True),
    "cybermetric_10000": ("inspect_evals.cybermetric", True), 
    # ✅ CyberSecEval_2
    "cyse2_interpreter_abuse": ("inspect_evals.cyberseceval_2", True), 
    "cyse2_prompt_injection": ("inspect_evals.cyberseceval_2", True), 
    "cyse2_vulnerability_exploit": ("inspect_evals.cyberseceval_2", True), 
    # ✅ GDM Dangerous Capabilities: Capture the Flag
    "gdm_in_house_ctf": ("inspect_evals.gdm_capabilities", True), 
    # ✅ InterCode: Security and Coding Capture-the-Flag Challenges
    "gdm_intercode_ctf": ("inspect_evals.gdm_capabilities", True), # TODO
    # ❌ SEvenLLM
    "sevenllm_mcq_en": ("inspect_evals.sevenllm", True), 
    "sevenllm_mcq_zh": ("inspect_evals.sevenllm", True), 
    "sevenllm_qa_en": ("inspect_evals.sevenllm", True), 
    "sevenllm_qa_zh": ("inspect_evals.sevenllm", False), # TODO: keras 3 issue
    # ✅ SecQA
    "sec_qa_v1": ("inspect_evals.sec_qa", True), 
    "sec_qa_v1_5_shot": ("inspect_evals.sec_qa", True), 
    "sec_qa_v2": ("inspect_evals.sec_qa", True), 
    "sec_qa_v2_5_shot": ("inspect_evals.sec_qa", True), 

    ## Safeguards
    # ✅ AbstentionBench
    "abstention_bench": ("inspect_evals.abstention_bench", True), # Info: #1 Dataset 'Idavidrein/gpqa' on HuggingFace is a gated dataset. You must be authenticated to access it. #2 Use openai scorer model as the default requires openrouter
    # ✅ AgentDojo
    "agentdojo": ("inspect_evals.agentdojo", True), 
    # ✅ AgentHarm
    "agentharm": ("inspect_evals.agentharm", True), 
    "agentharm_benign": ("inspect_evals.agentharm", True), 
    # ✅ LAB-Bench
    "lab_bench_cloning_scenarios": ("inspect_evals.lab_bench", True), 
    "lab_bench_dbqa": ("inspect_evals.lab_bench", True),  
    "lab_bench_figqa": ("inspect_evals.lab_bench", True), 
    "lab_bench_litqa": ("inspect_evals.lab_bench", True), 
    "lab_bench_protocolqa": ("inspect_evals.lab_bench", True), 
    "lab_bench_seqqa": ("inspect_evals.lab_bench", True), 
    # ✅ MASK
    "mask": ("inspect_evals.mask", True), # Info: Dataset 'cais/MASK' on HuggingFace is a gated dataset. You must be authenticated to access it.
    # ✅ StereoSet
    "stereoset": ("inspect_evals.stereoset", True), 
    # ✅ StrongREJECT
    "strong_reject": ("inspect_evals.strong_reject", True),  
    # ✅ WMDP
    "wmdp_bio": ("inspect_evals.wmdp", True), 
    "wmdp_chem": ("inspect_evals.wmdp", True), 
    "wmdp_cyber": ("inspect_evals.wmdp", True), 

    ## ✅✅ Mathematics
    # ✅ AIME 2024
    "aime2024": ("inspect_evals.aime2024", True), 
    # ✅ GSM8K
    "gsm8k": ("inspect_evals.gsm8k", True),  
    # ✅ MATH
    "math": ("inspect_evals.math", True),  
    # ✅ MGSM
    "mgsm": ("inspect_evals.mgsm", True), 
    # ✅ MathVista
    "mathvista": ("inspect_evals.mathvista", True), 

    ## Reasoning
    # ✅ ARC
    "arc": ("inspect_evals.arc", True),  
    "arc_challenge": ("inspect_evals.arc", True), 
    "arc_easy": ("inspect_evals.arc", True), 
    # ✅ BBH
    "bbh": ("inspect_evals.bbh", True), 
    # ✅ BIG-Bench Extra Hard
    "bbeh": ("inspect_evals.bbeh", True), 
    "bbeh_mini": ("inspect_evals.bbeh", True), 
    # ✅ BoolQ
    "boolq": ("inspect_evals.boolq", True), 
    # ✅ DROP
    "drop": ("inspect_evals.drop", True),  
    # ✅ HellaSwag
    "hellaswag": ("inspect_evals.hellaswag", True),  
    # ✅ IFEval
    "ifeval": ("inspect_evals.ifeval", True), 
    # ❌ LingOly
    "lingoly": ("inspect_evals.lingoly", False), # TODO: Dataset 'ambean/lingOly' is a gated dataset on the Hub. You must be authenticated to access it.
    "lingoly_too": ("inspect_evals.lingoly", True), 
    # ✅ MMMU
    "mmmu_multiple_choice": ("inspect_evals.mmmu", True), 
    "mmmu_open": ("inspect_evals.mmmu", True), 
    # ✅ MuSR
    "musr": ("inspect_evals.musr", True), 
    # ✅ Needle in a Haystack (NIAH)
    "niah": ("inspect_evals.niah", True), 
    # ✅ PAWS
    "paws": ("inspect_evals.paws", True), 
    # ✅ PIQA
    "piqa": ("inspect_evals.piqa", True),  
    # ✅ RACE-H
    "race_h": ("inspect_evals.race_h", True), 
    # ❌ SQuAD
    "squad": ("inspect_evals.squad", False), # TODO: ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', ...
    # ✅ WINOGRANDE
    "winogrande": ("inspect_evals.winogrande", True), 
    # ✅ WorldSense
    "worldsense": ("inspect_evals.worldsense", True), 
    # ✅ ∞Bench
    "infinite_bench_code_debug": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_code_run": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_kv_retrieval": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_longbook_choice_eng": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_longdialogue_qa_eng": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_math_calc": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_math_find": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_number_string": ("inspect_evals.infinite_bench", True), 
    "infinite_bench_passkey": ("inspect_evals.infinite_bench", True), 

    ## Knowledge
    # ✅ AGIEval
    "agie_aqua_rat": ("inspect_evals.agieval", True), 
    "agie_logiqa_en": ("inspect_evals.agieval", True), 
    "agie_lsat_ar": ("inspect_evals.agieval", True), 
    "agie_lsat_lr": ("inspect_evals.agieval", True), 
    "agie_lsat_rc": ("inspect_evals.agieval", True), 
    "agie_math": ("inspect_evals.agieval", True), 
    "agie_sat_en": ("inspect_evals.agieval", True), 
    "agie_sat_en_without_passage": ("inspect_evals.agieval", True), 
    "agie_sat_math": ("inspect_evals.agieval", True), 
    # ✅ AIR Bench
    "air_bench": ("inspect_evals.air_bench", True), 
    # ✅ ChemBench
    "chembench": ("inspect_evals.chembench", True), 
    # ✅ CommonsenseQA
    "commonsense_qa": ("inspect_evals.commonsense_qa", True),  
    # ✅ GPQA
    "gpqa_diamond": ("inspect_evals.gpqa", True),  
    # ✅ HealthBench
    "healthbench": ("inspect_evals.healthbench", True), 
    # ✅ Humanity's Last Exam
    "hle": ("inspect_evals.hle", True), # Info: Gated Dataset (cais/hle on HuggingFace)
    # ✅ LiveBench
    "livebench": ("inspect_evals.livebench", True), 
    # ✅ MMLU-Pro
    "mmlu_pro": ("inspect_evals.mmlu_pro", True),  
    # ✅ MMLU
    "mmlu": ("inspect_evals.mmlu", True), 
    "mmlu_0_shot": ("inspect_evals.mmlu", True), 
    "mmlu_5_shot": ("inspect_evals.mmlu", True), 
    # ✅ MedQA
    "medqa": ("inspect_evals.medqa", True),  
    # ✅ O-NET
    "onet_m6": ("inspect_evals.onet", True), 
    # ✅ Pre-Flight
    "pre_flight": ("inspect_evals.pre_flight", True), 
    # ✅ PubMedQA
    "pubmedqa": ("inspect_evals.pubmedqa", True), 
    # ✅ SOS BENCH
    "sosbench": ("inspect_evals.sosbench", True), # Info: Fails with bedrock: "The model returned the\nfollowing errors: Malformed input request: #: extraneous key [thinking] is not  \npermitted, please reformat your input and try again.
    # ❌ SciKnowEval
    "sciq": ("inspect_evals.sciknoweval", False), # TODO: No such file or directory: './src/inspect_evals/sciknoweval/evaluator_prompt.yaml'
    # ✅ SimpleQA
    "simpleqa": ("inspect_evals.simpleqa", True), 
    # ✅ TruthfulQA
    "truthfulqa": ("inspect_evals.truthfulqa", True),  
    # ✅ XSTest
    "xstest": ("inspect_evals.xstest", True), # Info: gated dataset (walledai/XSTest on HuggingFace)

    ## Scheming
    # ✅ Agentic Misalignment
    "agentic_misalignment": ("inspect_evals.agentic_misalignment", True), 
    # ❌ GDM Dangerous Capabilities: Self-reasoning
    "gdm_approved_directories": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_calculator_improvement": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_context_length_mod_instrumental_only": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_context_length_mod_irreversibility_only": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_database_tool": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_latency_calculator": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_max_messages_calculator": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_max_tokens": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_oversight_frequency": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_read_logs": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_turn_off_filters": ("inspect_evals.gdm_capabilities", False), # TODO
    # ❌ GDM Dangerous Capabilities: Stealth
    "gdm_classifier_evasion": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_cover_your_tracks": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_oversight_pattern": ("inspect_evals.gdm_capabilities", False), # TODO
    "gdm_strategic_rule_breaking": ("inspect_evals.gdm_capabilities", False), # TODO

    ## Multimodal
    # ❌ DocVQA
    "docvqa": ("inspect_evals.docvqa", False), # TODO
    # ❌ MMIU
    "mmiu": ("inspect_evals.mmiu", False), # TODO
    # ❌ V*Bench
    "vstar_bench_attribute_recognition": ("inspect_evals.vstar_bench", False), # TODO
    "vstar_bench_spatial_relationship_reasoning": ("inspect_evals.vstar_bench", False), # TODO
    # ✅ ZeroBench
    "zerobench": ("inspect_evals.zerobench", True), 
    "zerobench_subquestions": ("inspect_evals.zerobench", True), 

    ## Bias
    # ✅ BBQ
    "bbq": ("inspect_evals.bbq", True),  
    # ❌ BOLD
    "bold": ("inspect_evals.bold", True), # TODO: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.

    ## ✅✅ Personality
    # ✅ Personality
    "personality_BFI": ("inspect_evals.personality", True), 
    "personality_TRAIT": ("inspect_evals.personality", True), # Info: Gated Dataset (mirlab/TRAIT on HuggingFace)

    ## ✅✅ Writing
    # ✅ WritingBench
    "writingbench": ("inspect_evals.writingbench", True), # Info: Set judge_model to "openai/{model}"
}

# Configure datasets to trust remote code before any other imports
try:
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
except ImportError:
    pass

def _configure_provider_environment(base_url: str, model: str, api_key: str) -> tuple[str, str]:
    """
    Configure environment variables for LLM provider based on base URL.
    
    Args:
        base_url: The base URL of the LLM API endpoint
        model: The model name
        api_key: The API key for authentication
        
    Returns:
        Tuple of (full_model_name, provider_upper_case)
        
    Raises:
        ValueError: If URL parsing fails or provider cannot be determined
    """
    try:
        parsed = urlparse(base_url)
        if not parsed.netloc:
            raise ValueError(f"Invalid base_url: no netloc found in {base_url}")
            
        netloc = parsed.netloc
        parts = netloc.split('.')
        
        # Extract provider name more robustly
        if len(parts) < 2:
            if ':' in netloc:
                provider = netloc.split(':')[0]
            else:
                provider = netloc
        else:
            if ':' in parts[-2]:
                provider = parts[-2].split(':')[0]
            else:
                provider = parts[-2]
        
        provider = provider.lower().strip()
        if not provider:
            raise ValueError(f"Could not determine provider from URL: {base_url}")
        
        # For localhost or unknown providers, use a generic provider name
        if provider in ['localhost', 'unknown']:
            provider = 'lightllm'
            
        # Set full model name based on provider
        if provider == 'openai':
            full_model = f"openai/{model}"
        else:
            full_model = f"openai-api/{provider}/{model}"
        
        # Always set OpenAI compatibility variables
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        
        # Set provider-specific environment variables
        provider_upper = provider.upper().replace('-', '_')
        os.environ[f"{provider_upper}_API_KEY"] = api_key
        os.environ[f"{provider_upper}_BASE_URL"] = base_url
        
        return full_model, provider_upper
        
    except Exception as e:
        raise ValueError(f"Failed to configure provider environment: {e}") from e


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
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        sut_params = systems_params.get("system_under_test", {})
        if not sut_params:
            raise ValueError("Missing system_under_test in systems_params")

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
        evaluation = test_params.get("evaluation", "medqa")
        evaluation_params = test_params.get("evaluation_params", {})
        limit = test_params.get("limit", 10)
        volumes = test_params.get("volumes", {})
        
        store_logs = "output" in volumes

        # Set environment variables for inspect
        # Inspect requires model names in the format "provider/model" or "openai-api/provider/model"
        full_model, _ = _configure_provider_environment(base_url, model, api_key)
        
        os.environ["INSPECT_EVAL_MODEL"] = full_model
        os.environ["INSPECT_GRADER_MODEL"] = full_model

        # Get evaluation function
        if evaluation not in EVALUATION_REGISTRY:
            available_evals = ", ".join(sorted(EVALUATION_REGISTRY.keys()))
            raise ValueError(f"Unsupported evaluation: {evaluation}. Supported evaluations: {available_evals}")

        module_name, enabled = EVALUATION_REGISTRY[evaluation]

        if not enabled:
            raise ValueError(f"Evaluation '{evaluation}' is currently disabled.")
        
        try:
            module = __import__(module_name, fromlist=[evaluation])
            task_func = getattr(module, evaluation)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import evaluation '{evaluation}' from {module_name}.{evaluation}: {e}")

        with tempfile.TemporaryDirectory() as temp_dir:
            task = task_func(**evaluation_params)

            print(f"DEBUG: Starting evaluation with limit={limit}", file=sys.stderr)
            log = eval(
                task,
                limit=limit,
                log_dir=temp_dir
            )[0]  # eval returns a list, get first result
            print(f"DEBUG: Evaluation completed, log.location = {log.location}", file=sys.stderr)

            persistent_log_dir = None
            if store_logs:
                test_name = os.environ.get("TEST_NAME", evaluation)
                persistent_log_dir = f"/output/{test_name}"
                if os.path.exists(persistent_log_dir):
                    if os.path.isdir(persistent_log_dir):
                        shutil.rmtree(persistent_log_dir)
                    else:
                        os.remove(persistent_log_dir)
                
                if os.path.isdir(log.location):
                    shutil.copytree(log.location, persistent_log_dir)
                else:
                    # Create directory and copy file
                    os.makedirs(persistent_log_dir, exist_ok=True)
                    shutil.copy2(log.location, os.path.join(persistent_log_dir, os.path.basename(log.location)))

            metrics = {}
            try:
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

            result = {
                "success": True,
                "evaluation": evaluation,
                "evaluation_params": evaluation_params,
                "total_samples": total_samples,
                "model": model,
                "metrics": metrics,
                "log_dir": persistent_log_dir
            }

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
