import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Define available evaluations with their import paths and functions
EVALUATION_REGISTRY = {
    ## Coding
    # ❌ APPS
    "apps": ("inspect_evals.apps", "apps"), # TODO
    # ✅ AgentBench
    "agent_bench_os": ("inspect_evals.agent_bench", "agent_bench_os"), 
    # ❌ BigCodeBench
    "bigcodebench": ("inspect_evals.bigcodebench", "bigcodebench"), # TODO: very large
    # ❌ CORE-Bench
    "core_bench": ("inspect_evals.core_bench", "core_bench"), # TODO
    # ✅ ClassEval
    "class_eval": ("inspect_evals.class_eval", "class_eval"), 
    # ✅ DS-1000
    "ds1000": ("inspect_evals.ds1000", "ds1000"), 
    # ✅ HumanEval
    "humaneval": ("inspect_evals.humaneval", "humaneval"), 
    # ✅ MBPP
    "mbpp": ("inspect_evals.mbpp", "mbpp"), 
    # ❌ MLE-bench
    "mle_bench": ("inspect_evals.mle_bench", "mle_bench"), # TODO: Kaggle Auth required
    "mle_bench_full": ("inspect_evals.mle_bench", "mle_bench_full"), # TODO: Kaggle Auth required
    "mle_bench_lite": ("inspect_evals.mle_bench", "mle_bench_lite"), # TODO: Kaggle Auth required
    # ❌ SWE-bench
    "swe_bench": ("inspect_evals.swe_bench", "swe_bench"), # TODO extra deps
    "swe_bench_verified_mini": ("inspect_evals.swe_bench", "swe_bench_verified_mini"), # TODO extra deps
    # ❌ SciCode
    "scicode": ("inspect_evals.scicode", "scicode"), # TODO
    # ✅ USACO
    "usaco": ("inspect_evals.usaco", "usaco"), # TODO

    ## Assistants
    # 🟡 AssistantBench
    "assistant_bench_closed_book_one_shot": ("inspect_evals.assistant_bench", "assistant_bench_closed_book_one_shot"), 
    "assistant_bench_closed_book_zero_shot": ("inspect_evals.assistant_bench", "assistant_bench_closed_book_zero_shot"), 
    "assistant_bench_web_browser": ("inspect_evals.assistant_bench", "assistant_bench_web_browser"), # TODO: Requires GOOGLE_CSE_API_KEY
    "assistant_bench_web_search_one_shot": ("inspect_evals.assistant_bench", "assistant_bench_web_search_one_shot"), # TODO: Requires GOOGLE_CSE_API_KEY
    "assistant_bench_web_search_zero_shot": ("inspect_evals.assistant_bench", "assistant_bench_web_search_zero_shot"), # TODO: Requires GOOGLE_CSE_API_KEY
    # ✅ BrowseComp
    "browse_comp": ("inspect_evals.browse_comp", "browse_comp"), 
    # 🟡 GAIA
    "gaia": ("inspect_evals.gaia", "gaia"), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level1": ("inspect_evals.gaia", "gaia_level1"), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level2": ("inspect_evals.gaia", "gaia_level2"), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level3": ("inspect_evals.gaia", "gaia_level3"), # TODO: Requires HF_TOKEN and acces to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    # ❌ Mind2Web
    "mind2web": ("inspect_evals.mind2web", "mind2web"), # TODO
    # ❌ OSWorld
    "osworld": ("inspect_evals.osworld", "osworld"), # TODO
    # ✅ Sycophancy
    "sycophancy": ("inspect_evals.sycophancy", "sycophancy"), 

    ## Cybersecurity
    # ❌ CYBERSECEVAL 3
    "visual_prompt_injection": ("inspect_evals.cyberseceval_3", "visual_prompt_injection"), # TODO
    # ❌ Catastrophic Cyber Capabilities Benchmark (3CB)
    "threecb": ("inspect_evals.threecb", "threecb"), # TODO
    # ❌ Cybench
    "cybench": ("inspect_evals.cybench", "cybench"), # TODO simple
    # ✅ CyberMetric
    "cybermetric_80": ("inspect_evals.cybermetric", "cybermetric_80"), 
    "cybermetric_500": ("inspect_evals.cybermetric", "cybermetric_500"), 
    "cybermetric_2000": ("inspect_evals.cybermetric", "cybermetric_2000"), #done
    "cybermetric_10000": ("inspect_evals.cybermetric", "cybermetric_10000"), 
    # ✅ CyberSecEval_2
    "cyse2_interpreter_abuse": ("inspect_evals.cyberseceval_2", "cyse2_interpreter_abuse"), 
    "cyse2_prompt_injection": ("inspect_evals.cyberseceval_2", "cyse2_prompt_injection"), 
    "cyse2_vulnerability_exploit": ("inspect_evals.cyberseceval_2", "cyse2_vulnerability_exploit"), 
    # ✅ GDM Dangerous Capabilities: Capture the Flag
    "gdm_in_house_ctf": ("inspect_evals.gdm_capabilities", "gdm_in_house_ctf"), 
    # ✅ InterCode: Security and Coding Capture-the-Flag Challenges
    "gdm_intercode_ctf": ("inspect_evals.gdm_capabilities", "gdm_intercode_ctf"), # TODO
    # 🟡 SEvenLLM
    "sevenllm_mcq_en": ("inspect_evals.sevenllm", "sevenllm_mcq_en"), 
    "sevenllm_mcq_zh": ("inspect_evals.sevenllm", "sevenllm_mcq_zh"), 
    "sevenllm_qa_en": ("inspect_evals.sevenllm", "sevenllm_qa_en"), 
    "sevenllm_qa_zh": ("inspect_evals.sevenllm", "sevenllm_qa_zh"), # TODO: keras 3 issue
    # ✅ SecQA
    "sec_qa_v1": ("inspect_evals.sec_qa", "sec_qa_v1"), 
    "sec_qa_v1_5_shot": ("inspect_evals.sec_qa", "sec_qa_v1_5_shot"), 
    "sec_qa_v2": ("inspect_evals.sec_qa", "sec_qa_v2"), 
    "sec_qa_v2_5_shot": ("inspect_evals.sec_qa", "sec_qa_v2_5_shot"), 

    ## Safeguards
    # ❌ AbstentionBench
    "abstention_bench": ("inspect_evals.abstention_bench", "abstention_bench"), # TODO
    # ✅ AgentDojo
    "agentdojo": ("inspect_evals.agentdojo", "agentdojo"), 
    # ✅ AgentHarm
    "agentharm": ("inspect_evals.agentharm", "agentharm"), 
    "agentharm_benign": ("inspect_evals.agentharm", "agentharm_benign"), 
    # ✅ LAB-Bench
    "lab_bench_cloning_scenarios": ("inspect_evals.lab_bench", "lab_bench_cloning_scenarios"), 
    "lab_bench_dbqa": ("inspect_evals.lab_bench", "lab_bench_dbqa"),  
    "lab_bench_figqa": ("inspect_evals.lab_bench", "lab_bench_figqa"), 
    "lab_bench_litqa": ("inspect_evals.lab_bench", "lab_bench_litqa"), 
    "lab_bench_protocolqa": ("inspect_evals.lab_bench", "lab_bench_protocolqa"), 
    "lab_bench_seqqa": ("inspect_evals.lab_bench", "lab_bench_seqqa"), 
    # ❌ MASK
    "mask": ("inspect_evals.mask", "mask"), # TODO
    # ✅ StereoSet
    "stereoset": ("inspect_evals.stereoset", "stereoset"), 
    # ✅ StrongREJECT
    "strong_reject": ("inspect_evals.strong_reject", "strong_reject"),  
    # ✅ WMDP
    "wmdp_bio": ("inspect_evals.wmdp", "wmdp_bio"), 
    "wmdp_chem": ("inspect_evals.wmdp", "wmdp_chem"), 
    "wmdp_cyber": ("inspect_evals.wmdp", "wmdp_cyber"), 

    ## ✅✅ Mathematics
    # ✅ AIME 2024
    "aime2024": ("inspect_evals.aime2024", "aime2024"), 
    # ✅ GSM8K
    "gsm8k": ("inspect_evals.gsm8k", "gsm8k"),  
    # ✅ MATH
    "math": ("inspect_evals.math", "math"),  
    # ✅ MGSM
    "mgsm": ("inspect_evals.mgsm", "mgsm"), 
    # ✅ MathVista
    "mathvista": ("inspect_evals.mathvista", "mathvista"), 

    ## Reasoning
    # ✅ ARC
    "arc": ("inspect_evals.arc", "arc_challenge"),  
    "arc_challenge": ("inspect_evals.arc", "arc_challenge"), 
    "arc_easy": ("inspect_evals.arc", "arc_easy"), 
    # ✅ BBH
    "bbh": ("inspect_evals.bbh", "bbh"), 
    # ✅ BIG-Bench Extra Hard
    "bbeh": ("inspect_evals.bbeh", "bbeh"), 
    "bbeh_mini": ("inspect_evals.bbeh", "bbeh_mini"), 
    # ✅ BoolQ
    "boolq": ("inspect_evals.boolq", "boolq"), 
    # ✅ DROP
    "drop": ("inspect_evals.drop", "drop"),  
    # ✅ HellaSwag
    "hellaswag": ("inspect_evals.hellaswag", "hellaswag"),  
    # ✅ IFEval
    "ifeval": ("inspect_evals.ifeval", "ifeval"), 
    # 🟡 LingOly
    "lingoly": ("inspect_evals.lingoly", "lingoly"), # TODO: HuggingFace authentication required
    "lingoly_too": ("inspect_evals.lingoly", "lingoly_too"), 
    # ✅ MMMU
    "mmmu_multiple_choice": ("inspect_evals.mmmu", "mmmu_multiple_choice"), 
    "mmmu_open": ("inspect_evals.mmmu", "mmmu_open"), 
    # ✅ MuSR
    "musr": ("inspect_evals.musr", "musr"), 
    # ✅ Needle in a Haystack (NIAH)
    "niah": ("inspect_evals.niah", "niah"), 
    # ✅ PAWS
    "paws": ("inspect_evals.paws", "paws"), 
    # ✅ PIQA
    "piqa": ("inspect_evals.piqa", "piqa"),  
    # ✅ RACE-H
    "race_h": ("inspect_evals.race_h", "race_h"), 
    # 🟡 SQuAD
    "squad": ("inspect_evals.squad", "squad"), # TODO: Dataset issue on HuggingFace side
    # ✅ WINOGRANDE
    "winogrande": ("inspect_evals.winogrande", "winogrande"), 
    # ✅ WorldSense
    "worldsense": ("inspect_evals.worldsense", "worldsense"), 
    # ✅ ∞Bench
    "infinite_bench_code_debug": ("inspect_evals.infinite_bench", "infinite_bench_code_debug"), 
    "infinite_bench_code_run": ("inspect_evals.infinite_bench", "infinite_bench_code_run"), 
    "infinite_bench_kv_retrieval": ("inspect_evals.infinite_bench", "infinite_bench_kv_retrieval"), 
    "infinite_bench_longbook_choice_eng": ("inspect_evals.infinite_bench", "infinite_bench_longbook_choice_eng"), 
    "infinite_bench_longdialogue_qa_eng": ("inspect_evals.infinite_bench", "infinite_bench_longdialogue_qa_eng"), 
    "infinite_bench_math_calc": ("inspect_evals.infinite_bench", "infinite_bench_math_calc"), 
    "infinite_bench_math_find": ("inspect_evals.infinite_bench", "infinite_bench_math_find"), 
    "infinite_bench_number_string": ("inspect_evals.infinite_bench", "infinite_bench_number_string"), 
    "infinite_bench_passkey": ("inspect_evals.infinite_bench", "infinite_bench_passkey"), 

    ## Knowledge
    # ✅ AGIEval
    "agie_aqua_rat": ("inspect_evals.agieval", "agie_aqua_rat"), 
    "agie_logiqa_en": ("inspect_evals.agieval", "agie_logiqa_en"), 
    "agie_lsat_ar": ("inspect_evals.agieval", "agie_lsat_ar"), 
    "agie_lsat_lr": ("inspect_evals.agieval", "agie_lsat_lr"), 
    "agie_lsat_rc": ("inspect_evals.agieval", "agie_lsat_rc"), 
    "agie_math": ("inspect_evals.agieval", "agie_math"), 
    "agie_sat_en": ("inspect_evals.agieval", "agie_sat_en"), 
    "agie_sat_en_without_passage": ("inspect_evals.agieval", "agie_sat_en_without_passage"), 
    "agie_sat_math": ("inspect_evals.agieval", "agie_sat_math"), #done
    # ✅ AIR Bench
    "air_bench": ("inspect_evals.air_bench", "air_bench"), 
    # ✅ ChemBench
    "chembench": ("inspect_evals.chembench", "chembench"), 
    # ✅ CommonsenseQA
    "commonsense_qa": ("inspect_evals.commonsense_qa", "commonsense_qa"),  
    # ✅ GPQA
    "gpqa_diamond": ("inspect_evals.gpqa", "gpqa_diamond"),  
    # ✅ HealthBench
    "healthbench": ("inspect_evals.healthbench", "healthbench"), 
    # 🟡 Humanity's Last Exam
    "hle": ("inspect_evals.hle", "hle"), # TODO: HuggingFace authentication required
    # ✅ LiveBench
    "livebench": ("inspect_evals.livebench", "livebench"), 
    # ✅ MMLU-Pro
    "mmlu_pro": ("inspect_evals.mmlu_pro", "mmlu_pro"),  
    # ✅ MMLU
    "mmlu": ("inspect_evals.mmlu", "mmlu_0_shot"), 
    "mmlu_0_shot": ("inspect_evals.mmlu", "mmlu_0_shot"), 
    "mmlu_5_shot": ("inspect_evals.mmlu", "mmlu_5_shot"), 
    # ✅ MedQA
    "medqa": ("inspect_evals.medqa", "medqa"),  
    # ✅ O-NET
    "onet_m6": ("inspect_evals.onet", "onet_m6"), 
    # ✅ Pre-Flight
    "pre_flight": ("inspect_evals.pre_flight", "pre_flight"), 
    # ✅ PubMedQA
    "pubmedqa": ("inspect_evals.pubmedqa", "pubmedqa"), 
    # ✅ SOS BENCH
    "sosbench": ("inspect_evals.sosbench", "sosbench"), 
    # ❌ SciKnowEval
    "sciq": ("inspect_evals.sciknoweval", "sciknoweval"), # TODO
    # ✅ SimpleQA
    "simpleqa": ("inspect_evals.simpleqa", "simpleqa"), 
    # ✅ TruthfulQA
    "truthfulqa": ("inspect_evals.truthfulqa", "truthfulqa"),  
    # ✅ XSTest
    "xstest": ("inspect_evals.xstest", "xstest"),  

    ## Scheming
    # ✅ Agentic Misalignment
    "agentic_misalignment": ("inspect_evals.agentic_misalignment", "agentic_misalignment"), 
    # ❌ GDM Dangerous Capabilities: Self-reasoning
    "gdm_approved_directories": ("inspect_evals.gdm_capabilities", "gdm_approved_directories"), # TODO
    "gdm_calculator_improvement": ("inspect_evals.gdm_capabilities", "gdm_calculator_improvement"), # TODO
    "gdm_context_length_mod_instrumental_only": ("inspect_evals.gdm_capabilities", "gdm_context_length_mod_instrumental_only"), # TODO
    "gdm_context_length_mod_irreversibility_only": ("inspect_evals.gdm_capabilities", "gdm_context_length_mod_irreversibility_only"), # TODO
    "gdm_database_tool": ("inspect_evals.gdm_capabilities", "gdm_database_tool"), # TODO
    "gdm_latency_calculator": ("inspect_evals.gdm_capabilities", "gdm_latency_calculator"), # TODO
    "gdm_max_messages_calculator": ("inspect_evals.gdm_capabilities", "gdm_max_messages_calculator"), # TODO
    "gdm_max_tokens": ("inspect_evals.gdm_capabilities", "gdm_max_tokens"), # TODO
    "gdm_oversight_frequency": ("inspect_evals.gdm_capabilities", "gdm_oversight_frequency"), # TODO
    "gdm_read_logs": ("inspect_evals.gdm_capabilities", "gdm_read_logs"), # TODO
    "gdm_turn_off_filters": ("inspect_evals.gdm_capabilities", "gdm_turn_off_filters"), # TODO
    # ❌ GDM Dangerous Capabilities: Stealth
    "gdm_classifier_evasion": ("inspect_evals.gdm_capabilities", "gdm_classifier_evasion"), # TODO
    "gdm_cover_your_tracks": ("inspect_evals.gdm_capabilities", "gdm_cover_your_tracks"), # TODO
    "gdm_oversight_pattern": ("inspect_evals.gdm_capabilities", "gdm_oversight_pattern"), # TODO
    "gdm_strategic_rule_breaking": ("inspect_evals.gdm_capabilities", "gdm_strategic_rule_breaking"), # TODO

    ## Multimodal
    # ❌ DocVQA
    "docvqa": ("inspect_evals.docvqa", "docvqa"), # TODO
    # ❌ MMIU
    "mmiu": ("inspect_evals.mmiu", "mmiu"), # TODO
    # ❌ V*Bench
    "vstar_bench_attribute_recognition": ("inspect_evals.vstar_bench", "vstar_bench_attribute_recognition"), # TODO
    "vstar_bench_spatial_relationship_reasoning": ("inspect_evals.vstar_bench", "vstar_bench_spatial_relationship_reasoning"), # TODO
    # ✅ ZeroBench
    "zerobench": ("inspect_evals.zerobench", "zerobench"), 
    "zerobench_subquestions": ("inspect_evals.zerobench", "zerobench_subquestions"), 

    ## Bias
    # ✅ BBQ
    "bbq": ("inspect_evals.bbq", "bbq"),  
    # ❌ BOLD
    "bold": ("inspect_evals.bold", "bold"), # TODO: Your currently installed version of Keras is Keras 3, but this is not yet supported in Transformers.

    ## ✅✅ Personality
    # ✅ Personality
    "personality_BFI": ("inspect_evals.personality", "personality_BFI"), 
    "personality_TRAIT": ("inspect_evals.personality", "personality_TRAIT"), # Gated Dataset (mirlab/TRAIT on HuggingFace)
    
    ## Writing
    # 🟡 WritingBench
    "writingbench": ("inspect_evals.writingbench", "writingbench"), # TODO: requires ANTHROPIC_API_KEY
}

# Configure datasets to trust remote code before any other imports
try:
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
except ImportError:
    pass

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
        volumes = test_params.get("volumes", {})
        
        # Automatically store logs if output volume is configured
        store_logs = "output" in volumes

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

        # Set grader model to use the same model as the main evaluation to avoid authentication issues
        os.environ["INSPECT_GRADER_MODEL"] = full_model

        # Ensure HuggingFace datasets trust remote code
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

        # Import inspect modules
        from inspect_ai import eval as inspect_eval


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

            persistent_log_dir = None
            if store_logs:
                test_name = os.environ.get("TEST_NAME", evaluation)
                persistent_log_dir = f"/output/{test_name}"
                if os.path.exists(persistent_log_dir):
                    if os.path.isdir(persistent_log_dir):
                        shutil.rmtree(persistent_log_dir)
                    else:
                        os.remove(persistent_log_dir)
                
                # Check if log.location is a file or directory
                if os.path.isdir(log.location):
                    shutil.copytree(log.location, persistent_log_dir)
                else:
                    # Create directory and copy file
                    os.makedirs(persistent_log_dir, exist_ok=True)
                    shutil.copy2(log.location, os.path.join(persistent_log_dir, os.path.basename(log.location)))

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
                "metrics": metrics,
                "log_dir": persistent_log_dir
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
