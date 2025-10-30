import argparse
import json
import os
import shutil
import sys
import tempfile
from urllib.parse import urlparse

# Import inspect modules
from inspect_ai import eval
from inspect_ai.log import read_eval_log


EVALUATION_REGISTRY = {
    ## Coding
    # ❌ APPS
    "apps": ("inspect_evals.apps", False),
    # ✅ AgentBench
    "agent_bench_os": ("inspect_evals.agent_bench", False),
    # ❌ BigCodeBench
    "bigcodebench": ("inspect_evals.bigcodebench", False),  # TODO: very large
    # ❌ CORE-Bench
    "core_bench": ("inspect_evals.core_bench", False),  # TODO
    # ✅ ClassEval
    "class_eval": ("inspect_evals.class_eval", False),
    # ✅ DS-1000
    "ds1000": ("inspect_evals.ds1000", False),
    # ✅ HumanEval
    "humaneval": ("inspect_evals.humaneval", True),
    # ✅ MBPP
    "mbpp": ("inspect_evals.mbpp", False),
    # ❌ MLE-bench
    "mle_bench": ("inspect_evals.mle_bench", False),  # TODO: Kaggle Auth required
    "mle_bench_full": ("inspect_evals.mle_bench", False),  # TODO: Kaggle Auth required
    "mle_bench_lite": ("inspect_evals.mle_bench", False),  # TODO: Kaggle Auth required
    # ❌ SWE-bench
    "swe_bench": ("inspect_evals.swe_bench", False),  # TODO extra deps
    "swe_bench_verified_mini": ("inspect_evals.swe_bench", False),  # TODO extra deps
    # ❌ SciCode
    "scicode": ("inspect_evals.scicode", False),
    # ✅ USACO
    "usaco": ("inspect_evals.usaco", False),
    ## Assistants
    # ❌ AssistantBench
    "assistant_bench_closed_book_one_shot": ("inspect_evals.assistant_bench", False),
    "assistant_bench_closed_book_zero_shot": ("inspect_evals.assistant_bench", False),
    "assistant_bench_web_browser": (
        "inspect_evals.assistant_bench",
        False,
    ),
    "assistant_bench_web_search_one_shot": (
        "inspect_evals.assistant_bench",
        False,
    ),
    "assistant_bench_web_search_zero_shot": (
        "inspect_evals.assistant_bench",
        False,
    ),
    "browse_comp": ("inspect_evals.browse_comp", False),
    "gaia": (
        "inspect_evals.gaia",
        False,
    ),  # Info: Requires HF_TOKEN and access to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level1": (
        "inspect_evals.gaia",
        True,
    ),  # Info: Requires HF_TOKEN and access to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level2": (
        "inspect_evals.gaia",
        True,
    ),  # Info: Requires HF_TOKEN and access to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    "gaia_level3": (
        "inspect_evals.gaia",
        True,
    ),  # Info: Requires HF_TOKEN and access to GAIA: https://huggingface.co/datasets/gaia-benchmark/GAIA
    # ❌ Mind2Web
    "mind2web": ("inspect_evals.mind2web", True),
    # ❌ OSWorld
    "osworld": ("inspect_evals.osworld", True),
    # ❌ Sycophancy
    "sycophancy": ("inspect_evals.sycophancy", True),
    ## ✅ Cybersecurity
    "cyse3_visual_prompt_injection": ("inspect_evals.cyberseceval_3", True),
    "threecb": ("inspect_evals.threecb", True),
    "cybench": ("inspect_evals.cybench", False),
    "cybermetric_80": ("inspect_evals.cybermetric", True),
    "cybermetric_500": ("inspect_evals.cybermetric", True),
    "cybermetric_2000": ("inspect_evals.cybermetric", True),
    "cybermetric_10000": ("inspect_evals.cybermetric", True),
    "cyse2_interpreter_abuse": ("inspect_evals.cyberseceval_2", True),
    "cyse2_prompt_injection": ("inspect_evals.cyberseceval_2", True),
    "cyse2_vulnerability_exploit": ("inspect_evals.cyberseceval_2", True),
    "gdm_in_house_ctf": ("inspect_evals.gdm_capabilities", False),
    "gdm_intercode_ctf": ("inspect_evals.gdm_capabilities", True),
    "sevenllm_mcq_en": ("inspect_evals.sevenllm", True),
    "sevenllm_mcq_zh": ("inspect_evals.sevenllm", True),
    "sevenllm_qa_en": ("inspect_evals.sevenllm", True),
    "sevenllm_qa_zh": ("inspect_evals.sevenllm", True),
    "sec_qa_v1": ("inspect_evals.sec_qa", True),
    "sec_qa_v1_5_shot": ("inspect_evals.sec_qa", True),
    "sec_qa_v2": ("inspect_evals.sec_qa", True),
    "sec_qa_v2_5_shot": ("inspect_evals.sec_qa", True),
    ## ✅ Safeguards
    "abstention_bench": (
        "inspect_evals.abstention_bench",
        True,
    ),  # Info: #1 Dataset 'Idavidrein/gpqa' on HuggingFace is a gated dataset. You must be authenticated to access it. #2 Use openai scorer model as the default requires openrouter
    "agentdojo": ("inspect_evals.agentdojo", True),
    "agentharm": ("inspect_evals.agentharm", True),
    "agentharm_benign": ("inspect_evals.agentharm", True),
    "lab_bench_cloning_scenarios": ("inspect_evals.lab_bench", True),
    "lab_bench_dbqa": ("inspect_evals.lab_bench", True),
    "lab_bench_figqa": ("inspect_evals.lab_bench", True),
    "lab_bench_litqa": ("inspect_evals.lab_bench", True),
    "lab_bench_protocolqa": ("inspect_evals.lab_bench", True),
    "lab_bench_seqqa": ("inspect_evals.lab_bench", True),
    "mask": (
        "inspect_evals.mask",
        True,
    ),  # Info: Dataset 'cais/MASK' on HuggingFace is a gated dataset. You must be authenticated to access it.
    "make_me_pay": ("inspect_evals.make_me_pay", True),
    "stereoset": ("inspect_evals.stereoset", True),
    "strong_reject": ("inspect_evals.strong_reject", True),
    "wmdp_bio": ("inspect_evals.wmdp", True),
    "wmdp_chem": ("inspect_evals.wmdp", True),
    "wmdp_cyber": ("inspect_evals.wmdp", True),
    ## ✅ Mathematics
    "aime2024": ("inspect_evals.aime2024", True),
    "gsm8k": ("inspect_evals.gsm8k", True),
    "math": ("inspect_evals.math", True),
    "mgsm": ("inspect_evals.mgsm", True),
    "mathvista": ("inspect_evals.mathvista", True),
    ## ✅ Reasoning
    "arc_challenge": ("inspect_evals.arc", True),
    "arc_easy": ("inspect_evals.arc", True),
    "bbh": ("inspect_evals.bbh", True),
    "bbeh": ("inspect_evals.bbeh", True),
    "bbeh_mini": ("inspect_evals.bbeh", True),
    "boolq": ("inspect_evals.boolq", True),
    "drop": ("inspect_evals.drop", True),
    "hellaswag": ("inspect_evals.hellaswag", True),
    "ifeval": ("inspect_evals.ifeval", True),
    "lingoly": (
        "inspect_evals.lingoly",
        True,
    ),  # Info: Dataset 'ambean/lingOly' is a gated dataset on the Hub. You must be authenticated to access it.
    "lingoly_too": ("inspect_evals.lingoly", True),
    "mmmu_multiple_choice": ("inspect_evals.mmmu", True),
    "mmmu_open": ("inspect_evals.mmmu", True),
    "musr": ("inspect_evals.musr", True),
    "niah": ("inspect_evals.niah", True),
    "paws": ("inspect_evals.paws", True),
    "piqa": ("inspect_evals.piqa", True),
    "race_h": ("inspect_evals.race_h", True),
    "squad": (
        "inspect_evals.squad",
        False,
    ),  # TODO: ValueError: Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', ...
    "winogrande": ("inspect_evals.winogrande", True),
    "worldsense": ("inspect_evals.worldsense", True),
    "infinite_bench_code_debug": ("inspect_evals.infinite_bench", True),
    "infinite_bench_code_run": ("inspect_evals.infinite_bench", True),
    "infinite_bench_kv_retrieval": ("inspect_evals.infinite_bench", True),
    "infinite_bench_longbook_choice_eng": ("inspect_evals.infinite_bench", True),
    "infinite_bench_longdialogue_qa_eng": ("inspect_evals.infinite_bench", True),
    "infinite_bench_math_calc": ("inspect_evals.infinite_bench", True),
    "infinite_bench_math_find": ("inspect_evals.infinite_bench", True),
    "infinite_bench_number_string": ("inspect_evals.infinite_bench", True),
    "infinite_bench_passkey": ("inspect_evals.infinite_bench", True),
    ## ✅ Knowledge
    "agie_aqua_rat": ("inspect_evals.agieval", True),
    "agie_logiqa_en": ("inspect_evals.agieval", True),
    "agie_lsat_ar": ("inspect_evals.agieval", True),
    "agie_lsat_lr": ("inspect_evals.agieval", True),
    "agie_lsat_rc": ("inspect_evals.agieval", True),
    "agie_math": ("inspect_evals.agieval", True),
    "agie_sat_en": ("inspect_evals.agieval", True),
    "agie_sat_en_without_passage": ("inspect_evals.agieval", True),
    "agie_sat_math": ("inspect_evals.agieval", True),
    "air_bench": ("inspect_evals.air_bench", True),
    "chembench": ("inspect_evals.chembench", True),
    "commonsense_qa": ("inspect_evals.commonsense_qa", True),
    "gpqa_diamond": ("inspect_evals.gpqa", True),
    "healthbench": ("inspect_evals.healthbench", True),
    "hle": ("inspect_evals.hle", True),  # Info: Gated Dataset (cais/hle on HuggingFace)
    "livebench": ("inspect_evals.livebench", True),
    "mmlu_pro": ("inspect_evals.mmlu_pro", True),
    "mmlu_0_shot": ("inspect_evals.mmlu", True),
    "mmlu_5_shot": ("inspect_evals.mmlu", True),
    "medqa": ("inspect_evals.medqa", True),
    "onet_m6": ("inspect_evals.onet", True),
    "pre_flight": ("inspect_evals.pre_flight", True),
    "pubmedqa": ("inspect_evals.pubmedqa", True),
    "sosbench": (
        "inspect_evals.sosbench",
        True,
    ),  # Info: Fails with bedrock: "The model returned the\nfollowing errors: Malformed input request: #: extraneous key [thinking] is not  \npermitted, please reformat your input and try again.
    "sciknoweval": ("inspect_evals.sciknoweval", True),
    "simpleqa": ("inspect_evals.simpleqa", True),
    "truthfulqa": ("inspect_evals.truthfulqa", True),
    "xstest": (
        "inspect_evals.xstest",
        True,
    ),  # Info: gated dataset (walledai/XSTest on HuggingFace)
    ## ✅ Scheming
    "agentic_misalignment": ("inspect_evals.agentic_misalignment", True),
    "gdm_approved_directories": ("inspect_evals.gdm_capabilities", True),
    "gdm_calculator_improvement": ("inspect_evals.gdm_capabilities", True),
    "gdm_context_length_mod_instrumental_only": (
        "inspect_evals.gdm_capabilities",
        True,
    ),
    "gdm_context_length_mod_irreversibility_only": (
        "inspect_evals.gdm_capabilities",
        True,
    ),
    "gdm_database_tool": ("inspect_evals.gdm_capabilities", True),
    "gdm_latency_calculator": ("inspect_evals.gdm_capabilities", True),
    "gdm_max_messages_calculator": ("inspect_evals.gdm_capabilities", True),
    "gdm_max_tokens": ("inspect_evals.gdm_capabilities", True),
    "gdm_oversight_frequency": ("inspect_evals.gdm_capabilities", True),
    "gdm_read_logs": ("inspect_evals.gdm_capabilities", True),
    "gdm_turn_off_filters": ("inspect_evals.gdm_capabilities", True),
    "gdm_classifier_evasion": ("inspect_evals.gdm_capabilities", True),
    "gdm_cover_your_tracks": ("inspect_evals.gdm_capabilities", True),
    "gdm_oversight_pattern": ("inspect_evals.gdm_capabilities", True),
    "gdm_strategic_rule_breaking": ("inspect_evals.gdm_capabilities", True),
    ## Multimodal
    # ❌ DocVQA
    "docvqa": ("inspect_evals.docvqa", False),
    # ❌ MMIU
    "mmiu": ("inspect_evals.mmiu", False),
    # ❌ V*Bench
    "vstar_bench_attribute_recognition": ("inspect_evals.vstar_bench", False),
    "vstar_bench_spatial_relationship_reasoning": ("inspect_evals.vstar_bench", False),
    # ✅ ZeroBench
    "zerobench": ("inspect_evals.zerobench", False),
    "zerobench_subquestions": ("inspect_evals.zerobench", False),
    ## ✅ Bias
    "bbq": ("inspect_evals.bbq", True),
    "bold": ("inspect_evals.bold", True),
    ## ✅ Personality
    "personality_BFI": ("inspect_evals.personality", True),
    "personality_TRAIT": (
        "inspect_evals.personality",
        True,
    ),  # Info: Gated Dataset (mirlab/TRAIT on HuggingFace)
    ## ✅ Writing
    "writingbench": (
        "inspect_evals.writingbench",
        True,
    ),  # Info: Set judge_model to "openai/{model}"
}

# Configure datasets to trust remote code before any other imports
try:
    import datasets

    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
except ImportError:
    pass


def _configure_provider_environment(
    base_url: str, model: str, api_key: str
) -> tuple[str, str]:
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
        parts = netloc.split(".")

        # Extract provider name more robustly
        if len(parts) < 2:
            if ":" in netloc:
                provider = netloc.split(":")[0]
            else:
                provider = netloc
        else:
            if ":" in parts[-2]:
                provider = parts[-2].split(":")[0]
            else:
                provider = parts[-2]

        provider = provider.lower().strip()
        if not provider:
            raise ValueError(f"Could not determine provider from URL: {base_url}")

        # For localhost or unknown providers, use a generic provider name
        if provider in ["localhost", "unknown"]:
            provider = "lightllm"

        # Set full model name based on provider
        if provider == "openai":
            full_model = f"openai/{model}"
        else:
            full_model = f"openai-api/{provider}/{model}"

        # Always set OpenAI compatibility variables
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url

        # Set provider-specific environment variables
        provider_upper = provider.upper().replace("-", "_")
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
        sut_name = sut_params.get("name")
        base_url = sut_params.get("base_url")
        api_key = sut_params.get("api_key")
        model = sut_params.get("model")

        if not base_url or not api_key or not model:
            raise ValueError("Missing required parameters: base_url, api_key, or model")

        # Extract test parameters
        evaluation = test_params.get("evaluation", "ifeval")
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
            raise ValueError(
                f"Unsupported evaluation: {evaluation}. Supported evaluations: {available_evals}"
            )

        module_name, enabled = EVALUATION_REGISTRY[evaluation]

        if not enabled:
            raise ValueError(f"Evaluation '{evaluation}' is currently disabled.")

        try:
            module = __import__(module_name, fromlist=[evaluation])
            task_func = getattr(module, evaluation)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to import evaluation '{evaluation}' from {module_name}.{evaluation}: {e}"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            task = task_func(**evaluation_params)

            print(f"DEBUG: Starting evaluation with limit={limit}", file=sys.stderr)
            log = eval(task, limit=limit, log_dir=temp_dir)[
                0
            ]  # eval returns a list, get first result
            print(
                f"DEBUG: Evaluation completed, log.location = {log.location}",
                file=sys.stderr,
            )

            persistent_log_dir = None
            if store_logs:
                test_name = os.environ.get("TEST_NAME", evaluation)
                persistent_log_dir = f"/output/{test_name}_{sut_name}"
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
                    shutil.copy2(
                        log.location,
                        os.path.join(
                            persistent_log_dir, os.path.basename(log.location)
                        ),
                    )

            # ---  metrics + sample extraction ---------------------------------
            try:
                results = getattr(log, "results", None)
                if results is None:
                    try:
                        disk_log = read_eval_log(log.location)
                        results = getattr(disk_log, "results", None)
                        print(
                            "DEBUG: Used disk log fallback for results", file=sys.stderr
                        )
                    except Exception as e:
                        print(f"DEBUG: Failed disk log fallback: {e}", file=sys.stderr)

                if results is None:
                    raise ValueError("Evaluation failed - no results available")

                # Normalize to first score group
                score_group = None
                if hasattr(results, "scores") and getattr(results, "scores"):
                    score_group = results.scores[0]
                elif (
                    isinstance(results, list)
                    and results
                    and hasattr(results[0], "scores")
                    and results[0].scores
                ):
                    score_group = results[0].scores[0]

                metrics = {}
                if score_group and getattr(score_group, "metrics", None):
                    for name, metric_obj in score_group.metrics.items():
                        value = getattr(metric_obj, "value", None)
                        if value is not None:
                            metrics[name] = value

                # Determine sample count
                if score_group and getattr(score_group, "samples", None):
                    total_samples = len(score_group.samples)
                elif getattr(log, "samples", None):
                    total_samples = len(log.samples)
                else:
                    total_samples = 0

                # Preserve previous behavior: default to limit if still unknown
                if total_samples == 0:
                    total_samples = limit
            except Exception as e:
                print(f"DEBUG: Metric extraction failed: {e}", file=sys.stderr)
                metrics = {}
                total_samples = limit

            result = {
                "success": True,
                "evaluation": evaluation,
                "evaluation_params": evaluation_params,
                "total_samples": total_samples,
                "model": model,
                "metrics": metrics,
                "log_dir": persistent_log_dir,
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
            "metrics": {},
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
            "metrics": {},
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
