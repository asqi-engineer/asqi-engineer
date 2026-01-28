import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from asqi.utils import get_openai_tracking_kwargs
from trustllm import config
from trustllm.generation.oai_generation import (
    EthicsDataset,
    FairnessDataset,
    OpenAILLMGeneration,
    PrivacyDataset,
    RobustnessDataset,
    SafetyDataset,
    TestType,
    TruthfulnessDataset,
)
from trustllm.task import (
    ethics,
    fairness,
    privacy,
    robustness,
    safety,
    truthfulness,
)
from trustllm.utils import file_process


class OpenAILLMGenerationWithMetadata(OpenAILLMGeneration):
    """OpenAILLMGeneration subclass that injects ASQI execution metadata into OpenAI/LiteLLM requests without modifying TrustLLM internals."""

    def __init__(self, *args, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata or {}

    def _generation_openai(self, prompt, temperature=0.0):
        """
        Generate response using OpenAI API.

        :param prompt: Input prompt string
        :param temperature: Temperature for generation
        :return: Generated text
        """
        try:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            tracking_kwargs = get_openai_tracking_kwargs(self.metadata)

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                **tracking_kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise e


class TrustLLMTester:
    """TrustLLM evaluation container for ASQI framework"""

    # Map test types to their corresponding dataset enums
    DATASET_MAP = {
        "ethics": EthicsDataset,
        "privacy": PrivacyDataset,
        "fairness": FairnessDataset,
        "truthfulness": TruthfulnessDataset,
        "robustness": RobustnessDataset,
        "safety": SafetyDataset,
    }

    # Default datasets for each test type (without .json extension)
    DEFAULT_DATASETS = {
        "ethics": [
            "awareness",
            "explicit_moralchoice",
            "implicit_ETHICS",
            "implicit_SocialChemistry101",
        ],
        "privacy": [
            "privacy_awareness_confAIde",
            "privacy_awareness_query",
            "privacy_leakage",
        ],
        "fairness": [
            "disparagement",
            "preference",
            "stereotype_agreement",
            "stereotype_query_test",
            "stereotype_recognition",
        ],
        "truthfulness": [
            "external",
            "hallucination",
            "golden_advfactuality",
            "internal",
            "sycophancy",
        ],
        "robustness": [
            "ood_detection",
            "ood_generalization",
            "AdvGLUE",
            "AdvInstruction",
        ],
        "safety": ["jailbreak", "exaggerated_safety", "misuse"],
    }

    def __init__(
        self, systems_params: Dict[str, Any], test_params: Dict[str, Any] = None
    ):
        """Initialize with systems parameters and optional test parameters"""
        self.sut_params = systems_params.get("system_under_test", {})
        self.systems_params = systems_params
        self.test_params = test_params or {}
        self.data_path = "/app/dataset/dataset"  # Fixed path to extracted dataset
        self.api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        config.openai_key = os.environ.get("OPENAI_API_KEY")

        # Extract ASQI metadata if available
        self.metadata = self.test_params.get("metadata", {})

        # Initialize evaluators for each test type
        self.evaluators = {
            "ethics": ethics.EthicsEval(),
            "privacy": privacy.PrivacyEval(),
            "fairness": fairness.FairnessEval(),
            "truthfulness": truthfulness.TruthfulnessEval(),
            "robustness": robustness.RobustnessEval(),
            "safety": safety.SafetyEval(),
        }

    def _get_dataset_enum(self, test_type: str, dataset_name: str) -> Any:
        dataset_enum_class = self.DATASET_MAP.get(test_type.lower())
        if not dataset_enum_class:
            return None

        dataset_filename = (
            dataset_name if dataset_name.endswith(".json") else f"{dataset_name}.json"
        )
        for enum_value in dataset_enum_class:
            if enum_value.value.lower() == dataset_filename.lower():
                return enum_value
        return None

    def _get_test_type_enum(self, test_type: str) -> TestType:
        test_type_map = {
            "ethics": TestType.ETHICS,
            "privacy": TestType.PRIVACY,
            "fairness": TestType.FAIRNESS,
            "truthfulness": TestType.TRUTHFULNESS,
            "robustness": TestType.ROBUSTNESS,
            "safety": TestType.SAFETY,
        }
        return test_type_map.get(test_type.lower())

    def _evaluate_results(
        self, test_type: str, dataset_name: str, generated_data: Any
    ) -> Dict[str, Any]:
        try:
            evaluator = self.evaluators.get(test_type.lower())
            if not evaluator:
                return {"error": f"No evaluator found for test_type: {test_type}"}
            evaluation_results = None
            if test_type.lower() == "robustness":
                if dataset_name.lower() == "advglue":
                    evaluation_results = evaluator.advglue_eval(generated_data)
                elif dataset_name.lower() == "advinstruction":
                    evaluation_results = evaluator.advinstruction_eval(generated_data)
                elif dataset_name.lower() == "ood_detection":
                    evaluation_results = evaluator.ood_detection(generated_data)
                elif dataset_name.lower() == "ood_generalization":
                    evaluation_results = evaluator.ood_generalization(generated_data)

            elif test_type.lower() == "ethics":
                if dataset_name.lower() == "awareness":
                    evaluation_results = evaluator.awareness_eval(generated_data)
                elif dataset_name.lower() == "explicit_moralchoice":
                    evaluation_results = evaluator.explicit_ethics_eval(
                        generated_data, eval_type="low"
                    )
                elif dataset_name.lower() == "implicit_ethics":
                    evaluation_results = evaluator.implicit_ethics_eval(
                        generated_data, eval_type="ETHICS"
                    )
                elif dataset_name.lower() == "implicit_socialchemistry101":
                    evaluation_results = evaluator.implicit_ethics_eval(
                        generated_data, eval_type="social_norm"
                    )

            elif test_type.lower() == "safety":
                if dataset_name.lower() == "jailbreak":
                    evaluation_results = evaluator.jailbreak_eval(
                        generated_data, eval_type="total"
                    )
                elif dataset_name.lower() == "exaggerated_safety":
                    evaluation_results = evaluator.exaggerated_eval(generated_data)
                elif dataset_name.lower() == "misuse":
                    evaluation_results = evaluator.misuse_eval(generated_data)

            elif test_type.lower() == "fairness":
                if dataset_name.lower() == "disparagement":
                    evaluation_results = evaluator.disparagement_eval(generated_data)
                elif dataset_name.lower() == "preference":
                    evaluation_results = evaluator.preference_eval(generated_data)
                elif dataset_name.lower() == "stereotype_recognition":
                    evaluation_results = evaluator.stereotype_recognition_eval(
                        generated_data
                    )
                elif dataset_name.lower() == "stereotype_agreement":
                    evaluation_results = evaluator.stereotype_agreement_eval(
                        generated_data
                    )
                elif dataset_name.lower() == "stereotype_query_test":
                    evaluation_results = evaluator.stereotype_query_eval(generated_data)

            elif test_type.lower() == "privacy":
                if dataset_name.lower() == "privacy_awareness_confaide":
                    evaluation_results = evaluator.ConfAIDe_eval(generated_data)
                elif dataset_name.lower() == "privacy_awareness_query":
                    evaluation_results = evaluator.awareness_query_eval(
                        generated_data, type="normal"
                    )
                elif dataset_name.lower() == "privacy_leakage":
                    evaluation_results = evaluator.leakage_eval(generated_data)

            elif test_type.lower() == "truthfulness":
                if dataset_name.lower() == "golden_advfactuality":
                    evaluation_results = evaluator.advfact_eval(generated_data)
                elif dataset_name.lower() == "hallucination":
                    evaluation_results = evaluator.hallucination_eval(generated_data)
                elif dataset_name.lower() == "sycophancy":
                    evaluation_results = evaluator.sycophancy_eval(
                        generated_data, eval_type="persona"
                    )
                elif dataset_name.lower() == "internal":
                    evaluation_results = evaluator.internal_eval(generated_data)
                elif dataset_name.lower() == "external":
                    evaluation_results = evaluator.external_eval(generated_data)

            if evaluation_results is None:
                return {
                    "error": f"No evaluation method found for {test_type}/{dataset_name}"
                }

            return {"evaluation_results": evaluation_results}

        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

    def run_trustllm_evaluation(self, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TrustLLM evaluation"""
        try:
            # Extract test parameters
            test_type = test_params.get("test_type")
            if not test_type:
                return {
                    "success": False,
                    "error": "test_type is required",
                }

            # Extract output directory parameter
            output_dir_name = test_params.get("output_dir", "trustllm_results")

            # Validate directory name to prevent directory traversal
            output_dir_path = Path(output_dir_name)
            if len(output_dir_path.parts) != 1:
                raise ValueError(
                    "Invalid output_dir: must be a directory name only, no directory traversal allowed."
                )

            test_type = test_type.lower()
            if test_type not in self.DATASET_MAP:
                return {
                    "success": False,
                    "error": f"Unsupported test_type: {test_type}. Must be one of: {list(self.DATASET_MAP.keys())}",
                }

            # Get datasets to test
            datasets = test_params.get("datasets")
            if not datasets:
                datasets = self.DEFAULT_DATASETS[test_type]

            # Validate all datasets exist before processing any
            valid_datasets = [name.lower() for name in self.DEFAULT_DATASETS[test_type]]
            invalid_datasets = []
            for dataset_name in datasets:
                if dataset_name.lower() not in valid_datasets:
                    invalid_datasets.append(dataset_name)

            if invalid_datasets:
                return {
                    "success": False,
                    "error": f"Invalid datasets for {test_type}: {invalid_datasets}. Valid datasets: {self.DEFAULT_DATASETS[test_type]}",
                }

            max_new_tokens = test_params.get("max_new_tokens", 1024)
            max_rows = test_params.get("max_rows")
            test_type_enum = self._get_test_type_enum(test_type)
            if not test_type_enum:
                return {
                    "success": False,
                    "error": f"Failed to map test_type {test_type} to enum",
                }

            dataset_results = {}
            datasets_tested = []
            for dataset_name in datasets:
                try:
                    dataset_enum = self._get_dataset_enum(test_type, dataset_name)
                    if not dataset_enum:
                        dataset_results[dataset_name] = {
                            "error": f"Failed to find dataset enum for {dataset_name}"
                        }
                        continue
                    llm_gen = OpenAILLMGenerationWithMetadata(
                        test_type=test_type_enum,
                        dataset=dataset_enum,
                        data_path=self.data_path,
                        model_name=self.sut_params["model"],
                        api_key=self.api_key,
                        base_url=self.sut_params["base_url"],
                        max_new_tokens=max_new_tokens,
                        max_rows=max_rows,
                        debug=False,
                        metadata=self.metadata,
                    )

                    generation_status = llm_gen.generation_results()
                    if generation_status == "OK":
                        results_file_path = f"generation_results/{self.sut_params['model']}/{test_type}/{dataset_enum.value}"

                        try:
                            generated_results = file_process.load_json(
                                results_file_path
                            )
                        except Exception as e:
                            print(
                                f"Failed to load results from {results_file_path}: {e}"
                            )
                            dataset_results[dataset_name] = {
                                "error": f"Failed to load generated results: {e}"
                            }
                            continue

                        actual_sample_count = (
                            len(generated_results)
                            if isinstance(generated_results, list)
                            else 1
                        )
                        evaluation_result = self._evaluate_results(
                            test_type, dataset_name, generated_results
                        )
                        combined_results = {
                            "evaluation_results": evaluation_result.get(
                                "evaluation_results"
                            ),
                            "evaluation_error": evaluation_result.get("error"),
                            "actual_sample_count": actual_sample_count,
                        }

                        dataset_results[dataset_name] = combined_results
                        datasets_tested.append(dataset_name)
                    else:
                        dataset_results[dataset_name] = {
                            "error": f"Generation failed with status: {generation_status}"
                        }

                except Exception as e:
                    dataset_results[dataset_name] = {"error": str(e)}
                    continue

            if not datasets_tested:
                return {
                    "success": False,
                    "error": "No datasets were successfully processed",
                    "test_type": test_type,
                    "datasets_tested": [],
                    "results_summary": {},
                }

            summary_results = {}
            for dataset_name in datasets_tested:
                dataset_result = dataset_results[dataset_name]

                summary_results[dataset_name] = {
                    "evaluation_score": dataset_result.get("evaluation_results"),
                    "evaluation_error": dataset_result.get("evaluation_error"),
                    "sample_count": dataset_result.get("actual_sample_count", 0),
                }

            # Copy relevant generation results to user-specified directory
            try:
                output_mount_path = Path(os.environ["OUTPUT_MOUNT_PATH"])
                generation_results_dir = Path("generation_results")

                if generation_results_dir.exists():
                    # Create target directory structure: {output_dir}/generation_results/{model}/{test_type}/
                    target_base_dir = output_mount_path / output_dir_name
                    model_dir = (
                        generation_results_dir / self.sut_params["model"] / test_type
                    )

                    if model_dir.exists():
                        target_dir = (
                            target_base_dir
                            / "generation_results"
                            / self.sut_params["model"]
                            / test_type
                        )
                        target_dir.mkdir(parents=True, exist_ok=True)

                        # Copy only the relevant test_type results
                        for dataset_name in datasets_tested:
                            for dataset_file in model_dir.glob(f"*{dataset_name}*"):
                                shutil.copy2(dataset_file, target_dir)

                        print(
                            f"TrustLLM generation results copied to: {target_dir}",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"Warning: No results found in {model_dir}", file=sys.stderr
                        )
                else:
                    print(
                        "Warning: No generation_results directory found to copy",
                        file=sys.stderr,
                    )
            except (OSError, IOError, PermissionError) as e:
                print(
                    f"Warning: Could not copy TrustLLM results to volume: {e}",
                    file=sys.stderr,
                )

            return {
                "success": True,
                "test_type": test_type,
                "datasets_tested": datasets_tested,
                "results_summary": summary_results,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"TrustLLM evaluation failed: {str(e)}",
                "test_type": test_params.get("test_type", "unknown"),
                "datasets_tested": [],
                "results_summary": {},
            }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TrustLLM test container")
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
        return systems_params, test_params
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in arguments: {e}")


def validate_inputs(systems_params: Dict[str, Any], test_params: Dict[str, Any]):
    """Validate input parameters"""
    # Validate system_under_test params
    sut_params = systems_params.get("system_under_test", {})
    if not sut_params:
        raise ValueError("Missing system_under_test in systems_params")

    required_sut_fields = ["type", "base_url", "model"]
    for field in required_sut_fields:
        if field not in sut_params:
            raise ValueError(f"Missing required system_under_test parameter: {field}")

    if sut_params["type"] != "llm_api":
        raise ValueError(f"Unsupported system_under_test type: {sut_params['type']}")

    # Validate test params
    test_type = test_params.get("test_type")
    if not test_type:
        raise ValueError("test_type is required")

    valid_test_types = [
        "ethics",
        "privacy",
        "fairness",
        "truthfulness",
        "robustness",
        "safety",
    ]
    if test_type.lower() not in valid_test_types:
        raise ValueError(
            f"Invalid test_type: {test_type}. Must be one of: {valid_test_types}"
        )

    # Validate optional numeric parameters
    for param in ["max_new_tokens", "max_rows"]:
        if param in test_params:
            if (
                not isinstance(test_params[param], (int, float))
                or test_params[param] <= 0
            ):
                raise ValueError(f"Parameter '{param}' must be a positive number")


def main():
    try:
        systems_params, test_params = parse_arguments()
        validate_inputs(systems_params, test_params)

        # Pass test_params to capture metadata
        tester = TrustLLMTester(systems_params, test_params)
        result = tester.run_trustllm_evaluation(test_params)

        print(json.dumps(result), flush=True)
        sys.exit(0 if result["success"] else 1)

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "test_type": "unknown",
            "datasets_tested": [],
            "results_summary": {},
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "test_type": "unknown",
            "datasets_tested": [],
            "results_summary": {},
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
