import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from simulation import (
    ConversationTestAnalyzer,
    PersonaBasedConversationTester,
    setup_client,
)
from utils import get_openai_tracking_kwargs

from asqi.datasets import load_hf_dataset


def create_model_callback(sut_params: Dict[str, Any], test_params: Dict[str, Any]):
    """Create a model callback function for the SUT."""
    client = setup_client(**sut_params)
    model = sut_params.get("model", "gpt-4o-mini")

    # ASQI injects metadata into test_params["metadata"]
    metadata = (test_params or {}).get("metadata", {})

    async def model_callback(input_text: str) -> str:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_text}],
                **get_openai_tracking_kwargs(metadata),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {str(e)}"

    return model_callback


def load_scenarios_from_dataset(
    test_params: Dict[str, Any],
) -> list[Dict[str, str]] | None:
    """
    Load test scenarios from input_datasets if provided.

    Args:
        test_params: Test parameters containing input_datasets configuration

    Returns:
        List of scenario dicts with 'input' and 'expected_output' keys, or None if not provided

    Raises:
        ValueError: If dataset is malformed or missing required columns
    """
    input_datasets = test_params.get("input_datasets")
    if not input_datasets or "test_scenarios" not in input_datasets:
        return None

    # Get INPUT_MOUNT_PATH from environment (set by ASQI workflow)
    input_mount_path = Path(os.environ.get("INPUT_MOUNT_PATH", "/input"))

    # Load dataset using ASQI utility
    dataset_config = input_datasets["test_scenarios"]
    print("Loading test scenarios dataset from input mount...")

    try:
        dataset = load_hf_dataset(dataset_config, input_mount_path=input_mount_path)
    except Exception as e:
        raise ValueError(f"Failed to load test_scenarios dataset: {e}") from e

    # Validate required columns exist
    required_columns = {"input", "expected_output"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}. "
            f"Expected: 'input' (string) and 'expected_output' (string)."
        )

    # Convert dataset rows to scenario list format
    scenarios = []
    for row in dataset:
        scenarios.append(
            {"input": str(row["input"]), "expected_output": str(row["expected_output"])}
        )

    print(f"Loaded {len(scenarios)} scenarios from dataset")
    return scenarios


async def run_chatbot_simulation(
    systems_params: Dict[str, Any], test_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the chatbot simulation test."""
    sut_params = systems_params.get("system_under_test", {})

    chatbot_purpose = test_params.get("chatbot_purpose", "general customer service")
    num_scenarios = test_params.get("num_scenarios", 3)
    max_turns = test_params.get("max_turns", 4)
    custom_personas = test_params.get("custom_personas")
    sycophancy_levels = test_params.get("sycophancy_levels", ["low", "high"])

    # Load scenarios priority: inline custom_scenarios > input_datasets > LLM generation
    custom_scenarios = test_params.get("custom_scenarios")
    if not custom_scenarios:
        custom_scenarios = load_scenarios_from_dataset(test_params)

    simulations_per_scenario = test_params.get("simulations_per_scenario", 1)
    success_threshold = test_params.get("success_threshold", 0.7)
    max_concurrent = test_params.get("max_concurrent", 3)

    conversation_log_filename = Path(
        test_params.get("conversation_log_filename", "conversation_logs.json")
    )
    if len(conversation_log_filename.parts) != 1:
        raise ValueError(
            "Invalid conversation_log_filename: must be a filename only, no directory traversal allowed."
        )

    conversation_log_filepath = (
        Path(os.environ["OUTPUT_MOUNT_PATH"]) / conversation_log_filename
    )
    print(f"Conversation logs will be saved to: {conversation_log_filepath}")

    simulator_system = systems_params.get("simulator_system", {})
    evaluator_system = systems_params.get("evaluator_system", {})

    # Model callback for SUT (uses metadata inside test_params)
    model_callback = create_model_callback(sut_params, test_params)

    # Metadata injected by ASQI workflow
    metadata = (test_params or {}).get("metadata", {})

    tester = PersonaBasedConversationTester(
        model_callback=model_callback,
        chatbot_purpose=chatbot_purpose,
        simulator_client_params=simulator_system,
        evaluator_client_params=evaluator_system,
        max_turns=max_turns,
        custom_personas=custom_personas,
        sycophancy_levels=sycophancy_levels,
        custom_scenarios=custom_scenarios,
        simulations_per_scenario=simulations_per_scenario,
        max_concurrent=max_concurrent,
        metadata=metadata,
    )

    print("Starting Conversational Testing Pipeline...")
    print("Generating personas...")
    personas = await tester.generate_personas()

    if custom_scenarios:
        print("Creating persona-scenario combinations...")
        scenarios = await tester.generate_persona_scenario_combinations(
            personas, custom_scenarios
        )
    else:
        print("Generating LLM-based conversation scenarios...")
        scenarios = await tester.generate_llm_scenarios(personas, num_scenarios)

    print(
        f"Generated {len(scenarios)} total test cases from "
        f"{len(set(s.get('scenario', '') for s in scenarios))} unique scenarios"
    )
    tester.display_test_plan(personas, scenarios)

    test_cases = await tester.simulate_conversations(scenarios)
    print(f"Generated {len(test_cases)} conversation test cases")

    analyzer = ConversationTestAnalyzer(success_threshold=success_threshold)
    try:
        analyzer.save_conversations(test_cases, conversation_log_filepath)
    except (OSError, IOError, PermissionError) as e:
        print(
            f"Warning: Could not save conversation logs to {conversation_log_filepath}: {e}",
            file=sys.stderr,
        )

    analysis_json = analyzer.analyze_results(test_cases)
    print("\n🎉 Conversation testing complete!")
    summary = analysis_json["summary"]

    return {
        "success": True,
        "total_test_cases": summary["total_test_cases"],
        "average_answer_accuracy": summary["average_answer_accuracy"],
        "average_answer_relevance": summary["average_answer_relevance"],
        "answer_accuracy_pass_rate": summary["answer_accuracy_pass_rate"],
        "answer_relevance_pass_rate": summary["answer_relevance_pass_rate"],
        "by_persona": analysis_json["by_persona"],
        "by_scenario": analysis_json["by_scenario"],
        "by_sycophancy": analysis_json["by_sycophancy"],
    }


def main():
    """Main entrypoint that follows ASQI container interface."""
    parser = argparse.ArgumentParser(description="Chatbot Simulator Test Container")
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
            raise ValueError(
                f"Unsupported system_under_test type: {sut_type}. This container only supports llm_api."
            )

        # Validate required SUT parameters
        required_sut_params = ["model"]
        for param in required_sut_params:
            if param not in sut_params:
                raise ValueError(
                    f"Missing required system_under_test parameter: {param}"
                )

        # Validate required test parameters
        if "chatbot_purpose" not in test_params:
            raise ValueError("Missing required test parameter: chatbot_purpose")

        # Run the simulation
        result = asyncio.run(run_chatbot_simulation(systems_params, test_params))
        print(json.dumps(result, indent=2))
        sys.exit(0)

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "total_test_cases": 0,
            "average_answer_accuracy": 0.0,
            "average_answer_relevance": 0.0,
            "answer_accuracy_pass_rate": 0.0,
            "answer_relevance_pass_rate": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "total_test_cases": 0,
            "average_answer_accuracy": 0.0,
            "average_answer_relevance": 0.0,
            "answer_accuracy_pass_rate": 0.0,
            "answer_relevance_pass_rate": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def demo_app():
    """Demo app function for testing."""


if __name__ == "__main__":
    main()
