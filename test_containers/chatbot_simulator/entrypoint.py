import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict

import openai

from simulation import ConversationTestAnalyzer, PersonaBasedConversationTester


def setup_client(sut_params: Dict[str, Any]) -> openai.AsyncOpenAI:
    """Setup OpenAI client using provided SUT parameters"""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        # Fallback to provider-specific keys
        api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("HUGGINGFACE_API_KEY")
        )

    if not api_key:
        raise ValueError("No API key found. Set API_KEY environment variable.")

    return openai.AsyncOpenAI(
        base_url=sut_params.get("base_url", "https://api.openai.com/v1"),
        api_key=api_key,
    )


async def create_model_callback(sut_params: Dict[str, Any]):
    """Create a model callback function for the SUT"""

    client = setup_client(sut_params)

    async def model_callback(input_text: str) -> str:
        """Model callback that implements the chatbot logic"""
        try:
            response = await client.chat.completions.create(
                model=sut_params["model"],
                messages=[{"role": "user", "content": input_text}],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Error: {str(e)}"

    return model_callback


async def run_chatbot_simulation(
    sut_params: Dict[str, Any], test_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the chatbot simulation test"""

    # Extract test parameters with defaults
    chatbot_purpose = test_params.get("chatbot_purpose", "general customer service")
    num_scenarios = test_params.get("num_scenarios", 3)
    max_turns = test_params.get("max_turns", 4)
    simulator_model = test_params.get("simulator_model", "gpt-4o-mini")
    evaluation_model = test_params.get("evaluation_model", "gpt-4o-mini")
    custom_personas = test_params.get("custom_personas")
    sycophancy_levels = test_params.get("sycophancy_levels", ["low", "high"])
    custom_scenarios = test_params.get("custom_scenarios")
    simulations_per_scenario = test_params.get("simulations_per_scenario", 1)
    success_threshold = test_params.get("success_threshold", 0.7)

    # Create model callback
    model_callback = await create_model_callback(sut_params)

    # Create tester with configured parameters
    tester = PersonaBasedConversationTester(
        model_callback=model_callback,
        chatbot_purpose=chatbot_purpose,
        simulator_model=simulator_model,
        max_turns=max_turns,
        custom_personas=custom_personas,
        sycophancy_levels=sycophancy_levels,
        custom_scenarios=custom_scenarios,
        evaluation_model=evaluation_model,
        simulations_per_scenario=simulations_per_scenario,
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
        f"Generated {len(scenarios)} total test cases from {len(set(s.get('scenario', '') for s in scenarios))} unique scenarios"
    )
    tester.display_test_plan(personas, scenarios)
    print("\nRunning conversational simulations...")
    test_cases = await tester.simulate_conversations(scenarios)

    print(f"Generated {len(test_cases)} conversation test cases")

    analyzer = ConversationTestAnalyzer(success_threshold=success_threshold)
    conversations_file = "conversation_logs.json"
    analyzer.save_conversations(test_cases, conversations_file)

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
        "by_intent": analysis_json["by_intent"],
        "by_sycophancy": analysis_json["by_sycophancy"],
    }


def main():
    """Main entrypoint that follows ASQI container interface."""
    parser = argparse.ArgumentParser(description="Chatbot Simulator Test Container")
    parser.add_argument(
        "--sut-params", required=True, help="SUT parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        sut_params = json.loads(args.sut_params)
        test_params = json.loads(args.test_params)

        # Validate SUT type
        sut_type = sut_params.get("type")
        if sut_type not in ["llm_api"]:
            raise ValueError(
                f"Unsupported SUT type: {sut_type}. This container only supports llm_api."
            )

        # Validate required SUT parameters
        required_sut_params = ["model"]
        for param in required_sut_params:
            if param not in sut_params:
                raise ValueError(f"Missing required SUT parameter: {param}")

        # Validate required test parameters
        if "chatbot_purpose" not in test_params:
            raise ValueError("Missing required test parameter: chatbot_purpose")

        # Run the simulation
        result = asyncio.run(run_chatbot_simulation(sut_params, test_params))

        # Output results as JSON
        print(json.dumps(result, indent=2))

        # Exit with success code
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


if __name__ == "__main__":
    main()
