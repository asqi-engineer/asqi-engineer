import argparse
import json
import random
import sys
import time


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock rag test container")
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
        if sut_type not in ["rag_api"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        # Extract SUT parameters (flattened structure)
        base_url = sut_params["base_url"]  # Required, validated upstream
        api_key = sut_params["api_key"]  # Required, validated upstream
        model = sut_params["model"]  # Required, validated upstream
        user_group = sut_params.get("user_group")  # Optional, validated upstream

        extra_body = {} # OpenAI-compatible API extra body parameters
        if user_group is not None:
            extra_body = {'user_group': user_group}

        # Extract delay parameter
        delay_seconds = test_params.get("delay_seconds", 0)

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Simulate RAG API call (this is a mock, so we don't actually call the API)
        # In a real test container, you would use:
        # import openai
        # messages = [{"role": "user", "content": "Hello, world!"}]
        # client = openai.OpenAI(base_url=base_url, api_key=api_key)
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     extra_body=extra_body
        # )

        # The response variable would contain the actual API response and would need to contain follow the required response format.
        # content, context = response.choices[0].message.content, response.choices[0].message.context

        # # Validate context format, specific to RAG systems
        # assert isinstance(context, dict), "Context must be a dictionary"
        # assert "citations" in context, "Context must contain 'citations' key"
        # assert isinstance(context["citations"], list), "'citations' must be a list"
        # for citation in context["citations"]:
        #     assert isinstance(citation, dict), "Each citation must be a dictionary"
        #     assert "retrieved_context" in citation, "Citation must contain 'retrieved_context'"
        #     assert isinstance(citation["retrieved_context"], str), "'retrieved_context' must be a string"
        #     assert "document_id" in citation, "Citation must contain 'document_id'"
        #     assert isinstance(citation["document_id"], str), "'document_id' must be a string"
        #     # Optional fields
        #     if "score" in citation:
        #         assert isinstance(citation["score"], (float)), "'score' must be of type float"
        #     if "source_id" in citation:
        #         assert isinstance(citation["source_id"], str), "'source_id' must be a string"

        # Always succeed with a random score
        result = {
            "success": True,
            "score": random.uniform(0.7, 1.0),
            "delay_used": delay_seconds,
            "base_url": base_url,
            "model": model,
        }
        if user_group is not None:
            result["user_group"] = user_group

        # Output results as JSON
        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        sys.exit(0 if result["success"] else 1)

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"Invalid JSON in arguments: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
