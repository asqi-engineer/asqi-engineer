import argparse
import json
import sys
import time
import random
from rag_response_schema import validate_rag_response


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

    user_group = None

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

        user_group = test_params.get("user_group") # Optional

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
        #     extra_body=extra_body,
        #     temperature=0.0,
        #     max_tokens=500
        # )

        # Validate RAG Response format
        # citations = validate_rag_response(response.model_dump())

        # print(f"Received {len(citations)} citations from RAG response.")
        # print(f"Citations: {[c.model_dump() for c in citations]}")
    
        # result = {
        #     "success": True,
        #     "score": random.uniform(0.7, 1.0),
        #     "delay_used": delay_seconds,
        #     "base_url": base_url,
        #     "model": model
        # }
        # if user_group is not None:
        #     result["user_group"] = user_group
        #
        
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
        if user_group is not None:
            error_result["user_group"] = user_group
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0,
        }
        if user_group is not None:
            error_result["user_group"] = user_group
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
