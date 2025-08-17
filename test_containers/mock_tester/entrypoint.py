import argparse
import json
import random
import sys
import time


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock test container")
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
            raise ValueError(f"Unsupported SUT type: {sut_type}")

        # Extract SUT parameters (flattened structure)
        base_url = sut_params["base_url"]  # Required, validated upstream
        model = sut_params["model"]  # Required, validated upstream

        # Extract delay parameter
        delay_seconds = test_params.get("delay_seconds", 0)

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Simulate LLM API call (this is a mock, so we don't actually call the API)
        # In a real test container, you would use:
        # import openai
        # client = openai.OpenAI(base_url=base_url, api_key=os.getenv("API_KEY"))
        # response = client.chat.completions.create(model=model, messages=[...])

        # Always succeed with a random score
        result = {
            "success": True,
            "score": random.uniform(0.7, 1.0),
            "delay_used": delay_seconds,
            "base_url": base_url,
            "model": model,
        }

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
