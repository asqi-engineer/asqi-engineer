import argparse
import json
import random
import sys
import time


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock VLM evaluator test container")
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
        if sut_type not in ["vlm_api"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        # Extract SUT parameters (flattened structure)
        base_url = sut_params["base_url"]  # Required, validated upstream

        model = sut_params["model"]  # Required, validated upstream
        supports_vision = sut_params.get(
            "supports_vision", True
        )  # Should always be True for VLM systems

        # Validate that VLM system supports vision
        if not supports_vision:
            raise ValueError(
                "VLM system must support vision, but supports_vision is False"
            )

        # Extract delay parameter
        delay_seconds = test_params.get("delay_seconds", 0)

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Create a mock VLM chat completion response
        # In a real test container, you would call the actual VLM API:
        #
        # from openai import OpenAI
        # client = OpenAI(base_url=base_url, api_key=api_key)
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": evaluation_prompt},
        #                 {"type": "image_url", "image_url": {"url": image_url}}
        #             ]
        #         }
        #     ]
        # )
        # mock_response = response.model_dump()

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
