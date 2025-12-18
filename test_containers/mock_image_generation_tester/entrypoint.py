import argparse
import json
import random
import sys
import time


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock image generation test container")
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
        if sut_type not in ["image_generation_api"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        # Extract SUT parameters (flattened structure)
        base_url = sut_params["base_url"]  # Required, validated upstream
        # api_key = sut_params["api_key"]  # Required, validated upstream
        model = sut_params["model"]  # Required, validated upstream

        # Extract test parameters
        delay_seconds = test_params.get("delay_seconds", 0)
        prompt = test_params.get("prompt", "A beautiful sunset over mountains")
        response_format = test_params.get("response_format", "url")

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Create a mock image generation API response to demonstrate validation
        # In a real test container, you would call the actual image generation API:
        #
        # from openai import OpenAI
        # client = OpenAI(base_url=base_url, api_key=api_key)
        # response = client.images.generate(
        #     model=model,
        #     prompt=prompt,
        #     response_format=response_format
        # )
        # mock_response = response.model_dump()

        # Mock image generation response with proper structure
        if response_format == "b64_json":
            # Mock base64-encoded image data
            b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg=="
            image_data = {"b64_json": b64_data}
        else:
            # Mock URL response
            image_data = {
                "url": f"https://example.com/generated_image_{random.randint(1000, 9999)}.png"
            }

        mock_response = {
            "created": int(time.time()),
            "data": [
                {
                    "revised_prompt": f"A stunning and vivid depiction of {prompt}, with exceptional detail and artistic quality.",
                    **image_data,
                }
            ],
            "usage": {
                "input_tokens": len(prompt.split()),
                "output_tokens": 170,
                "total_tokens": len(prompt.split()) + 170,
            },
        }

        # Extract image metrics directly from mock response
        num_images = len(mock_response["data"])
        has_revised_prompt = any(
            img.get("revised_prompt") for img in mock_response["data"]
        )
        response_format_used = (
            "b64_json" if mock_response["data"][0].get("b64_json") else "url"
        )

        result = {
            "success": True,
            "score": random.uniform(0.7, 1.0),
            "delay_used": delay_seconds,
            "base_url": base_url,
            "model": model,
            "validation": "passed",
            "num_images": num_images,
            "response_format": response_format_used,
            "has_revised_prompt": has_revised_prompt,
            "prompt": prompt,
            "usage": mock_response.get("usage", {}),
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
