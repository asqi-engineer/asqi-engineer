import argparse
import json
import sys
import time
import random
from image_response_schema import validate_image_response


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock image editing test container")
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
        if sut_type not in ["image_editing_api"]:
            raise ValueError(f"Unsupported system_under_test type: {sut_type}")

        # Extract SUT parameters (flattened structure)
        base_url = sut_params["base_url"]  # Required, validated upstream
        # api_key = sut_params["api_key"]  # Required, validated upstream
        model = sut_params["model"]  # Required, validated upstream

        # Extract test parameters
        delay_seconds = test_params.get("delay_seconds", 0)
        prompt = test_params.get("prompt", "Add a sunset sky in the background")
        response_format = test_params.get("response_format", "url")
        mask_mode = test_params.get("mask_mode", "rectangle")

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Create a mock image editing API response to demonstrate validation
        # In a real test container, you would call the actual image editing API:
        #
        # from openai import OpenAI
        # client = OpenAI(base_url=base_url, api_key=api_key)
        # response = client.images.edit(
        #     model=model,
        #     image=open("original.png", "rb"),
        #     mask=open("mask.png", "rb") if mask_mode != "none" else None,
        #     prompt=prompt,
        #     response_format=response_format
        # )
        # mock_response = response.model_dump()

        # Mock image editing response with proper structure
        if response_format == "b64_json":
            # Mock base64-encoded edited image data
            b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg=="
            image_data = {"b64_json": b64_data}
        else:
            # Mock URL response
            image_data = {
                "url": f"https://example.com/edited_image_{random.randint(1000, 9999)}.png"
            }

        mock_response = {
            "created": int(time.time()),
            "data": [
                {
                    "revised_prompt": f"Professional image editing: {prompt}, applied with expert precision and natural blending.",
                    **image_data,
                }
            ],
            "usage": {
                "input_tokens": len(prompt.split())
                + 50,  # Extra tokens for image processing
                "output_tokens": 170,
                "total_tokens": len(prompt.split()) + 220,
            },
        }

        # Validate image editing response format using ASQI validation schema
        try:
            validated_response = validate_image_response(mock_response)

            # Validation succeeded - extract editing metrics
            num_images = len(validated_response.data)
            has_revised_prompt = any(
                img.revised_prompt for img in validated_response.data
            )
            response_format_used = (
                "b64_json" if validated_response.data[0].b64_json else "url"
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
                "mask_mode": mask_mode,
                "has_revised_prompt": has_revised_prompt,
                "prompt": prompt,
                "usage": {
                    "input_tokens": validated_response.usage.input_tokens
                    if validated_response.usage
                    else None,
                    "output_tokens": validated_response.usage.output_tokens
                    if validated_response.usage
                    else None,
                    "total_tokens": validated_response.usage.total_tokens
                    if validated_response.usage
                    else None,
                },
            }

        except Exception as e:
            # Validation failed - report error
            result = {
                "success": False,
                "score": 0.0,
                "error": f"Image editing response validation failed: {str(e)}",
                "validation": "failed",
                "base_url": base_url,
                "model": model,
                "prompt": prompt,
                "mask_mode": mask_mode,
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
