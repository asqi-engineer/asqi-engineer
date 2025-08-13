import argparse
import json
import random
import sys
import time


def main():
    """Main entrypoint that demonstrates the container interface."""
    parser = argparse.ArgumentParser(description="Mock test container")
    parser.add_argument(
        "--sut-config", required=True, help="SUT configuration as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )

    args = parser.parse_args()

    try:
        # Parse inputs
        _sut_config = json.loads(args.sut_config)
        test_params = json.loads(args.test_params)

        # Extract delay parameter
        delay_seconds = test_params.get("delay_seconds", 0)

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Always succeed with a random score
        result = {
            "success": True,
            "score": random.uniform(0.7, 1.0),
            "delay_used": delay_seconds,
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
