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
        # api_key = sut_params["api_key"]  # Required, validated upstream
        model = sut_params["model"]  # Required, validated upstream

        user_group = test_params.get("user_group")  # Optional

        # Extract delay parameter
        delay_seconds = test_params.get("delay_seconds", 0)

        # Simulate work
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        # Create a mock RAG API response to demonstrate validation
        # In a real test container, you would call the actual RAG API:
        #
        # from openai import OpenAI
        # client = OpenAI(base_url=base_url, api_key=api_key)
        # extra_body = {"user_group": user_group} if user_group else {}
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[{"role": "user", "content": "What is the refund policy?"}],
        #     extra_body=extra_body
        # )
        # mock_response = response.model_dump()

        # Mock RAG response with proper structure
        mock_response = {
            "id": f"chatcmpl-{random.randint(1000, 9999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "We offer 30-day returns at no additional cost for all customers.",
                        "context": {
                            "citations": [
                                {
                                    "retrieved_context": "All customers are eligible for a 30-day full refund at no extra cost.",
                                    "document_id": "return_policy.pdf",
                                    "score": round(random.uniform(0.85, 0.98), 3),
                                    "source_id": "company_policies",
                                },
                                {
                                    "retrieved_context": "Receipt required for 30-day refund processing.",
                                    "document_id": "return_policy.pdf",
                                    "score": round(random.uniform(0.70, 0.84), 3),
                                    "source_id": "company_policies",
                                },
                            ]
                        },
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }

        # Validate RAG response format using ASQI validation schema
        try:
            citations = validate_rag_response(mock_response)

            # Validation succeeded - extract citation metrics
            num_citations = len(citations)
            avg_score = (
                sum(c.score for c in citations if c.score) / num_citations
                if num_citations > 0
                else 0.0
            )
            unique_docs = len(set(c.document_id for c in citations))

            result = {
                "success": True,
                "score": random.uniform(0.7, 1.0),
                "delay_used": delay_seconds,
                "base_url": base_url,
                "model": model,
                "validation": "passed",
                "num_citations": num_citations,
                "avg_citation_score": round(avg_score, 3),
                "unique_documents": unique_docs,
                "citations": [
                    {
                        "document_id": c.document_id,
                        "score": c.score,
                        "context_length": len(c.retrieved_context),
                    }
                    for c in citations
                ],
            }
            if user_group is not None:
                result["user_group"] = user_group

        except Exception as e:
            # Validation failed - report error
            result = {
                "success": False,
                "score": 0.0,
                "error": f"RAG response validation failed: {str(e)}",
                "validation": "failed",
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
            "user_group": user_group,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0,
            "user_group": user_group,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
