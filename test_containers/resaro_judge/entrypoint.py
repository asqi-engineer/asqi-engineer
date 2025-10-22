import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List

from evaluator import Judge


def setup_client(**system_params) -> Any:
    """Setup OpenAI-compatible async client with unified env handling."""
    # Lazy import to avoid dev-time import issues if openai isn't installed
    import openai  # type: ignore

    base_url = system_params.get("base_url")
    api_key = system_params.get("api_key")
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key)


def setup_evaluator_client(**system_params) -> Any:
    """Setup an OpenAI-compatible client for the evaluator as well."""
    return setup_client(**system_params)


async def call_model(client: Any, model: str, prompt: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


async def sanity_check_client(client: Any, model: str) -> None:
    """Make a tiny request to validate the client+model works; raise if not."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            temperature=0,
            max_tokens=2,
        )
        _ = resp.choices[0].message.content  # Ensure shape is as expected
    except Exception as e:
        raise RuntimeError(f"Client/model sanity check failed: {e}")


async def run_judge(
    systems_params: Dict[str, Any], test_params: Dict[str, Any]
) -> Dict[str, Any]:
    sut_params = systems_params.get("system_under_test", {})
    if not sut_params:
        raise ValueError("Missing system_under_test in systems_params")

    sut_type = sut_params.get("type")
    if sut_type not in ["llm_api"]:
        raise ValueError(f"Unsupported system_under_test type: {sut_type}")

    test_type = test_params.get("test_type", "accuracy")
    if test_type != "accuracy":
        raise ValueError("Only test_type 'accuracy' is supported")

    dataset: List[Dict[str, str]] = test_params.get("dataset") or []
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("test_params.dataset must be a non-empty list")

    # Thresholding removed; we use the judge's numeric score directly as correctness

    # Setup SUT client
    sut_client = setup_client(**sut_params)
    sut_model = sut_params.get("model", "gpt-4o-mini")
    # Preflight: ensure SUT model is reachable before running dataset
    await sanity_check_client(sut_client, sut_model)

    # Optional evaluator as judge (LLM-based boolean similarity)
    evaluator_system = systems_params.get("evaluator_system", {})
    judge = None
    judge_model = None
    if evaluator_system:
        if isinstance(evaluator_system, list) and evaluator_system:
            judge_model = str(evaluator_system[0])
            # Use default env for credentials
            judge_client = setup_evaluator_client()
        elif isinstance(evaluator_system, dict):
            judge_model = evaluator_system.get("model", "gpt-4o-mini")
            judge_client = setup_evaluator_client(**evaluator_system)
        else:
            judge_model = str(evaluator_system)
            judge_client = setup_evaluator_client()
        # Preflight: ensure evaluator judge model is reachable
        await sanity_check_client(judge_client, judge_model)
        judge = Judge(judge_client, model=judge_model)
    else:
        judge = Judge(None)

    per_question = []
    for item in dataset:
        question = item.get("question") or item.get("prompt") or ""
        gold = item.get("answer") or item.get("gold") or ""
        if not question:
            continue

        # Single evaluation
        pred = await call_model(sut_client, sut_model, question)
        judge_result = await judge.evaluate(question=question, gold=gold, pred=pred)
        judge_match = bool(judge_result.get("match", False))
        judge_reasoning = judge_result.get("reasoning") or ""

        per_question.append(
            {
                "question": question,
                "gold_answer": gold,
                "generated_answer": pred,
                "judge_reasoning": judge_reasoning,
                "correct": judge_match,
            }
        )

    overall = 0.0
    if per_question:
        total_correct = sum(1.0 for q in per_question if q.get("correct"))
        overall = round(total_correct / len(per_question), 3)

    return {
        "success": True,
        "test_type": test_type,
        "per_question": per_question,
        "overall_average_accuracy": overall,
        "judge_type": "llm"
        if judge and getattr(judge, "client", None)
        else "heuristic",
        "judge_model": judge_model
        or (sut_model if judge and getattr(judge, "client", None) else None),
    }


def main():
    parser = argparse.ArgumentParser(description="Resaro Judge Test Container")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON string"
    )
    parser.add_argument(
        "--test-params", required=True, help="Test parameters as JSON string"
    )
    args = parser.parse_args()

    try:
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        result = asyncio.run(run_judge(systems_params, test_params))
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(
            json.dumps({"success": False, "error": f"Unexpected error: {e}"}, indent=2)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
