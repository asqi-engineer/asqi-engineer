import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List
import openai

from evaluator import Judge


def setup_client(**system_params) -> Any:
    """Setup OpenAI-compatible async client with unified env handling."""
    # Lazy import to avoid dev-time import issues if openai isn't installed

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


async def run_judge(
    systems_params: Dict[str, Any], test_params: Dict[str, Any]
) -> Dict[str, Any]:
    sut_params = systems_params.get("system_under_test", {})
    if not sut_params:
        raise ValueError("Missing system_under_test in systems_params")

    sut_type = sut_params.get("type") or "llm_api"
    if sut_type not in ["llm_api"]:
        raise ValueError(f"Unsupported system_under_test type: {sut_type}")

    test_type = test_params.get("test_type", "facts")

    dataset: List[Dict[str, str]] = test_params.get("dataset") or []
    if not isinstance(dataset, list):
        raise ValueError("test_params.dataset must be a list")

    if not dataset:
        raise ValueError("test_params.dataset must be defined")

    # Setup SUT client
    sut_client = setup_client(**sut_params)
    sut_model = sut_params.get("model")
    # Preflight: ensure SUT model is reachable before running dataset

    # Optional evaluator as judge (LLM-based)
    evaluator_system = systems_params.get("evaluator_system", {})
    if not evaluator_system:
        # Also support nested under `systems` per config example
        evaluator_system = (systems_params.get("systems") or {}).get(
            "evaluator_system", {}
        )
    judge = None
    judge_model = None
    if evaluator_system:
        if isinstance(evaluator_system, list) and evaluator_system:
            judge_model = str(evaluator_system[0])
            # Use default env for credentials
            judge_client = setup_evaluator_client()
        elif isinstance(evaluator_system, dict):
            judge_model = evaluator_system.get("model")
            judge_client = setup_evaluator_client(**evaluator_system)
        else:
            judge_model = str(evaluator_system)
            judge_client = setup_evaluator_client()
        # Preflight: ensure evaluator judge model is reachable
        judge = Judge(judge_client, model=judge_model)
    else:
        judge = Judge(None)

    if test_type == "facts":
        per_question = []
        for item in dataset:
            question = item.get("question")
            gold = item.get("answer")
            if not question or not gold:
                continue

            # Single evaluation
            pred = await call_model(sut_client, sut_model, question)
            judge_result = await judge.evaluate(question=question, gold=gold, pred=pred)
            judge_match = bool(judge_result.get("match", False))
            judge_reasoning = judge_result.get("reasoning") or ""

            per_question.append(
                {
                    "question": question,
                    "reference": gold,
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

    elif test_type == "summary":
        if len(dataset) != 1:
            raise ValueError("summary test requires dataset with exactly one item")
        item = dataset[0]
        prompt = item.get("prompt") or item.get("input") or ""
        gold = item.get("reference") or item.get("gold_answer") or ""
        if not prompt:
            raise ValueError("summary test dataset item requires 'prompt'")

        generated = await call_model(sut_client, sut_model, prompt)
        metrics = await judge.evaluate_summary(prompt=prompt, gold=gold, pred=generated)

        return {
            "success": True,
            "test_type": test_type,
            "dataset_item": item,
            "generated_summary": generated,
            "metrics": metrics,
            "judge_type": "llm"
            if judge and getattr(judge, "client", None)
            else "heuristic",
            "judge_model": judge_model
            or (sut_model if judge and getattr(judge, "client", None) else None),
        }
    else:
        raise ValueError("Unsupported test_type. Supported types: 'facts', 'summary'")


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
