import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import openai
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from evaluator import SimpleOrLLMJudge


def setup_client(**system_params) -> openai.AsyncOpenAI:
    """Setup OpenAI-compatible async client with unified env handling."""
    base_url = system_params.get("base_url")
    api_key = system_params.get("api_key")
    if base_url and not api_key:
        api_key = os.environ.get("API_KEY")
    if not base_url and not api_key:
        base_url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Provide base_url+api_key or OPENAI_API_KEY."
        )

    openai_params = {
        k: v
        for k, v in system_params.items()
        if k not in ["base_url", "model", "api_key", "type", "env_file"]
    }
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, **openai_params)


def setup_langchain_client(**system_params) -> ChatOpenAI:
    base_url = system_params.get("base_url")
    api_key = system_params.get("api_key")
    model = system_params.get("model", "gpt-4o-mini")
    if base_url and not api_key:
        api_key = os.environ.get("API_KEY")
    if not base_url and not api_key:
        base_url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found for evaluator.")

    extra = {
        k: v
        for k, v in system_params.items()
        if k not in ["base_url", "model", "api_key", "type", "env_file"]
    }
    return ChatOpenAI(
        model=model, api_key=SecretStr(api_key), base_url=base_url, **extra
    )


async def call_model(client: openai.AsyncOpenAI, model: str, prompt: str) -> str:
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

    # Optional evaluator as judge
    evaluator_system = systems_params.get("evaluator_system", {})
    judge = None
    if evaluator_system:
        judge_client = setup_langchain_client(**evaluator_system)
        judge = SimpleOrLLMJudge(judge_client)
    else:
        judge = SimpleOrLLMJudge(None)

    per_question = []
    for item in dataset:
        question = item.get("question") or item.get("prompt") or ""
        gold = item.get("answer") or item.get("gold") or ""
        if not question:
            continue

        # Single evaluation
        pred = await call_model(sut_client, sut_model, question)
        judge_result = await judge.evaluate(question=question, gold=gold, pred=pred)
        judge_score = float(judge_result.get("score", 0.0))
        judge_reasoning = judge_result.get("reasoning") or ""

        per_question.append(
            {
                "question": question,
                "gold_answer": gold,
                "generated_answer": pred,
                "judge_reasoning": judge_reasoning,
                "correct": round(float(judge_score), 3),
            }
        )

    overall = (
        round(sum(q["correct"] for q in per_question) / len(per_question), 3)
        if per_question
        else 0.0
    )

    return {
        "success": True,
        "test_type": test_type,
        "per_question": per_question,
        "overall_average_accuracy": overall,
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
