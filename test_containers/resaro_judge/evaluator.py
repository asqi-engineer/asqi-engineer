from __future__ import annotations

import math
from typing import Any, Dict, Optional, List, Set
import re

from langchain_core.language_models.chat_models import BaseChatModel
from openevals.llm import create_async_llm_as_judge


class SimpleOrLLMJudge:
    """
    Judge a candidate answer against a gold answer.

    If an LLM judge is provided, we prompt it to score accuracy on a 0..1 scale
    using a robust rubric. Otherwise, we fall back to a simple heuristic that
    checks containment and normalized string equality.
    """

    def __init__(
        self, judge_client: Optional[BaseChatModel] = None, retry_count: int = 1
    ):
        self.judge_client = judge_client
        self.retry_count = retry_count

        if judge_client is not None:
            prompt = (
                "You are an expert judge. Score how well the Candidate Answer matches the Gold Answer for the Question.\n"
                "Return only a number in [0,1] where: 1 = exact and fully correct, 0.5 = partially correct, 0 = incorrect.\n"
                "Consider semantic equivalence, numbers, units, and key facts.\n\n"
                "Question: {question}\n\nGold Answer: {gold}\n\nCandidate Answer: {pred}\n\nScore:"
            )
            self.llm_as_judge = create_async_llm_as_judge(
                prompt=prompt, choices=[0.0, 0.5, 1.0], judge=judge_client
            )
        else:
            self.llm_as_judge = None

    async def score(self, question: str, gold: str, pred: str) -> float:
        """Backward-compatible numeric score API."""
        result = await self.evaluate(question=question, gold=gold, pred=pred)
        return float(result.get("score", 0.0))

    async def evaluate(self, question: str, gold: str, pred: str) -> Dict[str, Any]:
        """Return both score and short reasoning.

        If an LLM judge is configured, it determines the score. We then try to
        extract or generate a brief reasoning. Otherwise, use the heuristic
        scorer with an explanatory note.
        """
        # Prefer LLM judge if available
        if self.llm_as_judge is not None:
            last_score = float("nan")
            reasoning = ""
            for _ in range(self.retry_count):
                try:
                    res = await self.llm_as_judge(
                        question=question, gold=gold, pred=pred, outputs={}
                    )
                    if isinstance(res, dict):
                        s = res.get("score")
                        if isinstance(s, (int, float)):
                            last_score = float(s)
                        elif isinstance(s, str):
                            try:
                                last_score = float(s.strip())
                            except Exception:
                                pass
                        # Try to capture reasoning if available in response
                        for key in ("reason", "reasoning", "explanation", "rationale"):
                            if key in res and isinstance(res[key], str):
                                reasoning = res[key]
                                break
                        if not math.isnan(last_score):
                            break
                    else:
                        # fallback attempt: parse any numeric in string
                        try:
                            last_score = float(str(res).strip())
                            break
                        except Exception:
                            pass
                except Exception:
                    continue

            # If no reasoning returned, optionally ask the judge model for a short explanation
            if not reasoning and self.judge_client is not None:
                try:
                    explain_prompt = (
                        "You are an expert judge. In 1-3 concise sentences, explain how well the Candidate Answer "
                        "matches the Gold Answer for the Question. Focus on key facts, numbers, and correctness.\n\n"
                        f"Question: {question}\n\nGold Answer: {gold}\n\nCandidate Answer: {pred}\n\nReasoning:"
                    )
                    # langchain BaseChatModel supports ainvoke on a string prompt
                    msg = await self.judge_client.ainvoke(explain_prompt)
                    # msg may be a BaseMessage; get content attribute when available
                    reasoning = getattr(msg, "content", str(msg)) or ""
                except Exception:
                    reasoning = ""

            score_val = (
                0.0 if math.isnan(last_score) else max(0.0, min(1.0, last_score))
            )
            return {"score": score_val, "reasoning": reasoning}

        # Heuristic judge: numeric-aware normalization and checks
        gold_norm = normalize_text(gold)
        pred_norm = normalize_text(pred)
        # If both contain numbers and gold's numbers are included in pred, consider correct
        gold_nums = extract_numbers(gold)
        pred_nums = extract_numbers(pred)
        if gold_nums and pred_nums and gold_nums.issubset(pred_nums):
            return {
                "score": 1.0,
                "reasoning": f"Numeric match: gold numbers {sorted(gold_nums)} found in prediction {sorted(pred_nums)}.",
            }
        if not gold_norm and not pred_norm:
            return {
                "score": 1.0,
                "reasoning": "Both gold and prediction are empty after normalization.",
            }
        if not pred_norm:
            return {
                "score": 0.0,
                "reasoning": "Prediction is empty after normalization while gold is not.",
            }
        if gold_norm == pred_norm:
            return {"score": 1.0, "reasoning": "Exact match after normalization."}
        if gold_norm in pred_norm:
            return {
                "score": 1.0,
                "reasoning": "Gold answer is contained within the prediction after normalization.",
            }
        # partial token overlap ratio
        gold_tokens = set(gold_norm.split())
        pred_tokens = set(pred_norm.split())
        overlap = len(gold_tokens & pred_tokens)
        ratio = overlap / max(1, len(gold_tokens))
        return {
            "score": ratio,
            "reasoning": f"Token overlap: {overlap}/{len(gold_tokens)} tokens match after normalization.",
        }


def normalize_text(s: str) -> str:
    # Lowercase, remove punctuation, collapse whitespace
    s = s.lower().strip().replace("\n", " ")
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def extract_numbers(s: str) -> Set[str]:
    """Extract a set of normalized numeric strings from input.

    Handles integers, decimals, and scientific notation in a basic way.
    """
    # Keep original case/format for numeric regex, then normalize by stripping leading zeros in integers
    nums: List[str] = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    normalized: Set[str] = set()
    for n in nums:
        try:
            # Normalize representation: use float for decimals/scientific, int when possible
            if re.match(r"^[-+]?\d+$", n):
                # integer
                normalized.add(str(int(n)))
            else:
                # float/scientific -> round to reasonable precision to avoid 1.0 vs 1 artifacts
                val = float(n)
                normalized.add(
                    str(val).rstrip("0").rstrip(".") if "." in str(val) else str(val)
                )
        except Exception:
            continue
    return normalized
