from __future__ import annotations

import json
import math
import re
import string
from typing import Any, Dict, Optional


class Judge:
    """
    Judge whether a candidate answer is reasonably similar to a gold answer.

    If an OpenAI-compatible client is provided, we ask it with a simple system
    prompt to return a JSON decision: {"match": true/false, "reasoning": "..."}.
    Otherwise, we use a lightweight heuristic on strings and numbers.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        judge_client: Optional[Any] = None,
        *,
        model: Optional[str] = None,
        retry_count: int = 1,
    ) -> None:
        self.client = judge_client
        self.model = model or self.DEFAULT_MODEL
        self.retry_count = retry_count

        self.system_prompt = (
            "You are a careful evaluator. Decide if the Candidate Answer is reasonably similar to the Gold Answer for the Question.\n"
            "Consider semantic equivalence, numbers (with small rounding differences allowed), units, and key facts.\n"
            'Respond ONLY with valid JSON like: {"match": true|false, "reasoning": "1-2 short sentences"}.'
        )

    # ------------- Heuristic helpers -------------
    @staticmethod
    def _normalize(text: str) -> str:
        if text is None:
            return ""
        text = text.strip().lower()
        # Keep digits and basic punctuation relevant for numbers, drop others
        allowed = set(string.ascii_lowercase + string.digits + " .,-+/:%")
        text = "".join(ch if ch in allowed else " " for ch in text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(Judge._normalize(text).split())

    @staticmethod
    def _numbers(text: str) -> list[float]:
        nums = re.findall(r"-?\d+\.\d+|-?\d+", text or "")
        out = []
        for n in nums:
            try:
                out.append(float(n))
            except Exception:
                pass
        return out

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @classmethod
    def _heuristic_match(cls, gold: str, pred: str) -> bool:
        g_norm = cls._normalize(gold)
        p_norm = cls._normalize(pred)

        if not g_norm and not p_norm:
            return True
        if not g_norm or not p_norm:
            return False

        # Direct equality or containment
        if g_norm == p_norm:
            return True
        if g_norm in p_norm or p_norm in g_norm:
            return True

        # Numbers: if gold has numbers, ensure pred has close matches
        g_nums = cls._numbers(gold)
        p_nums = cls._numbers(pred)
        if g_nums:
            matched = 0
            for gx in g_nums:
                if any(
                    math.isclose(gx, py, rel_tol=0.02, abs_tol=1e-6) for py in p_nums
                ):
                    matched += 1
            if matched / len(g_nums) >= 0.8:  # allow minor misses
                return True

        # Token overlap
        g_tok = cls._tokenize(gold)
        p_tok = cls._tokenize(pred)
        j = cls._jaccard(g_tok, p_tok)
        # Stricter threshold for short answers
        if len(g_tok) <= 8 or len(p_tok) <= 8:
            if j >= 0.6:
                return True
        else:
            if j >= 0.45:
                return True

        return False

    # ------------- Public API -------------
    async def evaluate(self, question: str, gold: str, pred: str) -> Dict[str, Any]:
        # Prefer LLM judgement if client is configured
        if self.client is not None:
            for _ in range(self.retry_count):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {question}\n\n"
                                    f"Gold Answer: {gold}\n\n"
                                    f"Candidate Answer: {pred}"
                                ),
                            },
                        ],
                        temperature=0,
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    # Try strict JSON parse
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "match" in data:
                            return {
                                "match": bool(data.get("match", False)),
                                "reasoning": str(data.get("reasoning", "")),
                            }
                    except json.JSONDecodeError:
                        pass

                    # Fallback: extract boolean-like token from text
                    lowered = content.lower()
                    match = None
                    if '"match"' in lowered:
                        # naive extraction
                        m = re.search(r'"match"\s*:\s*(true|false)', lowered)
                        if m:
                            match = m.group(1) == "true"
                    if match is None:
                        if "match":
                            if re.search(r"\btrue\b", lowered):
                                match = True
                            elif re.search(r"\bfalse\b", lowered):
                                match = False
                    if match is not None:
                        # Try to grab reasoning after a reasoning key or after a newline
                        reason = ""
                        m2 = re.search(r'"reasoning"\s*:\s*"(.*?)"', content, re.S)
                        if m2:
                            reason = m2.group(1).strip()
                        else:
                            parts = content.split("\n", 1)
                            if len(parts) > 1:
                                reason = parts[1].strip()
                        return {"match": match, "reasoning": reason}
                except Exception:
                    # On any error, try heuristic below
                    break

        # Heuristic fallback
        match = self._heuristic_match(gold, pred)
        reason = (
            "Heuristic match: numbers and token overlap suggest similarity."
            if match
            else "Heuristic mismatch: insufficient token/number similarity."
        )
        return {"match": match, "reasoning": reason}
