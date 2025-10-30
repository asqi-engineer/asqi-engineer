from __future__ import annotations

import json
import logging
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
            out.append(float(n))
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

    # ------------- Summary evaluation -------------
    async def evaluate_summary(
        self, prompt: str, gold: str, pred: str
    ) -> Dict[str, Any]:
        """
        Evaluate a generated summary along 5 dimensions:
        - faithfulness (accuracy)
        - coverage
        - conciseness
        - fluency
        - overall_quality

        Returns numeric scores between 0.0 and 1.0, plus optional reasoning when judged by LLM.
        """
        # If an evaluator LLM is available, ask it for structured JSON scores
        if self.client is not None:
            system = (
                "You are a careful summarization evaluator. Rate the Candidate Summary on 5 metrics "
                "relative to the Gold Summary/Reference (if provided) and the Prompt/Task. "
                "Each score must be between 0.0 and 1.0. Respond ONLY with JSON of the form: "
                '{"faithfulness": <0..1>, "coverage": <0..1>, "conciseness": <0..1>, "fluency": <0..1>, "overall_quality": <0..1>, "reasoning": "..."}'
            )
            user = (
                f"Prompt/Task: {prompt}\n\n"
                f"Gold Summary (may be empty if not provided): {gold}\n\n"
                f"Candidate Summary: {pred}\n\n"
                "Instructions:\n"
                "- Faithfulness: Are statements supported by the source intent and not hallucinated vs. gold?\n"
                "- Coverage: How well are key points from the gold captured?\n"
                "- Conciseness: Is it succinct without losing meaning?\n"
                "- Fluency: Grammar, readability, coherence.\n"
                "- Overall quality: holistic judgment of usefulness and quality.\n"
                "Return only valid JSON with 5 scores and a short reasoning."
            )
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                )
                content = (resp.choices[0].message.content or "").strip()
                data: Dict[str, Any] = {}
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Soft-extract with regex if needed
                    def _grab(name: str) -> Optional[float]:
                        m = re.search(rf'"{name}"\s*:\s*([0-9]*\.?[0-9]+)', content)
                        try:
                            return float(m.group(1)) if m else None
                        except Exception:
                            return None

                    data = {
                        "faithfulness": _grab("faithfulness"),
                        "coverage": _grab("coverage"),
                        "conciseness": _grab("conciseness"),
                        "fluency": _grab("fluency"),
                        "overall_quality": _grab("overall_quality"),
                        "reasoning": "",
                    }

                def _clamp(x: Any) -> float:
                    try:
                        v = float(x)
                        if v < 0.0:
                            return 0.0
                        if v > 1.0:
                            return 1.0
                        return v
                    except Exception:
                        return 0.0

                out = {
                    "faithfulness": _clamp(data.get("faithfulness")),
                    "coverage": _clamp(data.get("coverage")),
                    "conciseness": _clamp(data.get("conciseness")),
                    "fluency": _clamp(data.get("fluency")),
                    "overall_quality": _clamp(data.get("overall_quality")),
                    "reasoning": str(data.get("reasoning", "")),
                }
                return out
            except Exception as e:
                logging.warning(
                    "LLM evaluation failed, falling back to heuristics", exc_info=e
                )
                pass

        # Heuristic fallback if no evaluator or LLM failed
        def _safe_tokens(s: str) -> list[str]:
            return [t for t in self._normalize(s).split() if t]

        gold_toks = set(_safe_tokens(gold))
        pred_toks = set(_safe_tokens(pred))

        # Faithfulness heuristic: penalize contradictions via simple negation mismatch and hallucinated named entities (very rough)
        # For simplicity, approximate faithfulness by token overlap weighted by presence of numbers
        jacc = (
            self._jaccard(gold_toks, pred_toks) if gold_toks else 0.5
        )  # neutral if no gold
        g_nums = self._numbers(gold)
        p_nums = self._numbers(pred)
        num_score = 1.0
        if g_nums:
            matched = 0
            for gx in g_nums:
                if any(
                    math.isclose(gx, py, rel_tol=0.02, abs_tol=1e-6) for py in p_nums
                ):
                    matched += 1
            num_score = (matched / len(g_nums)) if g_nums else 1.0
        faithfulness = 0.7 * jacc + 0.3 * num_score

        # Coverage heuristic: how many gold tokens are present in pred
        coverage = (
            1.0
            if not gold_toks
            else (len(gold_toks & pred_toks) / max(1, len(gold_toks)))
        )

        # Conciseness heuristic: ratio of pred length to gold length (closer to 1, but penalize being too long)
        glen = max(1, len(_safe_tokens(gold)))
        plen = max(1, len(_safe_tokens(pred)))
        ratio = plen / glen
        if ratio <= 1:
            conciseness = 1.0 - 0.1 * (1 - ratio)  # small penalty when shorter
        else:
            conciseness = max(
                0.0, 1.0 - 0.3 * (ratio - 1)
            )  # stronger penalty when longer

        # Fluency heuristic: punctuation presence and average sentence length bounds
        sentences = re.split(r"[.!?]+\s+", pred.strip()) if pred.strip() else []
        sentences = [s for s in sentences if s]
        avg_len = (
            (sum(len(s.split()) for s in sentences) / len(sentences))
            if sentences
            else plen
        )
        # Reward moderate sentence length and presence of punctuation
        punct_bonus = 0.1 if re.search(r"[.!?]", pred) else 0.0
        if avg_len <= 30:
            fluency = min(1.0, 0.8 + punct_bonus)
        else:
            fluency = max(0.0, 1.0 - (avg_len - 30) * 0.02 + punct_bonus)

        overall = max(
            0.0,
            min(
                1.0,
                0.25 * faithfulness
                + 0.25 * coverage
                + 0.2 * conciseness
                + 0.3 * fluency,
            ),
        )

        return {
            "faithfulness": round(faithfulness, 3),
            "coverage": round(coverage, 3),
            "conciseness": round(conciseness, 3),
            "fluency": round(fluency, 3),
            "overall_quality": round(overall, 3),
            "reasoning": "Heuristic evaluation based on token overlap, numbers, length ratio, and simple fluency cues.",
        }
