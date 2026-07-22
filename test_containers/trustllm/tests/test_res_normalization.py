"""
Regression tests for the ``res`` KeyError in privacy_leakage evaluation.

Root cause (see docs/trustllm_res_keyerror_diagnosis.md): the upstream
``PrivacyEval.leakage_eval`` accesses the model response via a hard
``el["res"]`` subscript and only guards the *value* (``el["res"] != None``),
not the *presence* of the key. A generated record missing ``res`` (dropped API
call, thread exception, small-sample edge case) therefore raises
``KeyError: 'res'`` which surfaces as the opaque ``Evaluation failed: 'res'``.

``_evaluate_results`` now normalizes generated data (backfilling ``res=None``)
before invoking any evaluator. These tests lock that in.
"""

import entrypoint
import pytest


def _leakage_eval_like_upstream(data):
    """Mimic the fragile upstream access pattern: hard subscript on ``res``."""
    responses = [el["res"] for el in data if el["res"] is not None]
    return {"RtA": len(responses) / max(len(data), 1)}


class TestNormalizeGeneratedData:
    """The helper must guarantee a ``res`` key on every dict record."""

    def test_backfills_missing_res_with_none(self):
        data = [{"prompt": "a"}, {"prompt": "b", "res": "answer"}]
        entrypoint.TrustLLMTester._normalize_generated_data(data)

        assert data[0]["res"] is None
        assert data[1]["res"] == "answer"  # existing value untouched

    def test_leaves_existing_res_untouched(self):
        data = [{"prompt": "a", "res": None}, {"prompt": "b", "res": "x"}]
        entrypoint.TrustLLMTester._normalize_generated_data(data)

        assert data[0]["res"] is None
        assert data[1]["res"] == "x"

    def test_non_list_input_is_passed_through(self):
        assert entrypoint.TrustLLMTester._normalize_generated_data(None) is None
        assert entrypoint.TrustLLMTester._normalize_generated_data({"res": 1}) == {"res": 1}


class TestPrivacyLeakageNoLongerCrashesOnMissingRes:
    """The exact 'res' KeyError scenario must not abort evaluation anymore."""

    def test_missing_res_would_crash_upstream_pattern(self):
        """Sanity check: the raw fragile pattern really does raise on missing res."""
        raw = [{"prompt": "a"}, {"prompt": "b", "res": "answer"}]
        with pytest.raises(KeyError):
            _leakage_eval_like_upstream(raw)

    def test_evaluate_results_normalizes_before_leakage_eval(self, tester):
        """_evaluate_results must backfill res so leakage_eval succeeds."""
        # Wire the privacy evaluator with an upstream-like leakage_eval.
        tester.evaluators["privacy"].leakage_eval = _leakage_eval_like_upstream

        generated = [{"prompt": "a"}, {"prompt": "b", "res": "answer"}]
        result = tester._evaluate_results("privacy", "privacy_leakage", generated)

        assert "error" not in result
        assert result["evaluation_results"]["RtA"] == pytest.approx(0.5)
