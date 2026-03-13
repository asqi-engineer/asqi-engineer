"""
Integration test for cross-container metric expressions.

This test demonstrates the new feature that allows a single QI indicator
to combine metrics from multiple test containers.
"""

import pytest

from asqi.schemas import (
    AssessmentRule,
    MetricExpression,
    ScoreCard,
    ScoreCardFilter,
    ScoreCardIndicator,
)
from asqi.score_card_engine import ScoreCardEngine
from asqi.workflow import TestExecutionResult


class TestCrossContainerIntegration:
    """Integration tests for cross-container metric expressions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_accuracy_result(
        self, sut_name: str = "claude_model"
    ) -> TestExecutionResult:
        """Create a mock accuracy test result."""
        result = TestExecutionResult(
            "Mock Accuracy Test",
            "mock_accuracy_test",
            sut_name,
            "mock_tester:latest",
        )
        result.test_results = {
            "success": True,
            "score": 0.92,  # 92% accuracy
            "delay_used": 0,
            "base_url": "https://api.anthropic.com",
            "model": sut_name,
        }
        result.success = True
        return result

    def create_robustness_result(
        self, sut_name: str = "claude_model"
    ) -> TestExecutionResult:
        """Create a mock robustness test result."""
        result = TestExecutionResult(
            "Mock Robustness Test",
            "mock_robustness_test",
            sut_name,
            "mock_robustness_tester:latest",
        )
        result.test_results = {
            "success": True,
            "ood_detection_accuracy": 0.78,  # 78% OOD detection
            "adversarial_robustness": 0.72,  # 72% adversarial robustness
            "perturbation_tolerance": 0.85,  # 85% perturbation tolerance
            "model": sut_name,
        }
        result.success = True
        return result

    def test_simple_single_container_backward_compat(self):
        """Test that simple single-container indicators still work."""
        result = self.create_accuracy_result()

        indicator = ScoreCardIndicator(
            id="accuracy_only",
            name="Accuracy Only",
            apply_to=ScoreCardFilter(test_id="mock_accuracy_test"),
            metric="score",
            assessment=[
                AssessmentRule(outcome="A", condition="greater_equal", threshold=0.9),
                AssessmentRule(outcome="B", condition="greater_equal", threshold=0.75),
                AssessmentRule(outcome="C", condition="less_than", threshold=0.75),
            ],
        )

        eval_results = self.engine.evaluate_indicator([result], indicator)
        assert len(eval_results) == 1
        assert eval_results[0].error is None
        assert eval_results[0].metric_value == 0.92
        assert eval_results[0].outcome == "A"

    def test_cross_container_single_apply_to_rejected(self):
        """Test that cross-container references with single apply_to are now rejected at validation."""
        # Cross-container refs (::) are only allowed when apply_to.test_id is a list
        with pytest.raises(
            ValueError,
            match="uses cross-container metric references.*apply_to.test_id is a single string",
        ):
            ScoreCardIndicator(
                id="combined_score",
                name="Combined Score",
                apply_to=ScoreCardFilter(test_id="mock_accuracy_test"),
                metric=MetricExpression(
                    expression="0.6 * acc + 0.4 * robust",
                    values={
                        "acc": "score",
                        "robust": "mock_robustness_test::ood_detection_accuracy",
                    },
                ),
                assessment=[
                    AssessmentRule(
                        outcome="A", condition="greater_equal", threshold=0.85
                    ),
                ],
            )

    def test_multi_container_list_apply_to(self):
        """Test multi-container with list apply_to."""
        accuracy_result = self.create_accuracy_result("claude_model")
        robustness_result = self.create_robustness_result("claude_model")

        indicator = ScoreCardIndicator(
            id="balanced_quality",
            name="Balanced Quality",
            apply_to=ScoreCardFilter(
                test_id=["mock_accuracy_test", "mock_robustness_test"]
            ),
            metric=MetricExpression(
                expression="0.4 * acc + 0.3 * ood + 0.2 * adv + 0.1 * pert",
                values={
                    "acc": "mock_accuracy_test::score",
                    "ood": "mock_robustness_test::ood_detection_accuracy",
                    "adv": "mock_robustness_test::adversarial_robustness",
                    "pert": "mock_robustness_test::perturbation_tolerance",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="Excellent", condition="greater_equal", threshold=0.80
                ),
                AssessmentRule(
                    outcome="Good", condition="greater_equal", threshold=0.65
                ),
                AssessmentRule(outcome="Fair", condition="less_than", threshold=0.65),
            ],
        )

        eval_results = self.engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        assert len(eval_results) == 1
        assert eval_results[0].error is None
        assert eval_results[0].sut_name == "claude_model"

        # 0.4 * 0.92 + 0.3 * 0.78 + 0.2 * 0.72 + 0.1 * 0.85
        # = 0.368 + 0.234 + 0.144 + 0.085 = 0.831
        expected = 0.4 * 0.92 + 0.3 * 0.78 + 0.2 * 0.72 + 0.1 * 0.85
        assert eval_results[0].metric_value == pytest.approx(expected)
        assert eval_results[0].outcome == "Excellent"

    def test_multi_container_multiple_suts(self):
        """Test multi-container evaluation with multiple SUTs."""
        # Two models
        accuracy_claude = self.create_accuracy_result("claude_model")
        robustness_claude = self.create_robustness_result("claude_model")
        accuracy_gpt4 = self.create_accuracy_result("gpt4_model")
        robustness_gpt4 = self.create_robustness_result("gpt4_model")

        indicator = ScoreCardIndicator(
            id="balanced_quality",
            name="Balanced Quality",
            apply_to=ScoreCardFilter(
                test_id=["mock_accuracy_test", "mock_robustness_test"]
            ),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "mock_accuracy_test::score",
                    "ood": "mock_robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="Pass", condition="greater_equal", threshold=0.80
                ),
                AssessmentRule(outcome="Fail", condition="less_than", threshold=0.80),
            ],
        )

        results = [accuracy_claude, robustness_claude, accuracy_gpt4, robustness_gpt4]
        eval_results = self.engine.evaluate_indicator(results, indicator)

        # Should have one result per SUT
        assert len(eval_results) == 2
        sut_names = {r.sut_name for r in eval_results}
        assert sut_names == {"claude_model", "gpt4_model"}

        # Both should have success (no errors)
        for result in eval_results:
            assert result.error is None

    def test_complete_scorecard_evaluation(self):
        """Test evaluating a complete scorecard with mixed indicators."""
        accuracy_result = self.create_accuracy_result()
        robustness_result = self.create_robustness_result()

        scorecard = ScoreCard(
            score_card_name="Demo Scorecard",
            indicators=[
                # Single container
                ScoreCardIndicator(
                    id="accuracy_only",
                    name="Accuracy Only",
                    apply_to=ScoreCardFilter(test_id="mock_accuracy_test"),
                    metric="score",
                    assessment=[
                        AssessmentRule(
                            outcome="Pass", condition="greater_equal", threshold=0.85
                        ),
                        AssessmentRule(
                            outcome="Fail", condition="less_than", threshold=0.85
                        ),
                    ],
                ),
                # Multi-container with list apply_to
                ScoreCardIndicator(
                    id="combined_multi",
                    name="Combined (Multi Apply)",
                    apply_to=ScoreCardFilter(
                        test_id=["mock_accuracy_test", "mock_robustness_test"]
                    ),
                    metric=MetricExpression(
                        expression="0.6 * acc + 0.4 * ood",
                        values={
                            "acc": "mock_accuracy_test::score",
                            "ood": "mock_robustness_test::ood_detection_accuracy",
                        },
                    ),
                    assessment=[
                        AssessmentRule(
                            outcome="Pass", condition="greater_equal", threshold=0.80
                        ),
                        AssessmentRule(
                            outcome="Fail", condition="less_than", threshold=0.80
                        ),
                    ],
                ),
            ],
        )

        eval_results = self.engine.evaluate_scorecard(
            [accuracy_result, robustness_result], scorecard
        )

        # Should have 2 results (one per indicator: accuracy_only + combined_multi)
        assert len(eval_results) == 2

        # All should succeed
        for result in eval_results:
            assert result["error"] is None
            assert result["outcome"] in ["Pass", "Fail"]

    def test_max_function_cross_container(self):
        """Test using max() function with cross-container metrics."""
        accuracy_result = self.create_accuracy_result()
        robustness_result = self.create_robustness_result()

        indicator = ScoreCardIndicator(
            id="max_metric",
            name="Maximum of Metrics",
            apply_to=ScoreCardFilter(
                test_id=["mock_accuracy_test", "mock_robustness_test"]
            ),
            metric=MetricExpression(
                expression="max(acc, ood)",
                values={
                    "acc": "mock_accuracy_test::score",
                    "ood": "mock_robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="Pass", condition="greater_equal", threshold=0.80
                ),
                AssessmentRule(outcome="Fail", condition="less_than", threshold=0.80),
            ],
        )

        eval_results = self.engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        assert len(eval_results) == 1
        assert eval_results[0].error is None
        # max(0.92, 0.78) = 0.92
        assert eval_results[0].metric_value == pytest.approx(0.92)
        assert eval_results[0].outcome == "Pass"

    def test_min_function_cross_container(self):
        """Test using min() function with cross-container metrics."""
        accuracy_result = self.create_accuracy_result()
        robustness_result = self.create_robustness_result()

        indicator = ScoreCardIndicator(
            id="min_metric",
            name="Minimum of Metrics",
            apply_to=ScoreCardFilter(
                test_id=["mock_accuracy_test", "mock_robustness_test"]
            ),
            metric=MetricExpression(
                expression="min(acc, ood)",
                values={
                    "acc": "mock_accuracy_test::score",
                    "ood": "mock_robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="Pass", condition="greater_equal", threshold=0.75
                ),
                AssessmentRule(outcome="Fail", condition="less_than", threshold=0.75),
            ],
        )

        eval_results = self.engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        assert len(eval_results) == 1
        assert eval_results[0].error is None
        # min(0.92, 0.78) = 0.78
        assert eval_results[0].metric_value == pytest.approx(0.78)
        assert eval_results[0].outcome == "Pass"

    def test_cross_reference_test_id_not_in_apply_to_rejected(self):
        """Test that referencing a test_id not in apply_to.test_id is rejected at schema time."""
        # This should fail validation because we reference mock_robustness_test
        # but only list mock_accuracy_test in apply_to.test_id
        with pytest.raises(
            ValueError, match="references test_id\\(s\\).*not in apply_to.test_id"
        ):
            ScoreCardIndicator(
                id="invalid_cross_ref",
                name="Invalid Cross Reference",
                apply_to=ScoreCardFilter(
                    test_id=["mock_accuracy_test"]
                ),  # Missing mock_robustness_test
                metric=MetricExpression(
                    expression="0.6 * acc + 0.4 * ood",
                    values={
                        "acc": "mock_accuracy_test::score",
                        "ood": "mock_robustness_test::ood_detection_accuracy",  # Not in apply_to!
                    },
                ),
                assessment=[
                    AssessmentRule(
                        outcome="Pass", condition="greater_equal", threshold=0.75
                    ),
                    AssessmentRule(
                        outcome="Fail", condition="less_than", threshold=0.75
                    ),
                ],
            )

    def test_multi_container_tracks_all_containers_used(self):
        """Test that multi-container evaluation tracks all containers involved."""
        accuracy_result = self.create_accuracy_result()
        robustness_result = self.create_robustness_result()

        indicator = ScoreCardIndicator(
            id="balanced_quality",
            name="Balanced Quality",
            apply_to=ScoreCardFilter(
                test_id=["mock_accuracy_test", "mock_robustness_test"]
            ),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "mock_accuracy_test::score",
                    "ood": "mock_robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="Pass", condition="greater_equal", threshold=0.75
                ),
                AssessmentRule(outcome="Fail", condition="less_than", threshold=0.75),
            ],
        )

        eval_results = self.engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        # Should have 1 result (one per SUT)
        assert len(eval_results) == 1

        # The result should track both containers that were used
        result = eval_results[0]
        assert result.test_ids == ["mock_accuracy_test", "mock_robustness_test"]
        assert result.outcome == "Pass"  # Verify evaluation still works
