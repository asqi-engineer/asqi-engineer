from asqi.schemas import AssessmentRule, ScoreCard, ScoreCardFilter, ScoreCardIndicator
from asqi.score_card_engine import ScoreCardEngine
from asqi.workflow import TestExecutionResult


class TestscorecardEngine:
    """Test the ScoreCardEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self, test_name: str, image: str, test_results: dict
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, "test_sut", image)
        result.test_results = test_results
        result.success = test_results.get("success", True)
        return result

    def test_filter_results_by_test_name(self):
        """Test filtering test results by test name."""
        results = [
            self.create_test_result("test1", "image1", {"success": True}),
            self.create_test_result("test2", "image2", {"success": True}),
            self.create_test_result("test1", "image2", {"success": False}),
        ]

        filtered = self.engine.filter_results_by_test_name(results, "test1")

        assert len(filtered) == 2
        assert filtered[0].test_name == "test1"
        assert filtered[1].test_name == "test1"

    def test_extract_metric_values(self):
        """Test extracting metric values from test results."""
        results = [
            self.create_test_result("test1", "image1", {"success": True, "score": 0.9}),
            self.create_test_result(
                "test2", "image1", {"success": False, "score": 0.5}
            ),
            self.create_test_result("test3", "image1", {"success": True, "score": 0.8}),
        ]

        # Test extracting boolean values
        success_values = self.engine.extract_metric_values(results, "success")
        assert success_values == [True, False, True]

        # Test extracting numeric values
        score_values = self.engine.extract_metric_values(results, "score")
        assert score_values == [0.9, 0.5, 0.8]

    def test_apply_condition_to_value_equal_to_boolean(self):
        """Test the equal_to condition with boolean values."""
        result, description = self.engine.apply_condition_to_value(
            True, "equal_to", True
        )
        assert result is True
        assert "Value True equals True: True" in description

        result, description = self.engine.apply_condition_to_value(
            False, "equal_to", True
        )
        assert result is False
        assert "Value False equals True: False" in description

    def test_apply_condition_to_value_greater_equal(self):
        """Test the greater_equal condition with numeric values."""
        result, description = self.engine.apply_condition_to_value(
            0.9, "greater_equal", 0.8
        )
        assert result is True
        assert "Value 0.9 greater_equal 0.8: True" in description

        result, description = self.engine.apply_condition_to_value(
            0.7, "greater_equal", 0.8
        )
        assert result is False
        assert "Value 0.7 greater_equal 0.8: False" in description

    def test_apply_condition_to_value_less_equal(self):
        """Test the less_equal condition with integer values."""
        result, description = self.engine.apply_condition_to_value(2, "less_equal", 2)
        assert result is True
        assert "Value 2.0 less_equal 2.0: True" in description

        result, description = self.engine.apply_condition_to_value(3, "less_equal", 2)
        assert result is False
        assert "Value 3.0 less_equal 2.0: False" in description

    def test_evaluate_indicator_success(self):
        """Test successful indicator evaluation."""
        # Create test results
        results = [
            self.create_test_result(
                "test1",
                "image1",
                {"success": True, "score": 0.9},
            ),
            self.create_test_result(
                "test1",
                "image2",
                {"success": True, "score": 0.8},
            ),
            self.create_test_result(
                "test2", "image1", {"success": False, "score": 0.3}
            ),
        ]

        # Create indicator
        indicator = ScoreCardIndicator(
            name="Test individual success",
            apply_to=ScoreCardFilter(test_name="test1"),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True),
                AssessmentRule(outcome="FAIL", condition="equal_to", threshold=False),
            ],
        )

        evaluation_results = self.engine.evaluate_indicator(results, indicator)

        assert len(evaluation_results) == 2  # Two test1 results

        # Check first result
        result1 = evaluation_results[0]
        assert result1.indicator_name == "Test individual success"
        assert result1.test_name == "test1"
        assert result1.outcome == "PASS"
        assert result1.metric_value is True
        assert result1.error is None

        # Check second result
        result2 = evaluation_results[1]
        assert result2.indicator_name == "Test individual success"
        assert result2.test_name == "test1"
        assert result2.outcome == "PASS"
        assert result2.metric_value is True
        assert result2.error is None

    def test_evaluate_indicator_failure(self):
        """Test indicator evaluation with failures."""
        # Create test results with one failure
        results = [
            self.create_test_result(
                "test1",
                "image1",
                {"success": False, "score": 0.5},
            ),
        ]

        # Create indicator
        indicator = ScoreCardIndicator(
            name="Test individual success",
            apply_to=ScoreCardFilter(test_name="test1"),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True),
                AssessmentRule(outcome="FAIL", condition="equal_to", threshold=False),
            ],
        )

        evaluation_results = self.engine.evaluate_indicator(results, indicator)

        assert len(evaluation_results) == 1
        result = evaluation_results[0]
        assert result.outcome == "FAIL"
        assert result.metric_value is False

    def test_evaluate_scorecard(self):
        """Test complete score_card evaluation."""
        # Create test results
        results = [
            self.create_test_result(
                "test1",
                "image1",
                {"success": True, "score": 0.9},
            ),
            self.create_test_result(
                "test1",
                "image2",
                {"success": True, "score": 0.8},
            ),
        ]

        # Create score_card
        score_card = ScoreCard(
            score_card_name="Test score_card",
            indicators=[
                ScoreCardIndicator(
                    name="Individual test success",
                    apply_to=ScoreCardFilter(test_name="test1"),
                    metric="success",
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="equal_to", threshold=True
                        ),
                        AssessmentRule(
                            outcome="FAIL", condition="equal_to", threshold=False
                        ),
                    ],
                ),
                ScoreCardIndicator(
                    name="Individual score quality",
                    apply_to=ScoreCardFilter(test_name="test1"),
                    metric="score",
                    assessment=[
                        AssessmentRule(
                            outcome="EXCELLENT",
                            condition="greater_equal",
                            threshold=0.9,
                        ),
                        AssessmentRule(
                            outcome="GOOD", condition="greater_equal", threshold=0.8
                        ),
                        AssessmentRule(
                            outcome="NEEDS_IMPROVEMENT",
                            condition="less_than",
                            threshold=0.8,
                        ),
                    ],
                ),
            ],
        )

        result = self.engine.evaluate_scorecard(results, score_card)

        # Should return a list of individual evaluations
        assert isinstance(result, list)
        assert len(result) == 4  # 2 tests * 2 indicators

        # Check that all evaluations have the required fields
        for evaluation in result:
            assert "indicator_name" in evaluation
            assert "test_name" in evaluation
            assert "sut_name" in evaluation
            assert "outcome" in evaluation

    def test_evaluate_scorecard_with_no_matching_results(self):
        """Test score_card evaluation when no test results match the filter."""
        # Create test results with different test name
        results = [
            self.create_test_result("test2", "image1", {"success": True, "score": 0.9}),
        ]

        # Create score_card looking for different test name
        score_card = ScoreCard(
            score_card_name="Test score_card",
            indicators=[
                ScoreCardIndicator(
                    name="Individual test success",
                    apply_to=ScoreCardFilter(test_name="test1"),
                    metric="success",
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="equal_to", threshold=True
                        ),
                        AssessmentRule(
                            outcome="FAIL", condition="equal_to", threshold=False
                        ),
                    ],
                )
            ],
        )

        result = self.engine.evaluate_scorecard(results, score_card)

        # Should return a list with one error result
        assert isinstance(result, list)
        assert len(result) == 1
        assert "No test results found for test_name" in result[0]["error"]
