import pytest

from asqi.schemas import AssessmentRule, ScoreCard, ScoreCardFilter, ScoreCardIndicator
from asqi.score_card_engine import ScoreCardEngine, get_nested_value, parse_metric_path
from asqi.workflow import TestExecutionResult


class TestscorecardEngine:
    """Test the ScoreCardEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self, test_name: str, test_id: str, image: str, test_results: dict
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, test_id, "test_sut", image)
        result.test_results = test_results
        result.success = test_results.get("success", True)
        return result

    def test_filter_results_by_test_id(self):
        """Test filtering test results by test ids."""

        results = [
            self.create_test_result("test1", "test_id_1", "image1", {"success": True}),
            self.create_test_result("test2", "test_id_2", "image2", {"success": True}),
            self.create_test_result("test1", "test_id_1", "image2", {"success": False}),
        ]

        filtered = self.engine.filter_results_by_test_id(results, "test_id_1")

        assert len(filtered) == 2
        assert filtered[0].test_id == "test_id_1"
        assert filtered[1].test_id == "test_id_1"

    def test_extract_metric_values(self):
        """Test extracting metric values from test results."""
        results = [
            self.create_test_result(
                "test1", "test_id_1", "image1", {"success": True, "score": 0.9}
            ),
            self.create_test_result(
                "test2", "test_id_2", "image1", {"success": False, "score": 0.5}
            ),
            self.create_test_result(
                "test3", "test_id_3", "image1", {"success": True, "score": 0.8}
            ),
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
                "test1",
                "image1",
                {"success": True, "score": 0.9},
            ),
            self.create_test_result(
                "test1",
                "test1",
                "image2",
                {"success": True, "score": 0.8},
            ),
            self.create_test_result(
                "test2", "test2", "image1", {"success": False, "score": 0.3}
            ),
        ]

        # Create indicator
        indicator = ScoreCardIndicator(
            id="test_individual_success",
            name="Test individual success",
            apply_to=ScoreCardFilter(test_id="test1"),
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
        assert result1.indicator_id == "test_individual_success"
        assert result1.indicator_name == "Test individual success"
        assert result1.test_id == "test1"
        assert result1.outcome == "PASS"
        assert result1.metric_value is True
        assert result1.error is None

        # Check second result
        result2 = evaluation_results[1]
        assert result2.indicator_id == "test_individual_success"
        assert result2.indicator_name == "Test individual success"
        assert result2.test_id == "test1"
        assert result2.outcome == "PASS"
        assert result2.metric_value is True
        assert result2.error is None

    def test_evaluate_indicator_failure(self):
        """Test indicator evaluation with failures."""
        # Create test results with one failure
        results = [
            self.create_test_result(
                "test1",
                "test1",
                "image1",
                {"success": False, "score": 0.5},
            ),
        ]

        # Create indicator
        indicator = ScoreCardIndicator(
            id="test_individual_success",
            name="Test individual success",
            apply_to=ScoreCardFilter(test_id="test1"),
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
                "test1",
                "image1",
                {"success": True, "score": 0.9},
            ),
            self.create_test_result(
                "test1",
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
                    id="individual_test_success",
                    name="Individual test success",
                    apply_to=ScoreCardFilter(test_id="test1"),
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
                    id="individual_score_quality",
                    name="Individual score quality",
                    apply_to=ScoreCardFilter(test_id="test1"),
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
            assert "indicator_id" in evaluation
            assert "indicator_name" in evaluation
            assert "test_id" in evaluation
            assert "sut_name" in evaluation
            assert "outcome" in evaluation

    def test_evaluate_scorecard_with_some_matching_results(self):
        """Test score_card evaluation when some test results match the filter."""
        # Create test results with only one matching name
        results = [
            self.create_test_result(
                "test2", "test2", "image1", {"success": True, "score": 0.9}
            ),
        ]

        # Create score_card looking for several different test ids
        score_card = ScoreCard(
            score_card_name="Test score_card",
            indicators=[
                ScoreCardIndicator(
                    id="individual_test_success",
                    name="Individual test success",
                    apply_to=ScoreCardFilter(test_id="test1"),
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
                    id="individual_score_quality",
                    name="Individual score quality",
                    apply_to=ScoreCardFilter(test_id="test2"),
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

        # Should return a list with one good result and one error result
        assert isinstance(result, list)
        assert len(result) == 2
        assert "No test results found for test_id" in result[0]["error"]

    def test_evaluate_scorecard_with_no_matching_results(self):
        """Test score_card evaluation when no test results match the filter."""
        # Create test results with different test ids
        results = [
            self.create_test_result(
                "test2", "test2", "image1", {"success": True, "score": 0.9}
            ),
        ]

        # Create score_card looking for different test id
        score_card = ScoreCard(
            score_card_name="Test score_card",
            indicators=[
                ScoreCardIndicator(
                    id="individual_test_success",
                    name="Individual test success",
                    apply_to=ScoreCardFilter(test_id="test1"),
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

        with pytest.raises(ValueError, match="Score card indicators don't match"):
            self.engine.evaluate_scorecard(results, score_card)


class TestNestedMetricAccess:
    """Test nested metric access functionality."""

    def test_parse_metric_path_flat(self):
        """Test parsing simple flat metric paths."""
        assert parse_metric_path("success") == ["success"]
        assert parse_metric_path("score") == ["score"]

    def test_parse_metric_path_nested_dots(self):
        """Test parsing nested paths with dot notation."""
        assert parse_metric_path("vulnerability_stats.Toxicity.overall_pass_rate") == [
            "vulnerability_stats",
            "Toxicity",
            "overall_pass_rate",
        ]
        assert parse_metric_path("a.b.c.d") == ["a", "b", "c", "d"]

    def test_parse_metric_path_bracket_notation(self):
        """Test parsing paths with bracket notation for keys containing dots."""
        assert parse_metric_path('probe_results["encoding.InjectHex"]') == [
            "probe_results",
            "encoding.InjectHex",
        ]
        assert parse_metric_path(
            "probe_results['encoding.InjectHex']['encoding.DecodeMatch'].passed"
        ) == ["probe_results", "encoding.InjectHex", "encoding.DecodeMatch", "passed"]

    def test_parse_metric_path_mixed_notation(self):
        """Test parsing paths with mixed dot and bracket notation."""
        assert parse_metric_path(
            'probe_results["encoding.InjectHex"].total_attempts'
        ) == ["probe_results", "encoding.InjectHex", "total_attempts"]
        assert parse_metric_path('stats.probes["test.probe"].results.count') == [
            "stats",
            "probes",
            "test.probe",
            "results",
            "count",
        ]

    def test_parse_metric_path_invalid(self):
        """Test error handling for invalid paths."""
        try:
            parse_metric_path("")
            assert False, "Should have raised ValueError for empty path"
        except ValueError as e:
            assert "cannot be empty" in str(e)

        try:
            parse_metric_path("   ")
            assert False, "Should have raised ValueError for whitespace path"
        except ValueError as e:
            assert "whitespace" in str(e)

        try:
            parse_metric_path('probe_results["unclosed')
            assert False, "Should have raised ValueError for unclosed bracket"
        except ValueError as e:
            assert "Unmatched brackets" in str(e)

        try:
            parse_metric_path("probe_results[unquoted]")
            assert False, "Should have raised ValueError for unquoted bracket"
        except ValueError as e:
            assert "must be quoted" in str(e)

        try:
            parse_metric_path('probe_results[""]')
            assert False, "Should have raised ValueError for empty bracket content"
        except ValueError as e:
            assert "Empty bracket content not allowed" in str(e)

        try:
            parse_metric_path("probe_results[\"mixed']")
            assert False, "Should have raised ValueError for mixed quotes"
        except ValueError as e:
            assert "must be quoted" in str(e)

    def test_parse_metric_path_edge_cases(self):
        """Test parsing edge cases that should work."""
        # Consecutive dots should be handled gracefully
        assert parse_metric_path("a..b") == ["a", "b"]
        assert parse_metric_path("a...b.c") == ["a", "b", "c"]

        # Leading/trailing dots should be handled
        assert parse_metric_path(".success") == ["success"]
        assert parse_metric_path("success.") == ["success"]

        # Keys with special characters in brackets
        assert parse_metric_path('data["key-with-dashes"]') == [
            "data",
            "key-with-dashes",
        ]
        assert parse_metric_path('data["key_with_underscores"]') == [
            "data",
            "key_with_underscores",
        ]

    def test_get_nested_value_flat(self):
        """Test extracting flat values."""
        data = {"success": True, "score": 0.9}

        value, error = get_nested_value(data, "success")
        assert error is None
        assert value is True

        value, error = get_nested_value(data, "score")
        assert error is None
        assert value == 0.9

    def test_get_nested_value_nested(self):
        """Test extracting nested values using dot notation."""
        data = {
            "vulnerability_stats": {
                "Toxicity": {
                    "types": {"profanity": {"pass_rate": 1.0, "passing": 3}},
                    "overall_pass_rate": 0.95,
                }
            }
        }

        value, error = get_nested_value(
            data, "vulnerability_stats.Toxicity.overall_pass_rate"
        )
        assert error is None
        assert value == 0.95

        value, error = get_nested_value(
            data, "vulnerability_stats.Toxicity.types.profanity.pass_rate"
        )
        assert error is None
        assert value == 1.0

    def test_get_nested_value_with_dots_in_keys(self):
        """Test extracting values from keys containing dots using bracket notation."""
        data = {
            "probe_results": {
                "encoding.InjectHex": {
                    "encoding.DecodeMatch": {
                        "passed": 85,
                        "total": 256,
                        "score": 0.33203125,
                    }
                }
            }
        }

        value, error = get_nested_value(
            data, 'probe_results["encoding.InjectHex"]["encoding.DecodeMatch"].passed'
        )
        assert error is None
        assert value == 85

        value, error = get_nested_value(
            data, 'probe_results["encoding.InjectHex"]["encoding.DecodeMatch"].score'
        )
        assert error is None
        assert value == 0.33203125

    def test_get_nested_value_missing_keys(self):
        """Test error handling for missing keys."""
        data = {"a": {"b": {"c": 1}}}

        # Missing intermediate key
        value, error = get_nested_value(data, "a.missing.c")
        assert value is None
        assert error is not None
        assert "Key 'missing' not found at path 'a'" in error
        assert "Available keys: ['b']" in error

        # Missing final key
        value, error = get_nested_value(data, "a.b.missing")
        assert value is None
        assert error is not None
        assert "Key 'missing' not found at path 'a.b'" in error

    def test_get_nested_value_non_dict_traversal(self):
        """Test error handling when trying to traverse non-dictionary values."""
        data = {"a": {"b": 42}}  # b is an integer, not a dict

        value, error = get_nested_value(data, "a.b.c")
        assert value is None
        assert error is not None
        assert "Cannot access key 'c' at path 'a.b'" in error
        assert "value is not a dictionary: int" in error

    def test_scorecard_engine_with_nested_metrics(self):
        """Test the complete scorecard engine with nested metric paths."""
        engine = ScoreCardEngine()

        # Create test result with nested structure similar to garak output
        nested_test_results = {
            "success": True,
            "score": 0.259765625,
            "probe_results": {
                "encoding.InjectHex": {
                    "encoding.DecodeMatch": {
                        "passed": 85,
                        "total": 256,
                        "score": 0.33203125,
                    },
                    "encoding.DecodeApprox": {
                        "passed": 48,
                        "total": 256,
                        "score": 0.1875,
                    },
                }
            },
        }

        test_result = TestExecutionResult(
            "garak_test", "garak_test", "test_sut", "garak:latest"
        )
        test_result.test_results = nested_test_results
        test_result.success = True

        # Create scorecard with nested metric access
        score_card = ScoreCard(
            score_card_name="Nested Metrics Test",
            indicators=[
                ScoreCardIndicator(
                    id="garak_decode_match_score",
                    name="Garak DecodeMatch Score",
                    apply_to=ScoreCardFilter(test_id="garak_test"),
                    metric='probe_results["encoding.InjectHex"]["encoding.DecodeMatch"].score',
                    assessment=[
                        AssessmentRule(
                            outcome="GOOD", condition="greater_than", threshold=0.3
                        ),
                        AssessmentRule(
                            outcome="POOR", condition="less_equal", threshold=0.3
                        ),
                    ],
                ),
                ScoreCardIndicator(
                    id="overall_success_check",
                    name="Overall Success Check",
                    apply_to=ScoreCardFilter(test_id="garak_test"),
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
            ],
        )

        results = engine.evaluate_scorecard([test_result], score_card)

        assert len(results) == 2

        nested_result = next(
            r for r in results if r["indicator_id"] == "garak_decode_match_score"
        )
        assert nested_result["outcome"] == "GOOD"
        assert nested_result["metric_value"] == 0.33203125
        assert nested_result["error"] is None

        flat_result = next(
            r for r in results if r["indicator_id"] == "overall_success_check"
        )
        assert flat_result["outcome"] == "PASS"
        assert flat_result["metric_value"] is True
        assert flat_result["error"] is None
