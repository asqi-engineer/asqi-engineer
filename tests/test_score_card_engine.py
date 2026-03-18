import pytest

from asqi.response_schemas import GeneratedReport
from asqi.schemas import (
    AssessmentRule,
    AuditAssessmentRule,
    AuditResponses,
    AuditScoreCardIndicator,
    MetricExpression,
    ScoreCard,
    ScoreCardFilter,
    ScoreCardIndicator,
)
from asqi.metric_path import get_nested_value, parse_metric_path
from asqi.score_card_engine import ScoreCardEngine
from asqi.workflow import TestExecutionResult


class TestscorecardEngine:
    """Test the ScoreCardEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self,
        test_name: str,
        test_id: str,
        image: str,
        test_results: dict,
        sut_name: str = "test_sut",
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, test_id, sut_name, image)
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
        assert result1.test_ids[0] == "test1"
        assert result1.outcome == "PASS"
        assert result1.metric_value is True
        assert result1.error is None

        # Check second result
        result2 = evaluation_results[1]
        assert result2.indicator_id == "test_individual_success"
        assert result2.indicator_name == "Test individual success"
        assert result2.test_ids[0] == "test1"
        assert result2.outcome == "PASS"
        assert result2.metric_value is True
        assert result2.error is None

    def test_apply_condition_errors(self):
        """Test error cases in apply_condition_to_value."""
        # Missing threshold
        with pytest.raises(ValueError, match="condition requires threshold"):
            self.engine.apply_condition_to_value(0.5, "greater_than", None)

        # Unknown condition
        with pytest.raises(ValueError, match="Unknown condition"):
            self.engine.apply_condition_to_value(0.5, "unknown_condition", 0.5)

        # Non-numeric comparison
        with pytest.raises(ValueError, match="Cannot apply.*non-numeric value"):
            self.engine.apply_condition_to_value("not_a_number", "greater_than", 0.5)

    def test_apply_condition_deprecated_conditions(self):
        """Test deprecated conditions (all_true, any_false) still work for backward compatibility."""
        # all_true condition
        result, desc = self.engine.apply_condition_to_value(True, "all_true", None)
        assert result is True
        assert "truthy" in desc

        # any_false condition
        result, desc = self.engine.apply_condition_to_value(False, "any_false", None)
        assert result is True  # False is falsy, so any_false returns True
        assert "falsy" in desc

    def test_extract_metric_values_edge_cases(self):
        """Test extract_metric_values with edge cases (empty, missing data, missing keys)."""
        # Empty list
        values = self.engine.extract_metric_values([], "some_metric")
        assert values == []

        # No test_results data
        result = self.create_test_result("test", "test_id", "image", {})
        result.test_results = None
        values = self.engine.extract_metric_values([result], "metric")
        assert values == []

        # Missing key
        result = self.create_test_result(
            "test", "test_id", "image", {"existing_key": 0.8}
        )
        values = self.engine.extract_metric_values([result], "missing_key")
        assert values == []

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
            assert "test_ids" in evaluation
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

    def test_evaluate_audit_indicator_no_responses(self):
        """If no audit_responses is provided, we get an error result for that indicator."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
            ],
        )

        results = self.engine.evaluate_audit_indicator(indicator, audit_responses=None)

        assert len(results) == 1
        r = results[0]
        assert r.indicator_id == "config_easy"
        assert r.test_ids[0] == "audit"
        assert r.outcome is None
        assert r.error.startswith(
            "No audit responses provided for indicator_id 'config_easy'"
        )

    def test_evaluate_audit_indicator_missing_for_indicator(self):
        """If audit_responses exist but none match this indicator_id, we get an error."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "other_indicator",
                    "selected_outcome": "A",
                    "notes": "irrelevant",
                }
            ]
        )

        results = self.engine.evaluate_audit_indicator(indicator, audit_responses)

        assert len(results) == 1
        r = results[0]
        assert r.indicator_id == "config_easy"
        assert r.test_ids[0] == "audit"
        assert r.outcome is None
        assert r.error == "No audit response found for indicator_id 'config_easy'"

    def test_evaluate_audit_indicator_success(self):
        """Audit indicator should map selected_outcome + notes into evaluation result."""

        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
                AuditAssessmentRule(outcome="C", description="Medium"),
                AuditAssessmentRule(outcome="D", description="Hard"),
                AuditAssessmentRule(outcome="E", description="Very hard"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "selected_outcome": "C",
                    "notes": "a bit tricky but manageable",
                }
            ]
        )

        results = self.engine.evaluate_audit_indicator(indicator, audit_responses)

        assert len(results) == 1
        r = results[0]
        assert r.indicator_id == "config_easy"
        assert r.test_ids[0] == "audit"
        assert r.outcome == "C"
        assert r.notes == "a bit tricky but manageable"
        # description from the matching AuditAssessmentRule
        assert r.description == "Medium"
        # audit indicators don't attach numeric metric/computed values
        assert r.metric_value is None
        assert r.computed_value is None
        assert r.error is None

    def test_evaluate_audit_indicator_invalid_outcome(self):
        """If selected_outcome is not in the allowed outcomes, we get an error."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
                AuditAssessmentRule(outcome="C", description="Medium"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "selected_outcome": "Z",  # Invalid - not in A, B, C
                    "notes": "some notes",
                }
            ]
        )

        results = self.engine.evaluate_audit_indicator(indicator, audit_responses)

        assert len(results) == 1
        r = results[0]
        assert r.indicator_id == "config_easy"
        assert r.test_ids[0] == "audit"
        assert r.outcome == "Z"  # The invalid outcome is still recorded
        assert r.error is not None
        assert "Invalid selected_outcome 'Z'" in r.error
        assert "Allowed outcomes: ['A', 'B', 'C']" in r.error

    def test_evaluate_audit_indicator_per_system_success(self):
        """Audit responses with sut_name should emit one result per system."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
                AuditAssessmentRule(outcome="C", description="Medium"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_a",
                    "selected_outcome": "A",
                    "notes": "simple",
                },
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_b",
                    "selected_outcome": "C",
                    "notes": "harder",
                },
            ]
        )

        test_results = [
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_a"
            ),
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_b"
            ),
        ]

        score_card = ScoreCard(
            score_card_name="Audit SUT Scorecard",
            indicators=[indicator],
        )

        results = self.engine.evaluate_scorecard(
            test_results, score_card, audit_responses
        )

        assert len(results) == 2
        by_sut = {r["sut_name"]: r for r in results}
        assert set(by_sut.keys()) == {"sut_a", "sut_b"}
        assert by_sut["sut_a"]["outcome"] == "A"
        assert by_sut["sut_a"]["audit_notes"] == "simple"
        assert by_sut["sut_b"]["outcome"] == "C"
        assert by_sut["sut_b"]["audit_notes"] == "harder"
        assert all(r["error"] is None for r in results)

    def test_evaluate_audit_indicator_missing_sut_responses(self):
        """Per-system audits must cover all systems under test."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_a",
                    "selected_outcome": "A",
                }
            ]
        )

        test_results = [
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_a"
            ),
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_b"
            ),
        ]

        score_card = ScoreCard(
            score_card_name="Audit SUT Scorecard",
            indicators=[indicator],
        )

        results = self.engine.evaluate_scorecard(
            test_results, score_card, audit_responses
        )

        assert len(results) == 1
        assert (
            results[0]["error"]
            == "Audit indicator 'config_easy' requires responses for all systems: missing ['sut_b']"
        )

    def test_evaluate_audit_indicator_unknown_sut(self):
        """Audit responses with unknown sut_name should produce an error."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "sut_name": "unknown_sut",
                    "selected_outcome": "A",
                }
            ]
        )

        test_results = [
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_a"
            ),
        ]

        score_card = ScoreCard(
            score_card_name="Audit SUT Scorecard",
            indicators=[indicator],
        )

        results = self.engine.evaluate_scorecard(
            test_results, score_card, audit_responses
        )

        assert len(results) == 1
        assert (
            results[0]["error"]
            == "'unknown_sut' is not a valid system under test for this evaluation"
        )

    def test_evaluate_audit_indicator_mixed_per_system_and_global(self):
        """Mixed per-system and global responses should error."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_a",
                    "selected_outcome": "A",
                },
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_b",
                    "selected_outcome": "B",
                },
                {
                    "indicator_id": "config_easy",
                    "selected_outcome": "A",
                    "notes": "global view",
                },
            ]
        )

        test_results = [
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_a"
            ),
            self.create_test_result(
                "test1", "test1", "image1", {"success": True}, "sut_b"
            ),
        ]

        score_card = ScoreCard(
            score_card_name="Audit SUT Scorecard",
            indicators=[indicator],
        )

        results = self.engine.evaluate_scorecard(
            test_results, score_card, audit_responses
        )

        assert len(results) == 1
        assert (
            results[0]["error"]
            == "Audit indicator 'config_easy' cannot mix global and per-system responses"
        )

    def test_evaluate_audit_indicator_available_suts_none(self):
        """Per-system responses should not error when available_suts is missing."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_a",
                    "selected_outcome": "A",
                }
            ]
        )

        results = self.engine.evaluate_audit_indicator(
            indicator, audit_responses, available_suts=None
        )

        assert len(results) == 1
        assert results[0].sut_name == "sut_a"
        assert results[0].error is None

    def test_evaluate_audit_indicator_duplicate_responses(self):
        """Duplicate responses for same indicator + sut should error."""
        indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
            ],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_a",
                    "selected_outcome": "A",
                },
                {
                    "indicator_id": "config_easy",
                    "sut_name": "sut_a",
                    "selected_outcome": "B",
                },
            ]
        )

        results = self.engine.evaluate_audit_indicator(
            indicator, audit_responses, {"sut_a"}
        )

        assert len(results) == 1
        assert "Duplicate audit responses" in results[0].error


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
        """Test error handling for invalid paths with comprehensive ValueError coverage."""
        # Empty path
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_metric_path("")

        # Whitespace-only path
        with pytest.raises(ValueError, match="whitespace"):
            parse_metric_path("   ")

        # Unclosed bracket
        with pytest.raises(ValueError, match="Unmatched brackets"):
            parse_metric_path('probe_results["unclosed')

        # Unquoted bracket content
        with pytest.raises(ValueError, match="must be quoted"):
            parse_metric_path("probe_results[unquoted]")

        # Empty bracket content
        with pytest.raises(ValueError, match="Empty bracket content not allowed"):
            parse_metric_path('probe_results[""]')

        # Mixed quote types
        with pytest.raises(ValueError, match="must be quoted"):
            parse_metric_path("probe_results[\"mixed']")

        # Only dots (results in no keys)
        with pytest.raises(ValueError, match="Invalid metric path resulted in no keys"):
            parse_metric_path(".")

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

    def test_get_nested_value_invalid_path_format(self):
        """Test get_nested_value with invalid metric path formats (triggers parse_metric_path exceptions)."""
        data = {"a": 1, "b": 2}

        # Invalid path (empty) - triggers ValueError in parse_metric_path
        value, error = get_nested_value(data, "")
        assert value is None
        assert "cannot be empty" in error

        # Invalid path (whitespace only)
        value, error = get_nested_value(data, "   ")
        assert value is None
        assert "whitespace" in error

        # Invalid path (unquoted bracket)
        value, error = get_nested_value(data, "a[unquoted]")
        assert value is None
        assert "must be quoted" in error

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


class TestMetricExpressions:
    """Test metric expression evaluation in score card engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self, test_name: str, test_id: str, test_results: dict
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, test_id, "test_sut", "test_image")
        result.test_results = test_results
        result.success = True
        return result

    def test_simple_metric_backward_compatible(self):
        """Test that simple metric paths still work (backward compatibility)."""
        test_result = self.create_test_result("test1", "test_id_1", {"accuracy": 0.85})

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], "accuracy"
        )

        assert error is None
        assert value == 0.85

    def test_expression_weighted_sum(self):
        """Test weighted sum expression."""
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"accuracy": 0.8, "relevance": 0.9},
        )

        metric_expr = MetricExpression(
            expression="0.7 * accuracy + 0.3 * relevance",
            values={"accuracy": "accuracy", "relevance": "relevance"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert error is None
        assert value == pytest.approx(0.83)

    def test_expression_with_min(self):
        """Test expression using min function."""
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"score1": 0.9, "score2": 0.7, "score3": 0.8},
        )

        metric_expr = MetricExpression(
            expression="min(score1, score2, score3)",
            values={"score1": "score1", "score2": "score2", "score3": "score3"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert error is None
        assert value == 0.7

    def test_expression_with_max(self):
        """Test expression using max function."""
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"score1": 0.9, "score2": 0.7, "score3": 0.8},
        )

        metric_expr = MetricExpression(
            expression="max(score1, score2, score3)",
            values={"score1": "score1", "score2": "score2", "score3": "score3"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert error is None
        assert value == 0.9

    def test_expression_with_avg(self):
        """Test expression using avg function."""
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"score1": 0.6, "score2": 0.8, "score3": 1.0},
        )

        metric_expr = MetricExpression(
            expression="avg(score1, score2, score3)",
            values={"score1": "score1", "score2": "score2", "score3": "score3"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert error is None
        assert value == pytest.approx(0.8)

    def test_expression_complex_formula(self):
        """Test complex expression with multiple operations."""
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"accuracy": 0.9, "relevance": 0.8},
        )

        metric_expr = MetricExpression(
            expression="min(0.7 * accuracy + 0.3 * relevance, 1.0)",
            values={"accuracy": "accuracy", "relevance": "relevance"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert error is None
        assert value == pytest.approx(0.87)

    def test_expression_with_nested_metrics(self):
        """Test expression with nested metric paths.

        With dict mapping, we can use simple variable names in expressions
        while extracting from nested paths.
        """
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"stats": {"pass_rate": 0.7, "fail_rate": 0.3}},
        )

        metric_expr = MetricExpression(
            expression="pass_rate + fail_rate",
            values={"pass_rate": "stats.pass_rate", "fail_rate": "stats.fail_rate"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert error is None
        assert value == pytest.approx(1.0)

    def test_expression_missing_metric(self):
        """Test that missing metrics return appropriate error."""
        test_result = self.create_test_result("test1", "test_id_1", {"accuracy": 0.8})

        metric_expr = MetricExpression(
            expression="accuracy + missing_metric",
            values={"accuracy": "accuracy", "missing_metric": "missing_metric"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert value is None
        assert error is not None
        assert "missing_metric" in error

    def test_expression_non_numeric_metric(self):
        """Test that non-numeric metrics return appropriate error."""
        test_result = self.create_test_result(
            "test1",
            "test_id_1",
            {"accuracy": "high"},  # String, not number
        )

        metric_expr = MetricExpression(
            expression="accuracy * 2",
            values={"accuracy": "accuracy"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert value is None
        assert error is not None
        assert "non-numeric" in error

    def test_expression_division_by_zero(self):
        """Test that division by zero returns appropriate error."""
        test_result = self.create_test_result(
            "test1", "test_id_1", {"numerator": 10, "denominator": 0}
        )

        metric_expr = MetricExpression(
            expression="numerator / denominator",
            values={"numerator": "numerator", "denominator": "denominator"},
        )

        value, error = self.engine.resolve_metric_or_expression(
            [test_result], metric_expr
        )

        assert value is None
        assert error is not None
        assert "Division by zero" in error

    def test_evaluate_indicator_with_expression(self):
        """Test full indicator evaluation with expression."""
        test_results = [
            self.create_test_result(
                "test1", "chatbot_test", {"accuracy": 0.9, "relevance": 0.85}
            )
        ]

        metric_expr = MetricExpression(
            expression="0.6 * accuracy + 0.4 * relevance",
            values={"accuracy": "accuracy", "relevance": "relevance"},
        )

        indicator = ScoreCardIndicator(
            id="combined_score",
            name="Combined Quality Score",
            apply_to=ScoreCardFilter(test_id="chatbot_test"),
            metric=metric_expr,
            assessment=[
                AssessmentRule(outcome="A", condition="greater_equal", threshold=0.85),
                AssessmentRule(outcome="B", condition="greater_equal", threshold=0.75),
                AssessmentRule(outcome="C", condition="less_than", threshold=0.75),
            ],
        )

        results = self.engine.evaluate_indicator(test_results, indicator)

        assert len(results) == 1
        result = results[0]

        assert result.outcome == "A"
        assert result.metric_value == pytest.approx(0.88)
        assert result.error is None

    def test_evaluate_scorecard_with_expressions(self):
        """Test full scorecard evaluation with multiple expression indicators."""
        test_result = self.create_test_result(
            "chatbot_test",
            "chatbot_test",
            {
                "accuracy": 0.85,
                "relevance": 0.90,
                "score1": 0.7,
                "score2": 0.8,
                "score3": 0.75,
            },
        )

        metric_expr1 = MetricExpression(
            expression="0.5 * accuracy + 0.5 * relevance",
            values={"accuracy": "accuracy", "relevance": "relevance"},
        )

        metric_expr2 = MetricExpression(
            expression="min(score1, score2, score3)",
            values={"score1": "score1", "score2": "score2", "score3": "score3"},
        )

        score_card = ScoreCard(
            score_card_name="Expression Test Scorecard",
            indicators=[
                ScoreCardIndicator(
                    id="weighted_quality",
                    name="Weighted Quality",
                    apply_to=ScoreCardFilter(test_id="chatbot_test"),
                    metric=metric_expr1,
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="greater_equal", threshold=0.8
                        ),
                        AssessmentRule(
                            outcome="FAIL", condition="less_than", threshold=0.8
                        ),
                    ],
                ),
                ScoreCardIndicator(
                    id="min_score",
                    name="Minimum Score",
                    apply_to=ScoreCardFilter(test_id="chatbot_test"),
                    metric=metric_expr2,
                    assessment=[
                        AssessmentRule(
                            outcome="GOOD", condition="greater_equal", threshold=0.7
                        ),
                        AssessmentRule(
                            outcome="BAD", condition="less_than", threshold=0.7
                        ),
                    ],
                ),
            ],
        )

        results = self.engine.evaluate_scorecard([test_result], score_card)

        assert len(results) == 2

        weighted_result = next(
            r for r in results if r["indicator_id"] == "weighted_quality"
        )
        assert weighted_result["outcome"] == "PASS"
        assert weighted_result["metric_value"] == pytest.approx(0.875)

        min_result = next(r for r in results if r["indicator_id"] == "min_score")
        assert min_result["outcome"] == "GOOD"
        assert min_result["metric_value"] == 0.7

    def test_evaluate_scorecard_audit_only_no_test_results(self):
        """Audit-only scorecard should evaluate using audit_responses even without test results."""

        audit_indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
                AuditAssessmentRule(outcome="C", description="Medium"),
            ],
        )

        score_card = ScoreCard(
            score_card_name="Audit Only Scorecard",
            indicators=[audit_indicator],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "selected_outcome": "B",
                    "notes": "pretty simple",
                }
            ]
        )

        # No test_results, but should still work for audit indicators
        results = self.engine.evaluate_scorecard(
            test_results=[],
            score_card=score_card,
            audit_responses_data=audit_responses,
        )

        assert len(results) == 1
        r = results[0]
        assert r["indicator_id"] == "config_easy"
        assert r["test_ids"] == ["audit"]
        assert r["outcome"] == "B"
        assert r["audit_notes"] == "pretty simple"
        assert r["error"] is None

    def test_evaluate_scorecard_with_metric_and_audit_indicators(self):
        """Scorecard mixing metric and audit indicators evaluates both correctly."""

        # Metric-based test result
        test_results = [
            self.create_test_result(
                "quality_test",
                "quality_test",
                {"success": True, "accuracy": 0.9},
            )
        ]

        metric_indicator = ScoreCardIndicator(
            id="success_check",
            name="Success Check",
            apply_to=ScoreCardFilter(test_id="quality_test"),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True),
                AssessmentRule(outcome="FAIL", condition="equal_to", threshold=False),
            ],
        )

        audit_indicator = AuditScoreCardIndicator(
            id="config_easy",
            name="Configuration Ease",
            type="audit",
            assessment=[
                AuditAssessmentRule(outcome="A", description="Very easy"),
                AuditAssessmentRule(outcome="B", description="Easy"),
                AuditAssessmentRule(outcome="C", description="Medium"),
            ],
        )

        score_card = ScoreCard(
            score_card_name="Mixed Scorecard",
            indicators=[metric_indicator, audit_indicator],
        )

        audit_responses = AuditResponses(
            responses=[
                {
                    "indicator_id": "config_easy",
                    "selected_outcome": "C",
                    "notes": "UI is a bit complex",
                }
            ]
        )

        results = self.engine.evaluate_scorecard(
            test_results=test_results,
            score_card=score_card,
            audit_responses_data=audit_responses,
        )

        # We expect 1 metric + 1 audit evaluation
        assert len(results) == 2

        success_eval = next(r for r in results if r["indicator_id"] == "success_check")
        audit_eval = next(r for r in results if r["indicator_id"] == "config_easy")

        # Metric indicator result
        assert success_eval["test_ids"] == ["quality_test"]
        assert success_eval["outcome"] == "PASS"
        assert success_eval["metric_value"] is True
        assert success_eval["error"] is None

        # Audit indicator result
        assert audit_eval["test_ids"] == ["audit"]
        assert audit_eval["outcome"] == "C"
        assert audit_eval["audit_notes"] == "UI is a bit complex"
        assert audit_eval["description"] == "Medium"
        assert audit_eval["metric_value"] is None
        assert audit_eval["error"] is None

    def test_resolve_metric_empty_test_results(self):
        """Test resolve_metric_or_expression with empty test_results list."""
        value, error = self.engine.resolve_metric_or_expression([], "metric_path")
        assert value is None
        assert "No test results provided" in error

    def test_evaluate_indicator_assessment_rule_exception(self):
        """Test that exception in assessment rule evaluation is caught and logged."""
        # Set score to a complex object that can't be compared
        result = self.create_test_result(
            "test", "test_id", {"score": {"nested": "object"}}
        )

        indicator = ScoreCardIndicator(
            id="bad_assessment",
            name="Bad Assessment",
            apply_to=ScoreCardFilter(test_id="test_id"),
            metric="score",
            assessment=[
                AssessmentRule(
                    # Will fail when trying to convert complex object to number
                    outcome="PASS",
                    condition="greater_equal",
                    threshold=0.5,
                ),
            ],
        )

        eval_results = self.engine.evaluate_indicator([result], indicator)
        assert len(eval_results) == 1
        # Should have error from the exception during condition evaluation
        assert eval_results[0].error is not None

    def test_evaluate_indicator_no_rule_matches(self):
        """Test when metric value doesn't match any assessment rule."""
        result = self.create_test_result("test", "test_id", {"score": 0.3})

        indicator = ScoreCardIndicator(
            id="no_match",
            name="No Match",
            apply_to=ScoreCardFilter(test_id="test_id"),
            metric="score",
            assessment=[
                AssessmentRule(
                    outcome="PASS",
                    condition="greater_equal",
                    threshold=0.8,
                ),
            ],
        )

        eval_results = self.engine.evaluate_indicator([result], indicator)
        assert len(eval_results) == 1
        assert eval_results[0].error == "No assessment rule conditions were satisfied"
        assert eval_results[0].outcome is None


class TestDisplayGeneratedReports:
    @pytest.fixture
    def test_execution_result(self) -> TestExecutionResult:
        test_result = TestExecutionResult(
            test_name="report test",
            test_id="report_test",
            sut_name="sut",
            image="report-image:latest",
        )
        test_result.test_results = {"score": 0.95}
        test_result.success = True
        return test_result

    @pytest.fixture
    def indicator(self) -> ScoreCardIndicator:
        return ScoreCardIndicator(
            id="indicator_report",
            name="indicator report",
            apply_to=ScoreCardFilter(test_id="report_test"),
            metric="score",
            assessment=[
                AssessmentRule(outcome="PASS", condition="greater_equal", threshold=0.9)
            ],
        )

    def test_display_reports(self, test_execution_result, indicator):
        """
        Test that the ScoreCardEngine returns only the reports explicitly listed in display_reports.
        """
        engine = ScoreCardEngine()
        test_execution_result.generated_reports = [
            GeneratedReport(
                report_name="detailed_report",
                report_type="html",
                report_path="/reports/detailed_report.html",
            ),
            GeneratedReport(
                report_name="summary_report",
                report_type="html",
                report_path="/reports/summary_report.html",
            ),
        ]

        indicator.display_reports = ["detailed_report"]
        results = engine.evaluate_indicator([test_execution_result], indicator)

        assert len(results) == 1
        assert results[0].report_paths == ["/reports/detailed_report.html"]

    def test_reports_with_invalid_path(self, test_execution_result, indicator):
        """
        Test that only reports matching display_reports are included in results.
        Note: Pydantic validation ensures report_path is always non-empty,
        so invalid paths can't be created in the first place.
        """
        engine = ScoreCardEngine()

        test_execution_result.generated_reports = [
            GeneratedReport(
                report_name="valid_report",
                report_type="pdf",
                report_path="/reports/valid_report.pdf",
            ),
            GeneratedReport(
                report_name="other_report",
                report_type="pdf",
                report_path="/reports/other_report.pdf",
            ),
        ]
        indicator.display_reports = [
            "valid_report",
            "nonexistent_report",  # Report that doesn't exist in generated_reports
        ]
        results = engine.evaluate_indicator([test_execution_result], indicator)
        # Only the valid_report should be included since it matches display_reports
        assert results[0].report_paths == ["/reports/valid_report.pdf"]

    def test_explicit_display_reports_syntax(self):
        """Test display_reports with explicit test_id::report_name syntax."""
        engine = ScoreCardEngine()

        # Create test results for two containers
        accuracy_result = TestExecutionResult(
            "accuracy_test", "accuracy_test", "sut", "accuracy-image:latest"
        )
        accuracy_result.test_results = {"score": 0.95}
        accuracy_result.success = True
        accuracy_result.generated_reports = [
            GeneratedReport(
                report_name="accuracy_report",
                report_type="html",
                report_path="/reports/accuracy_report.html",
            ),
            GeneratedReport(
                report_name="quick_summary",
                report_type="html",
                report_path="/reports/accuracy_summary.html",
            ),
        ]

        robustness_result = TestExecutionResult(
            "robustness_test", "robustness_test", "sut", "robustness-image:latest"
        )
        robustness_result.test_results = {
            "ood_detection_accuracy": 0.85,
            "adversarial_robustness": 0.80,
        }
        robustness_result.success = True
        robustness_result.generated_reports = [
            GeneratedReport(
                report_name="robustness_report",
                report_type="html",
                report_path="/reports/robustness_report.html",
            ),
            GeneratedReport(
                report_name="quick_summary",
                report_type="html",
                report_path="/reports/robustness_summary.html",
            ),
        ]

        # Create indicator with explicit syntax
        indicator = ScoreCardIndicator(
            id="multi_container_reports",
            name="Multi-Container Reports",
            apply_to=ScoreCardFilter(test_id=["accuracy_test", "robustness_test"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_test::score",
                    "ood": "robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(outcome="PASS", condition="greater_equal", threshold=0.8)
            ],
            display_reports=[
                "accuracy_test::accuracy_report",
                "robustness_test::robustness_report",
            ],
        )

        results = engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        assert len(results) == 1
        assert results[0].outcome == "PASS"
        # Should have reports from both containers
        assert len(results[0].report_paths) == 2
        assert "/reports/accuracy_report.html" in results[0].report_paths
        assert "/reports/robustness_report.html" in results[0].report_paths

    def test_mixed_display_reports_syntax(self):
        """Test display_reports with mixed simple names and explicit syntax."""
        engine = ScoreCardEngine()

        # Both containers have quick_summary
        accuracy_result = TestExecutionResult(
            "accuracy_test", "accuracy_test", "sut", "accuracy-image:latest"
        )
        accuracy_result.test_results = {"score": 0.95}
        accuracy_result.success = True
        accuracy_result.generated_reports = [
            GeneratedReport(
                report_name="accuracy_report",
                report_type="html",
                report_path="/reports/accuracy_report.html",
            ),
            GeneratedReport(
                report_name="quick_summary",
                report_type="html",
                report_path="/reports/accuracy_summary.html",
            ),
        ]

        robustness_result = TestExecutionResult(
            "robustness_test", "robustness_test", "sut", "robustness-image:latest"
        )
        robustness_result.test_results = {"ood_detection_accuracy": 0.85}
        robustness_result.success = True
        robustness_result.generated_reports = [
            GeneratedReport(
                report_name="robustness_report",
                report_type="html",
                report_path="/reports/robustness_report.html",
            ),
            GeneratedReport(
                report_name="quick_summary",
                report_type="html",
                report_path="/reports/robustness_summary.html",
            ),
        ]

        # Mixed: simple name (found in both) + explicit syntax (specific container)
        indicator = ScoreCardIndicator(
            id="mixed_reports",
            name="Mixed Reports",
            apply_to=ScoreCardFilter(test_id=["accuracy_test", "robustness_test"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_test::score",
                    "ood": "robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(outcome="PASS", condition="greater_equal", threshold=0.8)
            ],
            display_reports=[
                "quick_summary",  # Simple name - should match from both
                "accuracy_test::accuracy_report",  # Explicit - only from accuracy
            ],
        )

        results = engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        assert len(results) == 1
        # Should have: accuracy_summary, robustness_summary, accuracy_report = 3 reports
        assert len(results[0].report_paths) == 3
        assert "/reports/accuracy_summary.html" in results[0].report_paths
        assert "/reports/robustness_summary.html" in results[0].report_paths
        assert "/reports/accuracy_report.html" in results[0].report_paths
        assert "/reports/robustness_report.html" not in results[0].report_paths

    def test_explicit_syntax_nonexistent_report(self):
        """Test error handling when explicit syntax references non-existent report."""
        engine = ScoreCardEngine()

        accuracy_result = TestExecutionResult(
            "accuracy_test", "accuracy_test", "sut", "accuracy-image:latest"
        )
        accuracy_result.test_results = {"score": 0.95}
        accuracy_result.success = True
        accuracy_result.generated_reports = [
            GeneratedReport(
                report_name="accuracy_report",
                report_type="html",
                report_path="/reports/accuracy_report.html",
            ),
        ]

        robustness_result = TestExecutionResult(
            "robustness_test", "robustness_test", "sut", "robustness-image:latest"
        )
        robustness_result.test_results = {"ood_detection_accuracy": 0.85}
        robustness_result.success = True
        robustness_result.generated_reports = [
            GeneratedReport(
                report_name="robustness_report",
                report_type="html",
                report_path="/reports/robustness_report.html",
            ),
        ]

        # Request nonexistent report from robustness container
        indicator = ScoreCardIndicator(
            id="nonexistent_report",
            name="Nonexistent Report",
            apply_to=ScoreCardFilter(test_id=["accuracy_test", "robustness_test"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_test::score",
                    "ood": "robustness_test::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(outcome="PASS", condition="greater_equal", threshold=0.8)
            ],
            display_reports=[
                "accuracy_test::accuracy_report",
                "robustness_test::nonexistent_report",  # Doesn't exist
            ],
        )

        results = engine.evaluate_indicator(
            [accuracy_result, robustness_result], indicator
        )

        assert len(results) == 1
        # Should only have the accuracy_report since robustness one doesn't exist
        assert len(results[0].report_paths) == 1
        assert "/reports/accuracy_report.html" in results[0].report_paths


class TestScoreCardSystemTypeFiltering:
    """Test score card filtering by system type (Issue #288)."""

    def test_filter_by_single_system_type(self):
        """Test filtering results by a single system type."""
        engine = ScoreCardEngine()

        # Create test results with different system types
        results = [
            TestExecutionResult("test1", "test1", "sut_llm", "image1", "llm_api"),
            TestExecutionResult("test1", "test1", "sut_vlm", "image1", "vlm_api"),
            TestExecutionResult("test1", "test1", "sut_rag", "image1", "rag_api"),
        ]

        # Filter for llm_api only
        filtered = engine.filter_results_by_test_and_type(results, "test1", ["llm_api"])

        assert len(filtered) == 1
        assert filtered[0].system_type == "llm_api"
        assert filtered[0].sut_name == "sut_llm"

    def test_filter_by_multiple_system_types(self):
        """Test filtering results by multiple system types."""
        engine = ScoreCardEngine()

        # Create test results with different system types
        results = [
            TestExecutionResult("test1", "test1", "sut_llm", "image1", "llm_api"),
            TestExecutionResult("test1", "test1", "sut_vlm", "image1", "vlm_api"),
            TestExecutionResult("test1", "test1", "sut_rag", "image1", "rag_api"),
        ]

        # Filter for llm_api and vlm_api
        filtered = engine.filter_results_by_test_and_type(
            results, "test1", ["llm_api", "vlm_api"]
        )

        assert len(filtered) == 2
        system_types = [r.system_type for r in filtered]
        assert "llm_api" in system_types
        assert "vlm_api" in system_types
        assert "rag_api" not in system_types

    def test_no_system_type_filter_matches_all(self):
        """Test that omitting system_type filter matches all system types."""
        engine = ScoreCardEngine()

        # Create test results with different system types
        results = [
            TestExecutionResult("test1", "test1", "sut_llm", "image1", "llm_api"),
            TestExecutionResult("test1", "test1", "sut_vlm", "image1", "vlm_api"),
            TestExecutionResult("test1", "test1", "sut_rag", "image1", "rag_api"),
        ]

        # Filter without system_type (None = all types)
        filtered = engine.filter_results_by_test_and_type(results, "test1", None)

        assert len(filtered) == 3

    def test_system_type_stored_in_test_result(self):
        """Test that TestExecutionResult correctly stores and exposes system_type."""
        result = TestExecutionResult(
            "my_test", "my_test_id", "my_sut", "my_image", "llm_api"
        )

        assert result.system_type == "llm_api"

        # Verify it appears in to_dict() output
        result_dict = result.result_dict()
        assert result_dict["metadata"]["system_type"] == "llm_api"

    def test_backward_compatibility_no_system_type(self):
        """Test that old test results without system_type field still work."""
        engine = ScoreCardEngine()

        # Create result without system_type (defaults to None)
        result_old = TestExecutionResult("test1", "test1", "sut_old", "image1")

        # Verify system_type is None
        assert result_old.system_type is None

        # Filter should not match when system_type is specified
        filtered = engine.filter_results_by_test_and_type(
            [result_old], "test1", ["llm_api"]
        )
        assert len(filtered) == 0

        # But should match when no system_type filter
        filtered_all = engine.filter_results_by_test_and_type(
            [result_old], "test1", None
        )
        assert len(filtered_all) == 1

    def test_evaluate_indicator_with_system_type_filter(self):
        """Test that score card indicators filter by system type correctly."""
        engine = ScoreCardEngine()

        # Create test results with different system types
        results = [
            TestExecutionResult("test1", "test1", "sut_llm", "image1", "llm_api"),
            TestExecutionResult("test1", "test1", "sut_vlm", "image1", "vlm_api"),
        ]

        # Set test results for evaluation
        for r in results:
            r.success = True
            r.test_results = {"success": True, "score": 0.9}

        # Create score card indicator with system type filter
        indicator = ScoreCardIndicator(
            id="llm_only",
            name="LLM Only Success Check",
            apply_to=ScoreCardFilter(test_id="test1", target_system_type="llm_api"),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True)
            ],
        )

        # Evaluate - should only match LLM result
        eval_results = engine.evaluate_indicator(results, indicator)

        assert len(eval_results) == 1
        assert eval_results[0].sut_name == "sut_llm"
        assert eval_results[0].outcome == "PASS"

    def test_evaluate_indicator_with_multiple_system_type_filter(self):
        """Test that score card indicators can filter by multiple system types."""
        engine = ScoreCardEngine()

        # Create test results with different system types
        results = [
            TestExecutionResult("test1", "test1", "sut_llm", "image1", "llm_api"),
            TestExecutionResult("test1", "test1", "sut_vlm", "image1", "vlm_api"),
            TestExecutionResult("test1", "test1", "sut_rag", "image1", "rag_api"),
        ]

        # Set test results for evaluation
        for r in results:
            r.success = True
            r.test_results = {"success": True, "score": 0.9}

        # Create score card indicator with multiple system types
        indicator = ScoreCardIndicator(
            id="llm_vlm_check",
            name="LLM and VLM Success Check",
            apply_to=ScoreCardFilter(
                test_id="test1", target_system_type=["llm_api", "vlm_api"]
            ),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True)
            ],
        )

        # Evaluate - should match both LLM and VLM results
        eval_results = engine.evaluate_indicator(results, indicator)

        assert len(eval_results) == 2
        sut_names = [r.sut_name for r in eval_results]
        assert "sut_llm" in sut_names
        assert "sut_vlm" in sut_names
        assert all(r.outcome == "PASS" for r in eval_results)

    def test_error_message_distinguishes_system_type_mismatch(self):
        """Test that error messages distinguish between missing test_id and system type mismatch."""
        engine = ScoreCardEngine()

        # Create test results with LLM and VLM system types
        results = [
            TestExecutionResult("test1", "test1", "sut_llm", "image1", "llm_api"),
            TestExecutionResult("test1", "test1", "sut_vlm", "image1", "vlm_api"),
        ]

        for r in results:
            r.success = True
            r.test_results = {"success": True}

        # Case 1: Filter for RAG (no results, system type mismatch)
        indicator_rag = ScoreCardIndicator(
            id="rag_check",
            name="RAG Success Check",
            apply_to=ScoreCardFilter(test_id="test1", target_system_type="rag_api"),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True)
            ],
        )

        eval_results = engine.evaluate_indicator(results, indicator_rag)
        assert len(eval_results) == 1
        assert eval_results[0].error is not None
        # Should mention that test1 exists but with different system types
        assert "test_id 'test1' with system type(s) [rag_api]" in eval_results[0].error
        assert (
            "has results for system type(s): llm_api, vlm_api" in eval_results[0].error
            or "has results for system type(s): vlm_api, llm_api"
            in eval_results[0].error
        )

        # Case 2: Filter for non-existent test (no results, test_id doesn't exist)
        indicator_missing = ScoreCardIndicator(
            id="missing_check",
            name="Missing Test Check",
            apply_to=ScoreCardFilter(
                test_id="test_does_not_exist", target_system_type="llm_api"
            ),
            metric="success",
            assessment=[
                AssessmentRule(outcome="PASS", condition="equal_to", threshold=True)
            ],
        )

        eval_results = engine.evaluate_indicator(results, indicator_missing)
        assert len(eval_results) == 1
        assert eval_results[0].error is not None
        # Should mention available tests
        assert (
            "No test results found for test_id 'test_does_not_exist'"
            in eval_results[0].error
        )
        assert "Available tests: test1" in eval_results[0].error


class TestCrossContainerExpressions:
    """Test cross-container metric expression support."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self,
        test_name: str,
        test_id: str,
        test_results: dict,
        sut_name: str = "test_sut",
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, test_id, sut_name, "test_image")
        result.test_results = test_results
        result.success = True
        return result

    def test_resolve_cross_container_metric(self):
        """Test resolving a metric from a different container."""
        result_accuracy = self.create_test_result(
            "accuracy_test", "accuracy_rag", {"pass_rate": 0.9}
        )
        result_robustness = self.create_test_result(
            "robustness_test", "robustness_rag", {"ood_accuracy": 0.8}
        )

        metric_expr = MetricExpression(
            expression="0.6 * acc + 0.4 * ood",
            values={
                "acc": "accuracy_rag::pass_rate",
                "ood": "robustness_rag::ood_accuracy",
            },
        )

        # Resolve from accuracy container, but with both containers available
        value, error = self.engine.resolve_metric_or_expression(
            [result_accuracy, result_robustness],
            metric_expr,
        )

        assert error is None
        assert value == pytest.approx(0.6 * 0.9 + 0.4 * 0.8)

    def test_resolve_cross_container_metric_missing_container(self):
        """Test error when referenced container is missing."""
        result_accuracy = self.create_test_result(
            "accuracy_test", "accuracy_rag", {"pass_rate": 0.9}
        )

        metric_expr = MetricExpression(
            expression="0.6 * acc + 0.4 * ood",
            values={
                "acc": "accuracy_rag::pass_rate",
                "ood": "robustness_rag::ood_accuracy",
            },
        )

        # Only provide accuracy container, robustness is missing
        value, error = self.engine.resolve_metric_or_expression(
            [result_accuracy],
            metric_expr,
        )

        assert error is not None
        assert "No result found for test_id 'robustness_rag'" in error

    def test_evaluate_indicator_multi_test_id(self):
        """Test evaluating indicator with multiple test_ids."""
        result_accuracy = self.create_test_result(
            "accuracy_test",
            "accuracy_rag",
            {"hard_gate_pass_rate": 0.9},
            sut_name="my_model",
        )
        result_robustness = self.create_test_result(
            "robustness_test",
            "robustness_rag",
            {"ood_detection_accuracy": 0.8},
            sut_name="my_model",
        )

        indicator = ScoreCardIndicator(
            id="combined_score",
            name="Combined Accuracy + Robustness",
            apply_to=ScoreCardFilter(test_id=["accuracy_rag", "robustness_rag"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_rag::hard_gate_pass_rate",
                    "ood": "robustness_rag::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="PASS", condition="greater_equal", threshold=0.8
                ),
                AssessmentRule(outcome="FAIL", condition="less_than", threshold=0.8),
            ],
        )

        eval_results = self.engine.evaluate_indicator(
            [result_accuracy, result_robustness], indicator
        )

        assert len(eval_results) == 1
        assert eval_results[0].error is None
        assert eval_results[0].sut_name == "my_model"
        expected_value = 0.6 * 0.9 + 0.4 * 0.8  # 0.86
        assert eval_results[0].metric_value == pytest.approx(expected_value)
        assert eval_results[0].outcome == "PASS"

    def test_evaluate_indicator_multi_test_id_missing_container(self):
        """Test error when SUT has results in one container but not another."""
        result_accuracy = self.create_test_result(
            "accuracy_test",
            "accuracy_rag",
            {"hard_gate_pass_rate": 0.9},
            sut_name="my_model",
        )
        # robustness_rag is missing for my_model

        indicator = ScoreCardIndicator(
            id="combined_score",
            name="Combined Accuracy + Robustness",
            apply_to=ScoreCardFilter(test_id=["accuracy_rag", "robustness_rag"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_rag::hard_gate_pass_rate",
                    "ood": "robustness_rag::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="PASS", condition="greater_equal", threshold=0.8
                ),
            ],
        )

        eval_results = self.engine.evaluate_indicator([result_accuracy], indicator)

        assert len(eval_results) == 1
        assert eval_results[0].error is not None
        assert "Missing test results for SUT 'my_model'" in eval_results[0].error
        assert "robustness_rag" in eval_results[0].error
        # Error result should indicate which containers are specifically missing
        assert eval_results[0].test_ids == ["robustness_rag"]

    def test_evaluate_indicator_multi_test_id_no_containers(self):
        """Test error when no containers have any results."""
        # No test results provided at all
        indicator = ScoreCardIndicator(
            id="combined_score",
            name="Combined Accuracy + Robustness",
            apply_to=ScoreCardFilter(test_id=["accuracy_rag", "robustness_rag"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_rag::hard_gate_pass_rate",
                    "ood": "robustness_rag::ood_detection_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="PASS", condition="greater_equal", threshold=0.8
                ),
            ],
        )

        eval_results = self.engine.evaluate_indicator([], indicator)

        assert len(eval_results) == 1
        assert eval_results[0].error is not None
        assert "No test results found for any of test_ids" in eval_results[0].error
        # Error result should include all required containers
        assert eval_results[0].test_ids == ["accuracy_rag", "robustness_rag"]

    def test_cross_reference_rejected_with_single_test_id(self):
        """Test that cross-container references with single test_id are rejected at validation time."""
        from pydantic import ValidationError

        # This is now caught when creating the indicator, not during evaluation
        with pytest.raises(
            ValidationError,
            match="uses cross-container metric references.*apply_to.test_id is a single string",
        ):
            ScoreCardIndicator(
                id="combined_score",
                name="Combined Score",
                apply_to=ScoreCardFilter(test_id="accuracy_rag"),
                metric=MetricExpression(
                    expression="0.6 * acc + 0.4 * ood",
                    values={
                        "acc": "accuracy_rag::pass_rate",
                        "ood": "robustness_rag::ood_accuracy",  # Not allowed!
                    },
                ),
                assessment=[
                    AssessmentRule(
                        outcome="PASS", condition="greater_than", threshold=0.75
                    ),
                ],
            )

    def test_metric_expression_validator_invalid_syntax(self):
        """Test MetricExpression validator rejects invalid :: syntax."""
        # Empty test_id before ::
        with pytest.raises(ValueError, match="test_id before"):
            MetricExpression(
                expression="x",
                values={"x": "::metric_path"},
            )

        # Empty metric path after ::
        with pytest.raises(ValueError, match="metric path after"):
            MetricExpression(
                expression="x",
                values={"x": "test_id::"},
            )

    def test_single_test_id_with_cross_reference_rejected(self):
        """Test that cross-container references with single test_id are rejected at validation."""
        # Cross-container refs are only allowed when test_id is a list
        with pytest.raises(
            ValueError,
            match="uses cross-container metric references.*apply_to.test_id is a single string",
        ):
            ScoreCardIndicator(
                id="cross_ref_score",
                name="Cross Reference Score",
                apply_to=ScoreCardFilter(test_id="primary_test"),  # Single test_id
                metric=MetricExpression(
                    expression="0.5 * primary + 0.5 * cross",
                    values={
                        "primary": "primary_metric",
                        "cross": "cross_test::cross_metric",  # Not allowed!
                    },
                ),
                assessment=[
                    AssessmentRule(
                        outcome="PASS", condition="greater_equal", threshold=0.8
                    ),
                ],
            )


class TestScoreCardEvaluationResultTestIds:
    """Test ScoreCardEvaluationResult.test_ids field for single and multi-container indicators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self,
        test_name: str,
        test_id: str,
        test_results: dict,
        sut_name: str = "test_sut",
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, test_id, sut_name, "test_image")
        result.test_results = test_results
        result.success = True
        return result

    def test_evaluation_result_test_ids_single_container(self):
        """
        Verify test_ids field is populated correctly for single-container indicators.
        """
        result = self.create_test_result(
            "accuracy_test", "accuracy_test", {"score": 0.9}
        )

        indicator = ScoreCardIndicator(
            id="accuracy_only",
            name="Accuracy Only",
            apply_to=ScoreCardFilter(test_id="accuracy_test"),
            metric="score",
            assessment=[
                AssessmentRule(
                    outcome="PASS", condition="greater_equal", threshold=0.85
                ),
            ],
        )

        eval_results = self.engine.evaluate_indicator([result], indicator)

        assert len(eval_results) == 1
        eval_result = eval_results[0]

        # Verify test_ids field is populated correctly
        assert eval_result.test_ids == ["accuracy_test"]
        # Verify to_dict() includes test_ids
        result_dict = eval_result.to_dict()
        assert result_dict["test_ids"] == ["accuracy_test"]

    def test_evaluation_result_test_ids_multi_container(self):
        """
        Verify test_ids field contains all containers for multi-container indicators.
        """
        result_accuracy = self.create_test_result(
            "accuracy_test", "accuracy_test", {"score": 0.92}
        )
        result_robustness = self.create_test_result(
            "robustness_test", "robustness_test", {"ood_accuracy": 0.78}
        )

        indicator = ScoreCardIndicator(
            id="combined",
            name="Combined",
            apply_to=ScoreCardFilter(test_id=["accuracy_test", "robustness_test"]),
            metric=MetricExpression(
                expression="0.6 * acc + 0.4 * ood",
                values={
                    "acc": "accuracy_test::score",
                    "ood": "robustness_test::ood_accuracy",
                },
            ),
            assessment=[
                AssessmentRule(
                    outcome="PASS", condition="greater_equal", threshold=0.80
                ),
            ],
        )

        eval_results = self.engine.evaluate_indicator(
            [result_accuracy, result_robustness], indicator
        )

        assert len(eval_results) == 1
        eval_result = eval_results[0]

        # Verify test_ids contains all containers
        assert eval_result.test_ids == ["accuracy_test", "robustness_test"]
        # Verify to_dict() includes all containers
        result_dict = eval_result.to_dict()
        assert result_dict["test_ids"] == ["accuracy_test", "robustness_test"]


class TestScoreCardFilterTestIdsProperty:
    """Test ScoreCardFilter.test_ids property normalization."""

    def test_test_ids_normalizes_single_string(self):
        """Verify test_ids property converts single string to list."""
        filter_obj = ScoreCardFilter(test_id="single_test")
        assert filter_obj.test_ids == ["single_test"]
        # Also verify the original field is unchanged
        assert filter_obj.test_id == "single_test"

    def test_test_ids_normalizes_list(self):
        """Verify test_ids property returns list unchanged."""
        filter_obj = ScoreCardFilter(test_id=["test_a", "test_b", "test_c"])
        assert filter_obj.test_ids == ["test_a", "test_b", "test_c"]
        # Also verify the original field is unchanged
        assert filter_obj.test_id == ["test_a", "test_b", "test_c"]

    def test_test_ids_empty_list(self):
        """Verify test_ids handles empty list."""
        filter_obj = ScoreCardFilter(test_id=[])
        assert filter_obj.test_ids == []

    def test_test_ids_consistency_in_indicator(self):
        """Verify test_ids property works consistently when used in ScoreCardIndicator."""
        indicator = ScoreCardIndicator(
            id="test_indicator",
            name="Test Indicator",
            apply_to=ScoreCardFilter(test_id=["accuracy_test", "robustness_test"]),
            metric="some_metric",
            assessment=[
                AssessmentRule(
                    outcome="PASS", condition="greater_equal", threshold=0.8
                ),
            ],
        )

        # Verify that test_ids works correctly on the apply_to filter
        assert indicator.apply_to.test_ids == ["accuracy_test", "robustness_test"]


class TestValidateScoreCardTestIdsMultiContainer:
    """Test validate_scorecard_test_ids collects test_ids from multi-container indicators."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ScoreCardEngine()

    def create_test_result(
        self,
        test_name: str,
        test_id: str,
        test_results: dict,
        sut_name: str = "test_sut",
    ) -> TestExecutionResult:
        """Helper to create a TestExecutionResult for testing."""
        result = TestExecutionResult(test_name, test_id, sut_name, "test_image")
        result.test_results = test_results
        result.success = True
        return result

    def test_validate_scorecard_extracts_multi_test_ids(self):
        """
        Verify validate_scorecard_test_ids correctly processes indicators
        with multi-container apply_to.test_id lists without raising errors.
        """
        # Create test results for all containers
        test_results = [
            self.create_test_result("accuracy_test", "accuracy_test", {"score": 0.9}),
            self.create_test_result(
                "robustness_test", "robustness_test", {"ood_accuracy": 0.8}
            ),
        ]

        scorecard = ScoreCard(
            score_card_name="Test Score Card",
            indicators=[
                # Single-container indicator
                ScoreCardIndicator(
                    id="accuracy_only",
                    name="Accuracy Only",
                    apply_to=ScoreCardFilter(test_id="accuracy_test"),
                    metric="score",
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="greater_equal", threshold=0.85
                        ),
                    ],
                ),
                # Multi-container indicator with list
                ScoreCardIndicator(
                    id="combined",
                    name="Combined",
                    apply_to=ScoreCardFilter(
                        test_id=["accuracy_test", "robustness_test"]
                    ),
                    metric=MetricExpression(
                        expression="0.6 * acc + 0.4 * ood",
                        values={
                            "acc": "accuracy_test::score",
                            "ood": "robustness_test::ood_accuracy",
                        },
                    ),
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="greater_equal", threshold=0.80
                        ),
                    ],
                ),
            ],
        )

        # Should not raise - validation should pass for multi-container indicators
        # when all required test_ids are present
        self.engine.validate_scorecard_test_ids(test_results, scorecard)

    def test_validate_scorecard_with_partial_multi_container_test_ids(self):
        """
        Verify validate_scorecard_test_ids succeeds when at least one test_id from
        a multi-container indicator matches, even if others are missing.
        Note: Per-SUT validation happens during evaluation (evaluate_indicator).
        """
        # Only have accuracy_test, missing robustness_test
        test_results = [
            self.create_test_result("accuracy_test", "accuracy_test", {"score": 0.9}),
        ]

        scorecard = ScoreCard(
            score_card_name="Test Score Card",
            indicators=[
                ScoreCardIndicator(
                    id="combined",
                    name="Combined",
                    apply_to=ScoreCardFilter(
                        test_id=["accuracy_test", "robustness_test"]
                    ),
                    metric=MetricExpression(
                        expression="0.6 * acc + 0.4 * ood",
                        values={
                            "acc": "accuracy_test::score",
                            "ood": "robustness_test::ood_accuracy",
                        },
                    ),
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="greater_equal", threshold=0.80
                        ),
                    ],
                ),
            ],
        )

        # Should succeed - at least one test_id matches (accuracy_test)
        # Per-SUT validation (missing containers) happens in evaluate_indicator
        self.engine.validate_scorecard_test_ids(test_results, scorecard)

    def test_validate_scorecard_handles_multi_container_deduplication(self):
        """
        Verify validate_scorecard_test_ids correctly handles multiple indicators
        referencing the same multi-container set.
        """
        # Create test results for all containers
        test_results = [
            self.create_test_result("accuracy_test", "accuracy_test", {"score": 0.9}),
            self.create_test_result(
                "robustness_test", "robustness_test", {"ood_accuracy": 0.8}
            ),
            self.create_test_result("fairness_test", "fairness_test", {"bias": 0.85}),
        ]

        scorecard = ScoreCard(
            score_card_name="Test Score Card",
            indicators=[
                ScoreCardIndicator(
                    id="indicator1",
                    name="Indicator 1",
                    apply_to=ScoreCardFilter(
                        test_id=["accuracy_test", "robustness_test"]
                    ),
                    metric=MetricExpression(
                        expression="0.5 * acc + 0.5 * ood",
                        values={
                            "acc": "accuracy_test::score",
                            "ood": "robustness_test::ood_accuracy",
                        },
                    ),
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="greater_equal", threshold=0.80
                        ),
                    ],
                ),
                ScoreCardIndicator(
                    id="indicator2",
                    name="Indicator 2",
                    apply_to=ScoreCardFilter(
                        test_id=["accuracy_test", "fairness_test"]
                    ),
                    metric=MetricExpression(
                        expression="0.7 * acc + 0.3 * bias",
                        values={
                            "acc": "accuracy_test::score",
                            "bias": "fairness_test::bias",
                        },
                    ),
                    assessment=[
                        AssessmentRule(
                            outcome="PASS", condition="greater_equal", threshold=0.80
                        ),
                    ],
                ),
            ],
        )

        # Should not raise - all test_ids from both indicators are present
        self.engine.validate_scorecard_test_ids(test_results, scorecard)


class TestScoreCardEvaluationResultFactoryMethods:
    """Test ScoreCardEvaluationResult factory methods."""

    def test_as_success_creates_successful_result(self):
        """Test as_success factory method creates a proper success result."""
        from asqi.score_card_engine import ScoreCardEvaluationResult

        result = ScoreCardEvaluationResult.as_success(
            indicator_id="test_indicator",
            indicator_name="Test Indicator",
            test_ids="test_container",
            sut_name="my_model",
            outcome="PASS",
            metric_value=0.95,
            computed_value=True,
            details="Threshold met",
            description="Excellent performance",
            test_result_ids=["test_container_my_model"],
            report_paths=["/path/to/report.html"],
        )

        assert result.indicator_id == "test_indicator"
        assert result.indicator_name == "Test Indicator"
        assert result.test_ids == ["test_container"]  # Normalized to list
        assert result.sut_name == "my_model"
        assert result.outcome == "PASS"
        assert result.metric_value == 0.95
        assert result.computed_value is True
        assert result.details == "Threshold met"
        assert result.description == "Excellent performance"
        assert result.test_result_ids == ["test_container_my_model"]
        assert result.report_paths == ["/path/to/report.html"]
        assert result.error is None

    def test_as_success_with_multi_container_test_ids(self):
        """Test as_success with multiple containers."""
        from asqi.score_card_engine import ScoreCardEvaluationResult

        result = ScoreCardEvaluationResult.as_success(
            indicator_id="combined",
            indicator_name="Combined Indicator",
            test_ids=["container1", "container2"],
            sut_name="model_v2",
            outcome="A",
            metric_value=0.87,
            computed_value=True,
            details="Combined score",
            description="Good score",
        )

        assert result.test_ids == ["container1", "container2"]
        assert result.error is None

    def test_as_error_creates_error_result(self):
        """Test as_error factory method creates a proper error result."""
        from asqi.score_card_engine import ScoreCardEvaluationResult

        result = ScoreCardEvaluationResult.as_error(
            indicator_id="bad_indicator",
            indicator_name="Bad Indicator",
            test_ids="missing_container",
            error_message="Container not found",
            sut_name="my_model",
        )

        assert result.indicator_id == "bad_indicator"
        assert result.indicator_name == "Bad Indicator"
        assert result.test_ids == ["missing_container"]  # Normalized to list
        assert result.error == "Container not found"
        assert result.sut_name == "my_model"
        assert result.outcome is None
        assert result.metric_value is None

    def test_as_error_with_multi_container_test_ids(self):
        """Test as_error with multiple containers."""
        from asqi.score_card_engine import ScoreCardEvaluationResult

        result = ScoreCardEvaluationResult.as_error(
            indicator_id="multi_error",
            indicator_name="Multi Error",
            test_ids=["container1", "container2"],
            error_message="Multiple containers missing",
        )

        assert result.test_ids == ["container1", "container2"]
        assert result.error == "Multiple containers missing"
        assert result.sut_name is None
