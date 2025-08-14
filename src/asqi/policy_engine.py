import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from asqi.schemas import GradingPolicy, PolicyIndicator
from asqi.workflow import TestExecutionResult

logger = logging.getLogger(__name__)


class PolicyEvaluationResult:
    """Result of evaluating a single policy indicator."""

    def __init__(self, indicator_name: str, test_name: str):
        self.indicator_name = indicator_name
        self.test_name = test_name
        self.outcome: Optional[str] = None
        self.metric_value: Optional[Any] = None
        self.test_result_id: Optional[str] = None
        self.sut_name: Optional[str] = None
        self.computed_value: Optional[Union[int, float, bool]] = None
        self.details: str = ""
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "indicator_name": self.indicator_name,
            "test_name": self.test_name,
            "sut_name": self.sut_name,
            "test_result_id": self.test_result_id,
            "outcome": self.outcome,
            "metric_value": self.metric_value,
            "computed_value": self.computed_value,
            "details": self.details,
            "error": self.error,
        }


class PolicyEngine:
    """Core policy evaluation engine."""

    def filter_results_by_test_name(
        self, test_results: List[TestExecutionResult], target_test_name: str
    ) -> List[TestExecutionResult]:
        """Filter test results to only include those with the specified test name."""
        filtered = [
            result for result in test_results if result.test_name == target_test_name
        ]
        logger.debug(
            f"Filtered {len(test_results)} results to {len(filtered)} for test_name '{target_test_name}'"
        )
        return filtered

    def extract_metric_values(
        self, test_results: List[TestExecutionResult], metric_path: str
    ) -> List[Any]:
        """Extract metric values from test results using the specified path."""
        values = []

        for result in test_results:
            try:
                # For now, support simple key access into test_results dict
                # Future enhancement could support JSONPath for nested access
                if metric_path in result.test_results:
                    values.append(result.test_results[metric_path])
                else:
                    logger.warning(
                        f"Metric '{metric_path}' not found in test result for {result.test_name}"
                    )
            except (AttributeError, KeyError) as e:
                logger.warning(
                    f"Failed to extract metric '{metric_path}' from test result: {e}"
                )

        return values

    def apply_condition_to_value(
        self, value: Any, condition: str, threshold: Optional[Union[int, float]] = None
    ) -> Tuple[bool, str]:
        """
        Apply the specified condition to a single value.

        Returns:
            Tuple of (condition_met, description)
        """

        if condition in [
            "equal_to",
            "greater_than",
            "less_than",
            "greater_equal",
            "less_equal",
        ]:
            if threshold is None:
                raise ValueError(f"{condition} condition requires threshold")

            # Handle boolean values for equal_to condition
            if condition == "equal_to":
                if isinstance(threshold, bool) and isinstance(value, bool):
                    result = value == threshold
                    return result, f"Value {value} equals {threshold}: {result}"
                # Handle numeric comparison
                try:
                    numeric_value = float(value)
                    numeric_threshold = float(threshold)
                    result = numeric_value == numeric_threshold
                    return (
                        result,
                        f"Value {numeric_value} equals {numeric_threshold}: {result}",
                    )
                except (ValueError, TypeError):
                    result = value == threshold
                    return result, f"Value {value} equals {threshold}: {result}"

            # Other comparison conditions require numeric values
            try:
                numeric_value = float(value)
                numeric_threshold = float(threshold)

                if condition == "greater_than":
                    result = numeric_value > numeric_threshold
                elif condition == "less_than":
                    result = numeric_value < numeric_threshold
                elif condition == "greater_equal":
                    result = numeric_value >= numeric_threshold
                elif condition == "less_equal":
                    result = numeric_value <= numeric_threshold
                else:
                    result = False  # Default assignment if none of the above matches

                return (
                    result,
                    f"Value {numeric_value} {condition} {numeric_threshold}: {result}",
                )

            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot apply {condition} to non-numeric value: {value}"
                )

        # Logical conditions (deprecated for individual evaluation, but keeping for compatibility)
        elif condition == "all_true":
            result = bool(value)
            return result, f"Value {value} is truthy: {result}"

        elif condition == "any_false":
            result = not bool(value)
            return result, f"Value {value} is falsy: {result}"

        else:
            raise ValueError(f"Unknown condition: {condition}")

    def evaluate_indicator(
        self, test_results: List[TestExecutionResult], indicator: PolicyIndicator
    ) -> List[PolicyEvaluationResult]:
        """Evaluate a single policy indicator against individual test results."""
        results = []

        try:
            # Filter results by test name
            filtered_results = self.filter_results_by_test_name(
                test_results, indicator.apply_to.test_name
            )

            if not filtered_results:
                # Create a single error result when no tests match
                error_result = PolicyEvaluationResult(
                    indicator.name, indicator.apply_to.test_name
                )
                error_result.error = f"No test results found for test_name '{indicator.apply_to.test_name}'"
                return [error_result]

            # Evaluate each individual test result
            for test_result in filtered_results:
                eval_result = PolicyEvaluationResult(
                    indicator.name, indicator.apply_to.test_name
                )
                eval_result.sut_name = test_result.sut_name
                eval_result.test_result_id = (
                    f"{test_result.test_name}_{test_result.sut_name}"
                )

                try:
                    # Extract metric value from this specific test
                    if indicator.metric in test_result.test_results:
                        metric_value = test_result.test_results[indicator.metric]
                        eval_result.metric_value = metric_value

                        # Evaluate each assessment rule to find the first match
                        for assessment_rule in indicator.assessment:
                            try:
                                condition_met, description = (
                                    self.apply_condition_to_value(
                                        metric_value,
                                        assessment_rule.condition,
                                        assessment_rule.threshold,
                                    )
                                )
                                eval_result.computed_value = condition_met
                                eval_result.details = description

                                # If this rule's condition is satisfied, assign the outcome
                                if condition_met:
                                    eval_result.outcome = assessment_rule.outcome
                                    logger.info(
                                        f"Policy indicator '{indicator.name}' for test '{test_result.test_name}' (SUT: {test_result.sut_name}) evaluated to '{assessment_rule.outcome}': {description}"
                                    )
                                    break

                            except Exception as e:
                                logger.error(
                                    f"Error evaluating assessment rule for indicator '{indicator.name}': {e}"
                                )
                                eval_result.error = str(e)
                                break

                        # If no rule matched, that's an error condition
                        if eval_result.outcome is None and eval_result.error is None:
                            eval_result.error = (
                                "No assessment rule conditions were satisfied"
                            )

                    else:
                        eval_result.error = (
                            f"Metric '{indicator.metric}' not found in test result"
                        )

                except Exception as e:
                    logger.error(
                        f"Error evaluating test result for indicator '{indicator.name}': {e}"
                    )
                    eval_result.error = str(e)

                results.append(eval_result)

        except Exception as e:
            logger.error(f"Error evaluating indicator '{indicator.name}': {e}")
            error_result = PolicyEvaluationResult(
                indicator.name, indicator.apply_to.test_name
            )
            error_result.error = str(e)
            results.append(error_result)

        return results

    def evaluate_policy(
        self, test_results: List[TestExecutionResult], policy: GradingPolicy
    ) -> List[Dict[str, Any]]:
        """Evaluate a complete grading policy against test results."""
        all_test_evaluations = []

        for indicator in policy.indicators:
            indicator_results = self.evaluate_indicator(test_results, indicator)

            for result in indicator_results:
                all_test_evaluations.append(result.to_dict())

        logger.info(
            f"Policy '{policy.policy_name}' evaluation completed with {len(all_test_evaluations)} individual evaluations"
        )

        return all_test_evaluations
