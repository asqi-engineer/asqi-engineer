import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from asqi.schemas import ScoreCard, ScoreCardIndicator
from asqi.workflow import TestExecutionResult

logger = logging.getLogger(__name__)


def _validate_bracket_syntax(path: str) -> None:
    """Validate that all bracket notation in the path is properly formatted.

    Args:
        path: Metric path to validate

    Raises:
        ValueError: If bracket syntax is invalid
    """
    # Find all bracket sequences
    bracket_sequences = re.findall(r"\[([^[\]]*)\]", path)

    for seq in bracket_sequences:
        # Check if it's properly quoted
        if not (seq.startswith('"') and seq.endswith('"')) and not (
            seq.startswith("'") and seq.endswith("'")
        ):
            raise ValueError(
                f"Invalid bracket syntax: '[{seq}]' must be quoted. "
                f"Use ['key'] or [\"key\"] format. Examples: "
                f"probe_results[\"encoding.InjectHex\"] or data['key.with.dots']"
            )

        # Check for empty content
        content = seq[1:-1]  # Remove quotes
        if not content:
            raise ValueError(
                f"Empty bracket content not allowed: '[{seq}]'. "
                f"Bracket notation must contain a non-empty key."
            )

    # Check for unmatched opening brackets
    open_brackets = path.count("[")
    close_brackets = path.count("]")
    if open_brackets != close_brackets:
        raise ValueError(
            f"Unmatched brackets in metric path: '{path}'. "
            f"Found {open_brackets} '[' and {close_brackets} ']'. "
            f"Each '[' must have a matching ']'."
        )


def _tokenize_metric_path(path: str) -> List[str]:
    """Tokenize a metric path into individual keys.

    Args:
        path: Pre-validated metric path

    Returns:
        List of keys to traverse
    """
    keys = []
    current_pos = 0

    while current_pos < len(path):
        # Look for the next bracket or end of string
        bracket_start = path.find("[", current_pos)

        if bracket_start == -1:
            # No more brackets, handle remaining as dot-separated
            remaining = path[current_pos:]
            if remaining:
                # Split by dots and filter out empty strings
                dot_parts = [p for p in remaining.split(".") if p]
                keys.extend(dot_parts)
            break

        # Handle the portion before the bracket
        before_bracket = path[current_pos:bracket_start]
        if before_bracket:
            # Remove trailing dot if present
            if before_bracket.endswith("."):
                before_bracket = before_bracket[:-1]
            # Split by dots and filter out empty strings
            if before_bracket:
                dot_parts = [p for p in before_bracket.split(".") if p]
                keys.extend(dot_parts)

        # Find the matching closing bracket
        bracket_end = path.find("]", bracket_start)
        if bracket_end == -1:
            # This shouldn't happen as validation should catch it
            raise ValueError(f"Unmatched '[' at position {bracket_start}")

        # Extract the key from within brackets (including quotes)
        bracket_content = path[bracket_start + 1 : bracket_end]

        # Remove quotes from bracket content
        if (bracket_content.startswith('"') and bracket_content.endswith('"')) or (
            bracket_content.startswith("'") and bracket_content.endswith("'")
        ):
            key = bracket_content[1:-1]
            keys.append(key)
        else:
            # This shouldn't happen as validation should catch it
            raise ValueError(f"Invalid bracket content: {bracket_content}")

        # Move past the bracket and any following dot
        current_pos = bracket_end + 1
        if current_pos < len(path) and path[current_pos] == ".":
            current_pos += 1

    return keys


def parse_metric_path(path: str) -> List[str]:
    """Parse a metric path supporting both dot notation and bracket notation.

    Examples:
        'success' -> ['success']
        'vulnerability_stats.Toxicity.overall_pass_rate' -> ['vulnerability_stats', 'Toxicity', 'overall_pass_rate']
        'probe_results["encoding.InjectHex"]["encoding.DecodeMatch"].passed' -> ['probe_results', 'encoding.InjectHex', 'encoding.DecodeMatch', 'passed']
        'probe_results["encoding.InjectHex"].total_attempts' -> ['probe_results', 'encoding.InjectHex', 'total_attempts']

    Args:
        path: Metric path string to parse

    Returns:
        List of keys to traverse

    Raises:
        ValueError: If path contains invalid syntax
    """
    if not path:
        raise ValueError("Metric path cannot be empty")
    if not path.strip():
        raise ValueError("Metric path cannot be only whitespace")

    if "[" in path or "]" in path:
        _validate_bracket_syntax(path)

    keys = _tokenize_metric_path(path)

    if not keys:
        raise ValueError(f"Invalid metric path resulted in no keys: '{path}'")

    return keys


def get_nested_value(data: Dict[str, Any], path: str) -> Tuple[Any, Optional[str]]:
    """Extract a nested value from a dictionary using dot/bracket notation.

    Args:
        data: Dictionary to extract value from
        path: Path to the nested value (e.g., 'a.b.c' or 'a["key.with.dots"].c')

    Returns:
        Tuple of (value, error_message). If successful, error_message is None.
        If failed, value is None and error_message describes the issue.
    """
    try:
        keys = parse_metric_path(path)
    except ValueError as e:
        return None, str(e)

    current = data
    traversed_path = []

    for key in keys:
        traversed_path.append(key)

        if not isinstance(current, dict):
            path_so_far = ".".join(traversed_path[:-1])
            return (
                None,
                f"Cannot access key '{key}' at path '{path_so_far}' - value is not a dictionary: {type(current).__name__}",
            )

        if key not in current:
            available_keys = list(current.keys()) if current else []
            path_so_far = (
                ".".join(traversed_path[:-1]) if len(traversed_path) > 1 else "root"
            )
            return (
                None,
                f"Key '{key}' not found at path '{path_so_far}'. Available keys: {available_keys}",
            )

        current = current[key]

    return current, None


class ScoreCardEvaluationResult:
    """Result of evaluating a single score_card indicator."""

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


class ScoreCardEngine:
    """Core score_card evaluation engine."""

    def filter_results_by_test_name(
        self, test_results: List[TestExecutionResult], target_test_name: str
    ) -> List[TestExecutionResult]:
        """Filter test results to only include those with the specified test name.

        Args:
            test_results: List of test execution results to filter
            target_test_name: Name of test to filter for

        Returns:
            Filtered list of test results matching the target test name
        """
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
        """Extract metric values from test results using the specified path.

        Supports both flat and nested metric access:
        - Flat: 'success', 'score'
        - Nested: 'vulnerability_stats.Toxicity.overall_pass_rate'
        - Bracket notation: 'probe_results["encoding.InjectHex"]["encoding.DecodeMatch"].passed'

        Args:
            test_results: List of test execution results
            metric_path: Path to metric within test results (supports dot and bracket notation)

        Returns:
            List of extracted metric values
        """
        values = []

        for result in test_results:
            try:
                if not result.test_results:
                    logger.warning(
                        f"No test_results data available for {result.test_name}"
                    )
                    continue

                # Use nested value extraction
                value, error = get_nested_value(result.test_results, metric_path)

                if error is None:
                    values.append(value)
                else:
                    logger.warning(
                        f"Failed to extract metric '{metric_path}' from test result for {result.test_name}: {error}"
                    )

            except Exception as e:
                logger.warning(
                    f"Unexpected error extracting metric '{metric_path}' from test result for {result.test_name}: {e}"
                )

        return values

    def apply_condition_to_value(
        self, value: Any, condition: str, threshold: Optional[Union[int, float]] = None
    ) -> Tuple[bool, str]:
        """
        Apply the specified condition to a single value.

        Args:
            value: Value to evaluate
            condition: Condition to apply (e.g., 'equal_to', 'greater_than')
            threshold: Threshold value for comparison (required for most conditions)

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
        self, test_results: List[TestExecutionResult], indicator: ScoreCardIndicator
    ) -> List[ScoreCardEvaluationResult]:
        """Evaluate a single score_card indicator against individual test results.

        Args:
            test_results: List of test execution results to evaluate
            indicator: Score card indicator configuration

        Returns:
            List of evaluation results for each matching test
        """
        results = []

        try:
            # Filter results by test name
            filtered_results = self.filter_results_by_test_name(
                test_results, indicator.apply_to.test_name
            )

            if not filtered_results:
                # Create a single error result when no tests match
                error_result = ScoreCardEvaluationResult(
                    indicator.name, indicator.apply_to.test_name
                )
                available_tests = (
                    ", ".join(set(r.test_name for r in test_results))
                    if test_results
                    else "none"
                )
                error_result.error = f"No test results found for test_name '{indicator.apply_to.test_name}'. Available tests: {available_tests}"
                return [error_result]

            # Evaluate each individual test result
            for test_result in filtered_results:
                eval_result = ScoreCardEvaluationResult(
                    indicator.name, indicator.apply_to.test_name
                )
                eval_result.sut_name = test_result.sut_name
                eval_result.test_result_id = (
                    f"{test_result.test_name}_{test_result.sut_name}"
                )

                try:
                    # Extract metric value from this specific test using nested path support
                    metric_value, error = get_nested_value(
                        test_result.test_results, indicator.metric
                    )

                    if error is None:
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
                                    logger.debug(
                                        f"score_card indicator '{indicator.name}' for test '{test_result.test_name}' (system under test: {test_result.sut_name}) evaluated to '{assessment_rule.outcome}': {description}"
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
                        eval_result.error = f"Failed to extract metric '{indicator.metric}' from test result for '{test_result.test_name}': {error}"

                except Exception as e:
                    logger.error(
                        f"Error evaluating test result for indicator '{indicator.name}': {e}"
                    )
                    eval_result.error = str(e)

                results.append(eval_result)

        except Exception as e:
            logger.error(f"Error evaluating indicator '{indicator.name}': {e}")
            error_result = ScoreCardEvaluationResult(
                indicator.name, indicator.apply_to.test_name
            )
            error_result.error = str(e)
            results.append(error_result)

        return results

    def evaluate_scorecard(
        self, test_results: List[TestExecutionResult], score_card: ScoreCard
    ) -> List[Dict[str, Any]]:
        """Evaluate a complete grading score_card against test results.

        Args:
            test_results: List of test execution results to evaluate
            score_card: Complete score card configuration

        Returns:
            List of evaluation result dictionaries
        """
        all_test_evaluations = []

        for indicator in score_card.indicators:
            indicator_results = self.evaluate_indicator(test_results, indicator)

            for result in indicator_results:
                all_test_evaluations.append(result.to_dict())

        return all_test_evaluations
