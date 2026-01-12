import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rich.console import Console

from asqi.metric_expression import (
    MetricExpressionError,
    MetricExpressionEvaluator,
)
from asqi.schemas import (
    AuditResponses,
    AuditScoreCardIndicator,
    MetricExpression,
    ScoreCard,
    ScoreCardIndicator,
)
from asqi.workflow import TestExecutionResult
from asqi.validation import normalize_types

logger = logging.getLogger(__name__)

console = Console()


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

    def __init__(self, indicator_id: str, indicator_name: Optional[str], test_id: str):
        self.indicator_id = indicator_id
        self.indicator_name = indicator_name
        self.test_id = test_id
        self.outcome: Optional[str] = None
        self.metric_value: Optional[Any] = None
        self.test_result_id: Optional[str] = None
        self.sut_name: Optional[str] = None
        self.computed_value: Optional[Union[int, float, bool]] = None
        self.details: str = ""
        self.description: Optional[str] = None
        self.notes: Optional[str] = None
        self.error: Optional[str] = None
        self.report_paths: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "indicator_id": self.indicator_id,
            "indicator_name": self.indicator_name,
            "test_id": self.test_id,
            "sut_name": self.sut_name,
            "test_result_id": self.test_result_id,
            "metric_value": self.metric_value,
            "computed_value": self.computed_value,
            "details": self.details,
            "outcome": self.outcome,
            "description": self.description,
            "report_paths": self.report_paths,
            "audit_notes": self.notes,
            "error": self.error,
        }


class ScoreCardEngine:
    """Core score_card evaluation engine."""

    def filter_results_by_test_id(
        self, test_results: List[TestExecutionResult], target_test_id: str
    ) -> List[TestExecutionResult]:
        """Filter test results to only include those with the specified test id.

        Args:
            test_results: List of test execution results to filter
            target_test_id: Name of test to filter for

        Returns:
            Filtered list of test results matching the target test id
        """
        filtered = [
            result for result in test_results if result.test_id == target_test_id
        ]
        logger.debug(
            f"Filtered {len(test_results)} results to {len(filtered)} for test_id '{target_test_id}'"
        )
        return filtered

    def filter_results_by_test_and_type(
        self,
        test_results: List[TestExecutionResult],
        target_test_id: str,
        target_system_types: Optional[List[str]] = None,
    ) -> List[TestExecutionResult]:
        """
        Filter test results by test_id and optionally by system type.

        Args:
            test_results: List of test execution results
            target_test_id: Test ID to match
            target_system_types: Optional list of system types to match (None = all types)

        Returns:
            Filtered list of test results
        """
        filtered = []
        for result in test_results:
            # Must match test_id
            if result.test_id != target_test_id:
                continue

            # If system types specified, must match one of them
            if target_system_types is not None:
                if result.system_type not in target_system_types:
                    continue

            filtered.append(result)

        logger.debug(
            f"Filtered {len(test_results)} results to {len(filtered)} for test_id '{target_test_id}' "
            f"and system_types {target_system_types}"
        )
        return filtered

    def validate_scorecard_test_ids(
        self,
        test_results: List[TestExecutionResult],
        score_card: ScoreCard,
    ) -> None:
        """
        Check that the score card indicators are applicable to the available test results.

        Audit indicators do not reference test_ids and are ignored here.
        """
        # Only consider non audit indicators
        metric_indicators = [
            ind for ind in score_card.indicators if isinstance(ind, ScoreCardIndicator)
        ]

        # If there are only audit indicators, skip validation
        if not metric_indicators:
            return

        results_test_ids = {result.test_id for result in test_results}
        score_card_test_ids = {ind.apply_to.test_id for ind in metric_indicators}

        if not results_test_ids & score_card_test_ids:
            raise ValueError(
                "Score card indicators don't match any test ids in the test results"
            )

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
                        f"No test_results data available for {result.test_id}"
                    )
                    continue

                # Use nested value extraction
                value, error = get_nested_value(result.test_results, metric_path)

                if error is None:
                    values.append(value)
                else:
                    logger.warning(
                        f"Failed to extract metric '{metric_path}' from test result for {result.test_id}: {error}"
                    )

            except Exception as e:
                logger.warning(
                    f"Unexpected error extracting metric '{metric_path}' from test result for {result.test_id}: {e}"
                )

        return values

    def resolve_metric_or_expression(
        self,
        test_result: TestExecutionResult,
        metric_config: Union[str, MetricExpression],
    ) -> Tuple[Optional[Union[int, float]], Optional[str]]:
        """
        Resolve a metric configuration (simple path or expression object).

        Args:
            test_result: Test execution result containing metric data
            metric_config: Either a simple metric path string or MetricExpression object

        Returns:
            Tuple of (resolved_value, error_message). If successful, error is None.
        """
        # Handle simple string path (backward compatible)
        if isinstance(metric_config, str):
            return get_nested_value(test_result.test_results, metric_config)

        # Handle MetricExpression object
        evaluator = MetricExpressionEvaluator()

        try:
            # Resolve all declared metric values
            metric_values: Dict[str, Union[int, float]] = {}
            for var_name, metric_path in metric_config.values.items():
                value, error = get_nested_value(test_result.test_results, metric_path)

                if error is not None:
                    return (
                        None,
                        f"Failed to resolve metric '{metric_path}' for variable '{var_name}': {error}",
                    )

                # Validate it's numeric for expression evaluation
                if not isinstance(value, (int, float)):
                    return (
                        None,
                        f"Metric '{metric_path}' (variable '{var_name}') has non-numeric value {type(value).__name__}: {value}",
                    )

                # Use the variable name directly from the dict key
                metric_values[var_name] = value

            # Evaluate the expression with resolved values
            result = evaluator.evaluate_expression(
                metric_config.expression, metric_values
            )
            return result, None

        except MetricExpressionError as e:
            return None, f"Expression evaluation error: {e}"
        except Exception as e:
            return None, f"Unexpected error evaluating expression: {e}"

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
            target_types = None
            # For backward compatibility - score cards without target_system_type match all types
            if (
                hasattr(indicator.apply_to, "target_system_type")
                and indicator.apply_to.target_system_type
            ):
                target_types = normalize_types(indicator.apply_to.target_system_type)

            # Filter results by test id and optionally by system type
            filtered_results = self.filter_results_by_test_and_type(
                test_results, indicator.apply_to.test_id, target_types
            )

            if not filtered_results:
                # Create a single error result when no tests match
                error_result = ScoreCardEvaluationResult(
                    indicator.id, indicator.name, indicator.apply_to.test_id
                )

                # Check if test_id exists but with different system types
                test_id_matches = [
                    r for r in test_results if r.test_id == indicator.apply_to.test_id
                ]

                if test_id_matches and target_types:
                    # Test ID exists but filtered out by system type
                    available_types = ", ".join(
                        set(r.system_type or "unknown" for r in test_id_matches)
                    )
                    target_types_str = ", ".join(target_types)
                    error_result.error = (
                        f"No test results found for test_id '{indicator.apply_to.test_id}' "
                        f"with system type(s) [{target_types_str}]. "
                        f"Test '{indicator.apply_to.test_id}' has results for system type(s): {available_types}"
                    )
                elif test_id_matches:
                    # Test ID exists but all filtered out (shouldn't happen without target_types)
                    error_result.error = f"Test '{indicator.apply_to.test_id}' found but no results matched filters"
                else:
                    # Test ID doesn't exist at all
                    available_tests = (
                        ", ".join(set(r.test_id for r in test_results))
                        if test_results
                        else "none"
                    )
                    error_result.error = (
                        f"No test results found for test_id '{indicator.apply_to.test_id}'. "
                        f"Available tests: {available_tests}"
                    )
                return [error_result]

            # Evaluate each individual test result
            for test_result in filtered_results:
                eval_result = ScoreCardEvaluationResult(
                    indicator.id, indicator.name, indicator.apply_to.test_id
                )
                eval_result.sut_name = test_result.sut_name
                eval_result.test_result_id = (
                    f"{test_result.test_id}_{test_result.sut_name}"
                )
                test_reports = test_result.generated_reports or []
                requested_reports = indicator.display_reports or []
                eval_result.report_paths = [
                    str(report.report_path)
                    for report in test_reports
                    if report.report_name in requested_reports and report.report_path
                ]

                try:
                    # Resolve metric value (handles both simple paths and expressions)
                    metric_value, error = self.resolve_metric_or_expression(
                        test_result, indicator.metric
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
                                    eval_result.description = (
                                        assessment_rule.description
                                    )
                                    logger.debug(
                                        f"score_card indicator id '{indicator.id}' for test id '{test_result.test_id}' (system under test: {test_result.sut_name}) evaluated to '{assessment_rule.outcome}': {description}"
                                    )
                                    break

                            except Exception as e:
                                logger.error(
                                    f"Error evaluating assessment rule for indicator id '{indicator.id}': {e}"
                                )
                                eval_result.error = str(e)
                                break

                        # If no rule matched, that's an error condition
                        if eval_result.outcome is None and eval_result.error is None:
                            eval_result.error = (
                                "No assessment rule conditions were satisfied"
                            )

                    else:
                        eval_result.error = f"Failed to extract metric '{indicator.metric}' from test result for '{test_result.test_id}': {error}"

                except Exception as e:
                    logger.error(
                        f"Error evaluating test result for indicator id '{indicator.id}': {e}"
                    )
                    eval_result.error = str(e)

                results.append(eval_result)

        except Exception as e:
            logger.error(f"Error evaluating indicator id '{indicator.id}': {e}")
            error_result = ScoreCardEvaluationResult(
                indicator.id, indicator.name, indicator.apply_to.test_id
            )
            error_result.error = str(e)
            results.append(error_result)

        return results

    def evaluate_audit_indicator(
        self,
        indicator: AuditScoreCardIndicator,
        audit_responses: Optional[AuditResponses] = None,
        available_suts: Optional[Set[str]] = None,
    ) -> List[ScoreCardEvaluationResult]:
        """
        Convert manual audit responses for a single audit indicator into evaluation results.
        """
        results: List[ScoreCardEvaluationResult] = []
        available_sut_set = set(s for s in (available_suts or []) if s is not None)

        # No responses object at all
        if audit_responses is None:
            result = ScoreCardEvaluationResult(
                indicator_id=indicator.id,
                indicator_name=indicator.name,
                test_id="audit",
            )
            result.error = (
                f"No audit responses provided for indicator_id '{indicator.id}'"
            )
            results.append(result)
            return results

        # Filter responses for this specific indicator
        matching_responses = [
            r for r in audit_responses.responses if r.indicator_id == indicator.id
        ]

        if not matching_responses:
            # We have an audit_responses file, but nothing for this id
            result = ScoreCardEvaluationResult(
                indicator_id=indicator.id,
                indicator_name=indicator.name,
                test_id="audit",
            )
            result.error = f"No audit response found for indicator_id '{indicator.id}'"
            results.append(result)
            return results
        # Detect duplicate responses for the same (indicator, sut)
        seen_keys = set()
        duplicate_keys = set()
        for resp in matching_responses:
            key = (indicator.id, resp.sut_name)
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)

        if duplicate_keys:
            result = ScoreCardEvaluationResult(
                indicator_id=indicator.id,
                indicator_name=indicator.name,
                test_id="audit",
            )
            result.error = (
                f"Duplicate audit responses for indicator '{indicator.id}' and sut(s): "
                f"{sorted({k[1] for k in duplicate_keys})}"
            )
            results.append(result)
            return results

        per_system_responses = [r for r in matching_responses if r.sut_name is not None]

        if per_system_responses:
            if len(per_system_responses) != len(matching_responses):
                result = ScoreCardEvaluationResult(
                    indicator_id=indicator.id,
                    indicator_name=indicator.name,
                    test_id="audit",
                )
                result.error = f"Audit indicator '{indicator.id}' cannot mix global and per-system responses"
                results.append(result)
                return results

            if available_sut_set:
                invalid_responses = [
                    r
                    for r in per_system_responses
                    if r.sut_name not in available_sut_set
                ]

                if invalid_responses:
                    for resp in invalid_responses:
                        eval_result = ScoreCardEvaluationResult(
                            indicator_id=indicator.id,
                            indicator_name=indicator.name,
                            test_id="audit",
                        )
                        eval_result.sut_name = resp.sut_name
                        eval_result.test_result_id = None
                        eval_result.metric_value = None
                        eval_result.computed_value = None
                        eval_result.details = "Manual audit indicator response"
                        eval_result.outcome = resp.selected_outcome
                        eval_result.notes = resp.notes
                        eval_result.error = f"'{resp.sut_name}' is not a valid system under test for this evaluation"
                        results.append(eval_result)

                    return results

                missing_suts = available_sut_set - {
                    r.sut_name for r in per_system_responses
                }

                if missing_suts:
                    result = ScoreCardEvaluationResult(
                        indicator_id=indicator.id,
                        indicator_name=indicator.name,
                        test_id="audit",
                    )
                    result.error = f"Audit indicator '{indicator.id}' requires responses for all systems: missing {sorted(missing_suts)}"
                    results.append(result)
                    return results

        # Build lookup: outcome -> description from the scorecard definition
        outcome_to_description: Dict[str, Optional[str]] = {}
        for rule in indicator.assessment:
            outcome_to_description[rule.outcome] = rule.description

        valid_outcomes = set(outcome_to_description.keys())

        # One ScoreCardEvaluationResult per audit response (often just 1)
        for resp in matching_responses:
            eval_result = ScoreCardEvaluationResult(
                indicator_id=indicator.id,
                indicator_name=indicator.name,
                test_id="audit",
            )
            eval_result.sut_name = resp.sut_name
            eval_result.test_result_id = None
            eval_result.metric_value = None
            eval_result.computed_value = None
            eval_result.details = "Manual audit indicator response"
            eval_result.outcome = resp.selected_outcome
            eval_result.notes = resp.notes

            # Attach description if we have one for that outcome
            if resp.selected_outcome not in valid_outcomes:
                eval_result.error = (
                    f"Invalid selected_outcome '{resp.selected_outcome}' for indicator_id "
                    f"'{indicator.id}'. Allowed outcomes: {sorted(valid_outcomes)}"
                )

                eval_result.outcome = resp.selected_outcome
                results.append(eval_result)
                continue

            # valid outcome
            eval_result.outcome = resp.selected_outcome
            eval_result.description = outcome_to_description[resp.selected_outcome]

            results.append(eval_result)

        return results

    def evaluate_scorecard(
        self,
        test_results: List[TestExecutionResult],
        score_card: ScoreCard,
        audit_responses_data: Optional[AuditResponses] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate a complete grading score_card against test results.

        Args:
            test_results: List of test execution results to evaluate
            score_card: Complete score card configuration
            audit_responses_data: User-provided audit responses

        Returns:
            List of evaluation result dictionaries

        Raises:
            ValueError: If no indicators match any test ids in the test results
        """
        self.validate_scorecard_test_ids(test_results, score_card)

        all_test_evaluations = []
        sut_names = {r.sut_name for r in test_results if r.sut_name is not None}

        for indicator in score_card.indicators:
            # non-audit indicators
            if isinstance(indicator, ScoreCardIndicator):
                indicator_results = self.evaluate_indicator(test_results, indicator)
                all_test_evaluations.extend(r.to_dict() for r in indicator_results)
                continue

            # audit indicators
            if isinstance(indicator, AuditScoreCardIndicator):
                audit_results = self.evaluate_audit_indicator(
                    indicator=indicator,
                    audit_responses=audit_responses_data,
                    available_suts=sut_names,
                )
                all_test_evaluations.extend(r.to_dict() for r in audit_results)
                continue

        return all_test_evaluations
