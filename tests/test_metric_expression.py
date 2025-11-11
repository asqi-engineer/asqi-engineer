"""Tests for metric expression evaluation."""

import pytest

from asqi.metric_expression import (
    MetricExpressionError,
    MetricExpressionEvaluator,
)


class TestMetricExpressionEvaluator:
    """Tests for MetricExpressionEvaluator class."""

    def test_parse_expression_valid(self):
        """Test parsing of valid expressions."""
        evaluator = MetricExpressionEvaluator()

        # Should not raise
        evaluator.parse_expression("0.5 * a + 0.5 * b")
        evaluator.parse_expression("min(x, y, z)")
        evaluator.parse_expression("(a + b) / 2")

    def test_parse_expression_invalid_syntax(self):
        """Test that invalid syntax raises appropriate error."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="Invalid expression syntax"):
            evaluator.parse_expression("0.5 * a +")  # Incomplete

        with pytest.raises(MetricExpressionError, match="Invalid expression syntax"):
            evaluator.parse_expression("a b c")  # Invalid syntax

    def test_parse_expression_disallowed_operations(self):
        """Test that disallowed operations are rejected."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="not allowed"):
            evaluator.parse_expression("pow(a, 2)")  # pow not in allowed list

        with pytest.raises(MetricExpressionError, match="Unsupported operator"):
            evaluator.parse_expression("a ** 2")  # Exponentiation not allowed

    def test_evaluate_expression_simple_arithmetic(self):
        """Test evaluation of simple arithmetic expressions."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression(
            "0.5 * a + 0.5 * b", {"a": 0.8, "b": 0.6}
        )
        assert result == pytest.approx(0.7)

    def test_evaluate_expression_with_min(self):
        """Test evaluation with min function."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression(
            "min(a, b, c)", {"a": 0.9, "b": 0.7, "c": 0.8}
        )
        assert result == 0.7

    def test_evaluate_expression_with_max(self):
        """Test evaluation with max function."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression(
            "max(a, b, c)", {"a": 0.9, "b": 0.7, "c": 0.8}
        )
        assert result == 0.9

    def test_evaluate_expression_with_avg(self):
        """Test evaluation with avg function."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression(
            "avg(a, b, c)", {"a": 0.6, "b": 0.8, "c": 1.0}
        )
        assert result == pytest.approx(0.8)

    def test_evaluate_expression_complex(self):
        """Test evaluation of complex expressions."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression(
            "min(0.7 * accuracy + 0.3 * relevance, 1.0)",
            {"accuracy": 0.9, "relevance": 0.8},
        )
        assert result == pytest.approx(0.87)

    def test_evaluate_expression_missing_metric(self):
        """Test that missing metrics raise appropriate error."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="not found"):
            evaluator.evaluate_expression("a + b", {"a": 0.5})  # b is missing

    def test_evaluate_expression_division_by_zero(self):
        """Test that division by zero raises appropriate error."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="Division by zero"):
            evaluator.evaluate_expression("a / b", {"a": 1.0, "b": 0.0})

    def test_evaluate_expression_nested_metrics(self):
        """Test evaluation with nested metric paths.

        Note: Nested paths with dots in expressions need underscores as Python identifiers.
        The actual metric path resolution happens in score_card_engine.
        """
        evaluator = MetricExpressionEvaluator()

        # Use underscore notation for nested paths in expressions
        result = evaluator.evaluate_expression(
            "stats_pass_rate + stats_fail_rate",
            {"stats_pass_rate": 0.7, "stats_fail_rate": 0.3},
        )
        assert result == pytest.approx(1.0)

    def test_evaluate_expression_parentheses(self):
        """Test that parentheses work correctly."""
        evaluator = MetricExpressionEvaluator()

        result1 = evaluator.evaluate_expression("(a + b) * c", {"a": 1, "b": 2, "c": 3})
        assert result1 == 9

        result2 = evaluator.evaluate_expression("a + b * c", {"a": 1, "b": 2, "c": 3})
        assert result2 == 7

    def test_evaluate_expression_negative_numbers(self):
        """Test expressions with negative numbers."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression("-a + b", {"a": 0.3, "b": 0.5})
        assert result == pytest.approx(0.2)

    def test_evaluate_expression_subtraction(self):
        """Test subtraction operations."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression("a - b", {"a": 0.9, "b": 0.2})
        assert result == pytest.approx(0.7)

    def test_evaluate_expression_multiplication(self):
        """Test multiplication operations."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression("a * b * c", {"a": 2, "b": 3, "c": 4})
        assert result == 24

    def test_evaluate_expression_division(self):
        """Test division operations."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression("a / b", {"a": 10, "b": 4})
        assert result == pytest.approx(2.5)


class TestExpressionSafety:
    """Tests to ensure expression evaluator is secure."""

    def test_no_imports_allowed(self):
        """Test that imports are not allowed."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError):
            evaluator.parse_expression("import os")

    def test_no_attribute_access(self):
        """Test that attribute access on objects is blocked."""
        evaluator = MetricExpressionEvaluator()

        # This should fail during evaluation since __builtins__ is restricted
        with pytest.raises(MetricExpressionError):
            evaluator.evaluate_expression("x.__class__", {"x": 1})

    def test_no_exec_eval(self):
        """Test that exec/eval cannot be used."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError):
            evaluator.parse_expression('eval("1 + 1")')

        with pytest.raises(MetricExpressionError):
            evaluator.parse_expression('exec("x = 1")')
