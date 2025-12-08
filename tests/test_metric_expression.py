"""Tests for metric expression evaluation."""

import pytest

from asqi.metric_expression import MetricExpressionEvaluator
from asqi.errors import MetricExpressionError


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
            evaluator.parse_expression("open('file.txt')")  # open not in allowed list

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

    def test_evaluate_expression_with_abs(self):
        """Test absolute value function."""
        evaluator = MetricExpressionEvaluator()

        # Test with negative number
        result = evaluator.evaluate_expression("abs(a)", {"a": -0.5})
        assert result == pytest.approx(0.5)

        # Test with positive number
        result = evaluator.evaluate_expression("abs(b)", {"b": 0.3})
        assert result == pytest.approx(0.3)

        # Test in expression
        result = evaluator.evaluate_expression("abs(a - b)", {"a": 0.2, "b": 0.7})
        assert result == pytest.approx(0.5)

    def test_evaluate_expression_with_round(self):
        """Test rounding function."""
        evaluator = MetricExpressionEvaluator()

        # Test with 2 decimal places
        result = evaluator.evaluate_expression("round(a, 2)", {"a": 0.666})
        assert result == pytest.approx(0.67)

        # Test default rounding (to integer)
        result = evaluator.evaluate_expression("round(b)", {"b": 2.5})
        assert result == 2

        # Test in expression
        result = evaluator.evaluate_expression(
            "round(a + b, 1)", {"a": 0.333, "b": 0.444}
        )
        assert result == pytest.approx(0.8)

    def test_evaluate_expression_with_pow(self):
        """Test power function."""
        evaluator = MetricExpressionEvaluator()

        # Test integer power
        result = evaluator.evaluate_expression("pow(a, 2)", {"a": 3})
        assert result == 9

        # Test fractional power (square root)
        result = evaluator.evaluate_expression("pow(b, 0.5)", {"b": 4})
        assert result == pytest.approx(2.0)

        # Test in expression
        result = evaluator.evaluate_expression(
            "pow(a, 2) + pow(b, 2)", {"a": 3, "b": 4}
        )
        assert result == 25

    def test_evaluate_expression_division(self):
        """Test division operations."""
        evaluator = MetricExpressionEvaluator()

        result = evaluator.evaluate_expression("a / b", {"a": 10, "b": 4})
        assert result == pytest.approx(2.5)

    def test_evaluate_expression_empty_string(self):
        """Test that empty expression string raises appropriate error."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="Invalid expression syntax"):
            evaluator.evaluate_expression("", {"a": 0.5})

    def test_evaluate_expression_with_invalid_function_name(self):
        """Test that invalid function names (like 'average' instead of 'avg') raise error."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="not allowed"):
            evaluator.evaluate_expression(
                "average(a, b, c)", {"a": 0.6, "b": 0.8, "c": 1.0}
            )

    def test_evaluate_expression_with_assignment(self):
        """Test that variable assignment is not allowed."""
        evaluator = MetricExpressionEvaluator()

        with pytest.raises(MetricExpressionError, match="Invalid expression syntax"):
            evaluator.parse_expression("min = 3")

        with pytest.raises(MetricExpressionError, match="Invalid expression syntax"):
            evaluator.parse_expression("x = a + b")

    def test_evaluate_expression_with_non_numeric_metric_value(self):
        """Test that non-numeric metric values raise appropriate error."""
        evaluator = MetricExpressionEvaluator()

        # String value instead of number - causes TypeError during addition
        with pytest.raises(MetricExpressionError, match="Type error"):
            evaluator.evaluate_expression("a + b", {"a": "string", "b": 0.5})

        # List value instead of number - list * int returns list, which fails numeric check
        with pytest.raises(MetricExpressionError, match="must evaluate to a number"):
            evaluator.evaluate_expression("a * 2", {"a": [1, 2, 3]})

        # Dict value instead of number - causes TypeError during addition
        with pytest.raises(MetricExpressionError, match="Type error"):
            evaluator.evaluate_expression("a + 1", {"a": {"value": 5}})


class TestConditionalExpressions:
    """Tests for conditional (if-else) expressions."""

    def test_evaluate_expression_simple_if_else(self):
        """Test simple if-else expressions."""
        evaluator = MetricExpressionEvaluator()

        # Test when condition is true
        result = evaluator.evaluate_expression(
            "a if b > 0.5 else c", {"a": 1.0, "b": 0.8, "c": 2.0}
        )
        assert result == 1.0

        # Test when condition is false
        result = evaluator.evaluate_expression(
            "a if b > 0.5 else c", {"a": 1.0, "b": 0.3, "c": 2.0}
        )
        assert result == 2.0

    def test_evaluate_expression_hard_gates_pattern(self):
        """Test the hard gates pattern used in accuracy scoring."""
        evaluator = MetricExpressionEvaluator()

        # Test case: all gates pass (faith >= 0.7 and retr >= 0.6 and instruct >= 0.7)
        result = evaluator.evaluate_expression(
            "(0.45 * task_success + 0.35 * answer_correctness + 0.20 * helpfulness) if (faith >= 0.7 and retr >= 0.6 and instruct >= 0.7) else -1",
            {
                "task_success": 0.9,
                "answer_correctness": 0.8,
                "helpfulness": 0.7,
                "faith": 0.8,
                "retr": 0.7,
                "instruct": 0.75,
            },
        )
        expected = 0.45 * 0.9 + 0.35 * 0.8 + 0.20 * 0.7  # 0.405 + 0.28 + 0.14 = 0.825
        assert result == pytest.approx(expected)

        # Test case: faith gate fails (faith = 0.6 < 0.7)
        result = evaluator.evaluate_expression(
            "(0.6 * acc + 0.4 * rel) if (faith >= 0.7 and retr >= 0.6) else -1",
            {"acc": 0.9, "rel": 0.8, "faith": 0.6, "retr": 0.7},
        )
        assert result == -1

    def test_evaluate_expression_nested_conditionals(self):
        """Test nested conditional expressions."""
        evaluator = MetricExpressionEvaluator()

        # Test nested if-else
        result = evaluator.evaluate_expression(
            "a if b > 0.5 else (c if d < 0.3 else e)",
            {"a": 1.0, "b": 0.8, "c": 2.0, "d": 0.2, "e": 3.0},
        )
        assert result == 1.0  # b > 0.5, so returns a

        result = evaluator.evaluate_expression(
            "a if b > 0.5 else (c if d < 0.3 else e)",
            {"a": 1.0, "b": 0.3, "c": 2.0, "d": 0.2, "e": 3.0},
        )
        assert result == 2.0  # b <= 0.5 and d < 0.3, so returns c

        result = evaluator.evaluate_expression(
            "a if b > 0.5 else (c if d < 0.3 else e)",
            {"a": 1.0, "b": 0.3, "c": 2.0, "d": 0.5, "e": 3.0},
        )
        assert result == 3.0  # b <= 0.5 and d >= 0.3, so returns e


class TestComparisonOperators:
    """Tests for comparison operators (>, >=, <, <=, ==, !=)."""

    def test_evaluate_expression_comparison_operators(self):
        """Test comparison operators (>, >=, <, <=, ==, !=)."""
        evaluator = MetricExpressionEvaluator()

        # Test greater than and greater equal
        result = evaluator.evaluate_expression(
            "1 if a > b else 0", {"a": 0.8, "b": 0.6}
        )
        assert result == 1
        result = evaluator.evaluate_expression(
            "1 if a > b else 0", {"a": 0.4, "b": 0.6}
        )
        assert result == 0

        result = evaluator.evaluate_expression(
            "1 if a >= b else 0", {"a": 0.7, "b": 0.7}
        )
        assert result == 1
        result = evaluator.evaluate_expression(
            "1 if a >= b else 0", {"a": 0.6, "b": 0.7}
        )
        assert result == 0

        # Test less than and less equal
        result = evaluator.evaluate_expression(
            "1 if a < b else 0", {"a": 0.4, "b": 0.6}
        )
        assert result == 1
        result = evaluator.evaluate_expression(
            "1 if a < b else 0", {"a": 0.8, "b": 0.6}
        )
        assert result == 0

        result = evaluator.evaluate_expression(
            "1 if a <= b else 0", {"a": 0.7, "b": 0.7}
        )
        assert result == 1
        result = evaluator.evaluate_expression(
            "1 if a <= b else 0", {"a": 0.8, "b": 0.7}
        )
        assert result == 0

        # Test equality operators
        result = evaluator.evaluate_expression(
            "1 if a == b else 0", {"a": 0.7, "b": 0.7}
        )
        assert result == 1
        result = evaluator.evaluate_expression(
            "1 if a == b else 0", {"a": 0.7, "b": 0.8}
        )
        assert result == 0

        result = evaluator.evaluate_expression(
            "1 if a != b else 0", {"a": 0.7, "b": 0.8}
        )
        assert result == 1
        result = evaluator.evaluate_expression(
            "1 if a != b else 0", {"a": 0.7, "b": 0.7}
        )
        assert result == 0

    def test_evaluate_expression_comparison_in_arithmetic(self):
        """Test comparison operators used in arithmetic expressions."""
        evaluator = MetricExpressionEvaluator()

        # Use comparison result in arithmetic (True = 1, False = 0)
        result = evaluator.evaluate_expression(
            "(a > b) + (c >= d) + (e < f)",
            {"a": 0.8, "b": 0.6, "c": 0.7, "d": 0.7, "e": 0.4, "f": 0.5},
        )
        assert result == 3  # All three comparisons are true

        result = evaluator.evaluate_expression(
            "(a > b) + (c >= d) + (e < f)",
            {"a": 0.4, "b": 0.6, "c": 0.6, "d": 0.7, "e": 0.4, "f": 0.5},
        )
        assert result == 1  # Only e < f is true


class TestBooleanOperators:
    """Tests for boolean operators (and, or, not)."""

    def test_evaluate_expression_and_operator(self):
        """Test logical AND operator."""
        evaluator = MetricExpressionEvaluator()

        # Both conditions true
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 and b > 0.3) else 0", {"a": 0.8, "b": 0.6}
        )
        assert result == 1

        # First condition false
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 and b > 0.3) else 0", {"a": 0.3, "b": 0.6}
        )
        assert result == 0

        # Second condition false
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 and b > 0.3) else 0", {"a": 0.8, "b": 0.2}
        )
        assert result == 0

        # Both conditions false
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 and b > 0.3) else 0", {"a": 0.3, "b": 0.2}
        )
        assert result == 0

    def test_evaluate_expression_or_operator(self):
        """Test logical OR operator."""
        evaluator = MetricExpressionEvaluator()

        # Both conditions true
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 or b > 0.3) else 0", {"a": 0.8, "b": 0.6}
        )
        assert result == 1

        # First condition true, second false
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 or b > 0.3) else 0", {"a": 0.8, "b": 0.2}
        )
        assert result == 1

        # First condition false, second true
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 or b > 0.3) else 0", {"a": 0.3, "b": 0.6}
        )
        assert result == 1

        # Both conditions false
        result = evaluator.evaluate_expression(
            "1 if (a > 0.5 or b > 0.3) else 0", {"a": 0.3, "b": 0.2}
        )
        assert result == 0

    def test_evaluate_expression_not_operator(self):
        """Test logical NOT operator."""
        evaluator = MetricExpressionEvaluator()

        # Not true = false
        result = evaluator.evaluate_expression("1 if not (a > 0.5) else 0", {"a": 0.8})
        assert result == 0

        # Not false = true
        result = evaluator.evaluate_expression("1 if not (a > 0.5) else 0", {"a": 0.3})
        assert result == 1

    def test_evaluate_expression_complex_boolean(self):
        """Test complex boolean expressions."""
        evaluator = MetricExpressionEvaluator()

        # Complex expression: (a > 0.7 and b >= 0.6) or (c < 0.3 and not (d == 0.5))
        result = evaluator.evaluate_expression(
            "1 if ((a > 0.7 and b >= 0.6) or (c < 0.3 and not (d == 0.5))) else 0",
            {"a": 0.8, "b": 0.7, "c": 0.2, "d": 0.4},
        )
        assert result == 1  # First part (a > 0.7 and b >= 0.6) is true

        result = evaluator.evaluate_expression(
            "1 if ((a > 0.7 and b >= 0.6) or (c < 0.3 and not (d == 0.5))) else 0",
            {"a": 0.6, "b": 0.7, "c": 0.2, "d": 0.4},
        )
        assert result == 1  # Second part (c < 0.3 and not (d == 0.5)) is true

        result = evaluator.evaluate_expression(
            "1 if ((a > 0.7 and b >= 0.6) or (c < 0.3 and not (d == 0.5))) else 0",
            {"a": 0.6, "b": 0.5, "c": 0.4, "d": 0.5},
        )
        assert result == 0  # Neither part is true


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
