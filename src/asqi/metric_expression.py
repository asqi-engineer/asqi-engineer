"""
Safe metric expression evaluator for score card indicators.

Supports arithmetic operations and aggregation functions while maintaining security
by using AST parsing instead of eval().
"""

import ast
import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


class MetricExpressionError(Exception):
    """Raised when metric expression parsing or evaluation fails."""

    pass


class MetricExpressionEvaluator:
    """
    Safe evaluator for metric expressions in score cards.

    Supports:
    - Arithmetic operators: +, -, *, /
    - Aggregation functions: min(), max(), avg()
    - Numeric literals and parentheses
    - Complex formulas: '0.7 * accuracy + 0.3 * relevance'

    Does NOT support:
    - Code execution (no eval/exec)
    - Arbitrary function calls
    - Variable assignment
    - Imports or other Python statements

    Examples:
        >>> evaluator = MetricExpressionEvaluator()
        >>> # Simple variable
        >>> evaluator.evaluate_expression("accuracy", {"accuracy": 0.85})
        0.85
        >>> # Weighted average
        >>> evaluator.evaluate_expression("0.7 * a + 0.3 * b", {"a": 0.8, "b": 0.9})
        0.83
        >>> # Min function
        >>> evaluator.evaluate_expression("min(x, y, z)", {"x": 0.9, "y": 0.7, "z": 0.8})
        0.7
    """

    # Allowed AST node types for safety
    ALLOWED_OPS = {
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,  # Unary plus
        ast.USub,  # Unary minus
    }

    ALLOWED_FUNCTIONS = {"min", "max", "avg"}

    def parse_expression(self, expression: str) -> ast.Expression:
        """
        Parse an expression string into an AST.

        Args:
            expression: The expression string to parse

        Returns:
            Parsed AST expression

        Raises:
            MetricExpressionError: If parsing fails or contains unsafe operations
        """
        try:
            # Parse in 'eval' mode (expression, not statement)
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise MetricExpressionError(f"Invalid expression syntax: {e}") from e

        # Validate the AST for safety
        self._validate_ast(tree)

        return tree

    def _validate_ast(self, tree: ast.Expression) -> None:
        """
        Validate that an AST only contains allowed operations.

        Args:
            tree: The AST to validate

        Raises:
            MetricExpressionError: If AST contains unsafe operations
        """
        for node in ast.walk(tree):
            node_type = type(node)

            # Allow expression root, operators, constants, and function calls
            if node_type in (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Constant,
                ast.Num,  # For older Python versions
                ast.Name,
                ast.Load,
                ast.Attribute,  # For nested paths like stats.pass_rate
            ):
                # Validate binary and unary operators
                if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                    op_type = type(node.op)
                    if op_type not in self.ALLOWED_OPS:
                        raise MetricExpressionError(
                            f"Unsupported operator: {op_type.__name__}"
                        )

                # Validate function calls
                elif isinstance(node, ast.Call):
                    if not isinstance(node.func, ast.Name):
                        raise MetricExpressionError(
                            "Only simple function calls are allowed"
                        )
                    func_name = node.func.id
                    if func_name not in self.ALLOWED_FUNCTIONS:
                        raise MetricExpressionError(
                            f"Function '{func_name}' is not allowed. "
                            f"Allowed functions: {', '.join(sorted(self.ALLOWED_FUNCTIONS))}"
                        )
            elif node_type in self.ALLOWED_OPS:
                # The operator types themselves (Add, Sub, etc.) - these are fine
                pass
            else:
                raise MetricExpressionError(
                    f"Unsupported AST node type: {type(node).__name__}"
                )

    def evaluate_expression(
        self, expression: str, metric_values: Dict[str, Union[int, float]]
    ) -> Union[int, float]:
        """
        Evaluate a metric expression with provided metric values.

        Args:
            expression: The expression string to evaluate
            metric_values: Dictionary mapping metric paths to their numeric values

        Returns:
            The computed numeric result

        Raises:
            MetricExpressionError: If evaluation fails

        Example:
            >>> evaluator.evaluate_expression(
            ...     "0.5 * accuracy + 0.5 * relevance",
            ...     {"accuracy": 0.8, "relevance": 0.9}
            ... )
            0.85
        """
        try:
            tree = self.parse_expression(expression)
        except MetricExpressionError:
            raise

        # Create evaluation context with metric values and functions
        context: Dict[str, Any] = metric_values.copy()
        context["min"] = min
        context["max"] = max
        context["avg"] = lambda *args: sum(args) / len(args) if args else 0

        try:
            # Compile and evaluate
            # Note: This eval is safe because:
            # 1. AST is validated to only contain allowed operations (arithmetic, allowed functions)
            # 2. __builtins__ is empty, preventing access to dangerous functions
            # 3. Context only contains validated numeric values and safe functions
            code = compile(tree, "<expression>", "eval")
            result = eval(code, {"__builtins__": {}}, context)  # nosec B307

            # Ensure result is numeric
            if not isinstance(result, (int, float)):
                raise MetricExpressionError(
                    f"Expression must evaluate to a number, got {type(result).__name__}"
                )

            return result

        except NameError as e:
            # Extract missing metric name
            missing_metric = str(e).split("'")[1] if "'" in str(e) else "unknown"
            available_metrics = list(metric_values.keys())
            raise MetricExpressionError(
                f"Metric '{missing_metric}' not found. Available metrics: {available_metrics}"
            ) from e
        except ZeroDivisionError as e:
            raise MetricExpressionError("Division by zero in expression") from e
        except TypeError as e:
            raise MetricExpressionError(
                f"Type error in expression evaluation: {e}"
            ) from e
        except Exception as e:
            raise MetricExpressionError(f"Error evaluating expression: {e}") from e
