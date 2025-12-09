"""
Safe metric expression evaluator for score card indicators.

Supports arithmetic operations and aggregation functions while maintaining security by using AST parsing instead of eval().
"""

import ast
import logging
from typing import Any, Dict, Union
from asqi.errors import MetricExpressionError

logger = logging.getLogger(__name__)


class MetricExpressionEvaluator:
    """
    Safe evaluator for metric expressions in score cards.

    Supports:
    - Arithmetic operators: +, -, *, /
    - Comparison operators: >, >=, <, <=, ==, !=
    - Boolean operators: and, or, not
    - Conditional expressions: if-else
    - Aggregation functions: min(), max(), avg()
    - Numeric literals and parentheses
    - Complex formulas with hard gates: '(score) if (gate1 >= 0.7 and gate2 >= 0.6) else -1'

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
        >>> # Hard gates with conditional
        >>> evaluator.evaluate_expression(
        ...     "(0.6 * acc + 0.4 * rel) if (faith >= 0.7 and retr >= 0.6) else -1",
        ...     {"acc": 0.8, "rel": 0.9, "faith": 0.8, "retr": 0.7}
        ... )
        0.84
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

    ALLOWED_COMPARISONS = {
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
    }

    ALLOWED_BOOL_OPS = {
        ast.And,
        ast.Or,
        ast.Not,
    }

    ALLOWED_FUNCTIONS = {"min", "max", "avg", "abs", "round", "pow"}

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
                ast.Compare,
                ast.BoolOp,
                ast.IfExp,
            ):
                # Validate binary and unary operators
                if isinstance(node, ast.BinOp):
                    op_type = type(node.op)
                    if op_type not in self.ALLOWED_OPS:
                        raise MetricExpressionError(
                            f"Unsupported operator: {op_type.__name__}"
                        )

                # Validate comparison operators
                elif isinstance(node, ast.Compare):
                    for op in node.ops:
                        if type(op) not in self.ALLOWED_COMPARISONS:
                            raise MetricExpressionError(
                                f"Unsupported comparison: {type(op).__name__}"
                            )

                # Validate boolean operators
                elif isinstance(node, ast.BoolOp):
                    if type(node.op) not in self.ALLOWED_BOOL_OPS:
                        raise MetricExpressionError(
                            f"Unsupported boolean operator: {type(node.op).__name__}"
                        )

                # Validate unary operators
                elif isinstance(node, ast.UnaryOp):
                    op_type = type(node.op)
                    if op_type not in self.ALLOWED_OPS and op_type != ast.Not:
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
            elif node_type in self.ALLOWED_COMPARISONS:
                # Comparison operators - these are fine
                pass
            elif node_type in self.ALLOWED_BOOL_OPS:
                # Boolean operators - these are fine
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
        context["abs"] = abs
        context["round"] = round
        context["pow"] = pow
        context["avg"] = lambda *args: sum(args) / len(args) if args else 0

        try:
            # Compile and evaluate
            # Note: This eval is safe because:
            # 1. AST is validated to only contain allowed operations (arithmetic, allowed functions)
            # 2. __builtins__ is empty, preventing access to dangerous functions
            # 3. Context only contains validated numeric values and safe functions
            code = compile(tree, "<expression>", "eval")
            result = eval(code, {"__builtins__": {}}, context)  # nosec B307

            # Convert boolean to int (True->1, False->0)
            # This handles comparison and boolean expressions consistently
            if isinstance(result, bool):
                result = int(result)

            # Ensure result is numeric
            if not isinstance(result, (int, float)):
                raise MetricExpressionError(
                    f"Expression must evaluate to a number, got {type(result).__name__}"
                )

            return result

        except NameError as e:
            # Extract missing metric name
            missing_metric = getattr(e, "name", "unknown")
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
