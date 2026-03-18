"""Utilities for accessing nested data structures with flexible path syntax.

Supports both dot notation (a.b.c) and bracket notation (a["b.c"]),
useful for extracting values from complex nested dictionaries
like test results or configuration objects.

Examples:
    'success' -> extract top-level key
    'vulnerability_stats.Toxicity.overall_pass_rate' -> nested with dots
    'probe_results["encoding.InjectHex"]["encoding.DecodeMatch"].passed' -> bracket notation
    'probe_results["encoding.InjectHex"].total_attempts' -> mixed notation
"""

import re
from typing import Any, Dict, List, Optional, Tuple


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

        # Find the matching closing bracket (validation ensures it exists)
        bracket_end = path.find("]", bracket_start)

        # Extract the key from within brackets (validation ensures proper quoting)
        bracket_content = path[bracket_start + 1 : bracket_end]
        key = bracket_content[1:-1]  # Remove quotes (validation ensures they exist)
        keys.append(key)

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
