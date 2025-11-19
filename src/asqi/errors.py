from typing import Any, Dict, Optional


class DuplicateIDError(Exception):
    """
    Exception raised when duplicate IDs are found across configuration files.

    Args:
        duplicate_dict: Dictionary of duplicate IDs with duplication data

    Example:
        duplicate_dict = {
            "t_duplicate_id": {
                "id": "duplicate_id",
                "config_type": "test_suite",
                "occurrences": [
                    {"location": "config.yaml", "test_suite_name": "suite", "test_name": "test 1"},
                    {"location": "config.yaml", "test_suite_name": "suite", "test_name": "test 2"}
                ]
            }
        }
    """

    def __init__(self, duplicate_dict: Dict[str, Any]):
        self.duplicate_dict = duplicate_dict
        message = self._get_message()
        super().__init__(message)

    def _get_message(self) -> str:
        """
        Returns a message with all duplicates.
        """
        lines = ["\n"]

        for duplicate_count, (_, id_list) in enumerate(self.duplicate_dict.items(), 1):
            lines.append(
                f"#{duplicate_count}: Duplicate id -> {id_list['id']} in {id_list['config_type']}"
            )
            for occurrence_count, occurrence_details in enumerate(
                id_list["occurrences"], 1
            ):
                lines.append(f"--{occurrence_count}-- {occurrence_details}")
            lines.append("")

        lines.append("IDs must be unique within the same file.")

        return "\n".join(lines)


class MissingIDFieldError(Exception):
    """Exception raised when required ID fields are missing."""

    pass


class ManifestExtractionError(Exception):
    """Exception raised when manifest extraction fails."""

    def __init__(
        self, message: str, error_type: str, original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class MissingImageError(Exception):
    """Exception raised when required Docker images are missing."""

    pass


class MountExtractionError(Exception):
    """Exception raised when extracting mounts from args fails."""

    pass


class MetricExpressionError(Exception):
    """Raised when metric expression parsing or evaluation fails."""

    pass
