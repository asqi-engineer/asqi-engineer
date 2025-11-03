from typing import Dict, List, Optional


class DuplicateIDError(Exception):
    """
    Exception raised when duplicate IDs are found across configuration files.

    Args:
        duplicate_dict: Dictionary of duplicate IDs with duplication data

    Example:
        duplicate_dict = {
            "id: example_id, config type: test suite": [
                "location: 'config.yaml', suite name: 'suite example', test name: 'test 1'",
                "location: 'config.yaml', suite name: 'suite example', test name: 'test 2'"
            ]
        }
    """

    def __init__(self, duplicate_dict: Dict[str, List[str]]):
        self.duplicate_dict = duplicate_dict
        message = self._get_message()
        super().__init__(message)

    def _get_message(self) -> str:
        """
        Returns a message with all duplicates.
        """
        lines = ["\n"]

        for duplicate_count, (duplicate_id, id_list) in enumerate(
            self.duplicate_dict.items(), 1
        ):
            lines.append(f"#{duplicate_count}: Duplicate -> {duplicate_id}")
            for occurrence_count, occurrence_details in enumerate(id_list, 1):
                lines.append(f"--{occurrence_count}-- {occurrence_details}")
            lines.append("")

        lines.append("IDs must be unique within the same file.")

        return "\n".join(lines)


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
