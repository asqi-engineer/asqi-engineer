from typing import Dict, List, Optional


class DuplicateTestIDError(Exception):
    """
    Exception raised when duplicate test IDs are found.

    Args:
        duplicate_dict: Dict of duplicate IDs with duplication data

    Notes:
    - test IDs must be unique within the same file
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

        for ix_id, (duplicate_id, test_list) in enumerate(
            self.duplicate_dict.items(), 1
        ):
            lines.append(f"#{ix_id}: Duplicate ID({duplicate_id})")
            for ix_test, test in enumerate(test_list, 1):
                lines.append(f"--{ix_test}-- {test}")
            lines.append("")

        lines.append("Test IDs must be unique within the same file")

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
