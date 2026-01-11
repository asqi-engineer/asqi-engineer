from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class GeneratedDataset(BaseModel):
    """Represents a dataset produced by a container during execution.
    The paths are container-internal and get translated to host paths
    by the workflow system.
    """

    dataset_name: str = Field(..., min_length=1, description="Name of the dataset")
    dataset_type: Literal["huggingface", "pdf", "txt"] = Field(
        ..., description="Type of dataset: 'huggingface', 'pdf', or 'txt'"
    )
    dataset_path: str = Field(
        ..., min_length=1, description="Path to the dataset file inside container"
    )
    format: Optional[str] = Field(
        None, description="File format (e.g., 'parquet', 'json', 'csv')"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the dataset (e.g., num_rows, size, etc.)",
    )

    @field_validator("dataset_path")
    @classmethod
    def validate_path_not_empty(cls, v: str) -> str:
        """Validate that dataset_path is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("dataset_path cannot be empty or whitespace-only")
        return v

    @field_validator("dataset_name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that dataset_name is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("dataset_name cannot be empty or whitespace-only")
        return v


class GeneratedReport(BaseModel):
    """Runtime output for a generated report.

    Represents a report (HTML, PDF, JSON, etc.) produced by a container
    during execution. The paths are container-internal and get translated
    to host paths by the workflow system.
    """

    report_name: str = Field(..., min_length=1, description="Name of the report")
    report_type: Literal["html", "pdf", "json"] = Field(
        ..., description="Type of report: 'html', 'pdf', or 'json'"
    )
    report_path: str = Field(
        ..., min_length=1, description="Path to the report file inside container"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the report (e.g., file_size_bytes, checksum, etc.)",
    )

    @field_validator("report_path")
    @classmethod
    def validate_path_not_empty(cls, v: str) -> str:
        """Validate that report_path is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("report_path cannot be empty or whitespace-only")
        return v

    @field_validator("report_name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Validate that report_name is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("report_name cannot be empty or whitespace-only")
        return v


class ContainerOutput(BaseModel):
    """Complete container output schema.

    Supports both 'results' (recommended) and 'test_results' (legacy) field names
    for backward compatibility.
    """

    # Accept both field names for backward compatibility
    results: Optional[Dict[str, Any]] = Field(
        None, description="Test/generation results (recommended field name)"
    )
    test_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Legacy field name for results (deprecated but still supported)",
    )

    generated_reports: List[GeneratedReport] = Field(
        default_factory=list,
        description="List of generated reports from container execution",
    )
    generated_datasets: List[GeneratedDataset] = Field(
        default_factory=list,
        description="List of generated datasets from container execution",
    )

    model_config = {"extra": "allow"}  # Allow extra fields for backward compatibility

    @field_validator("test_results", "results")
    @classmethod
    def validate_results_not_empty_if_present(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Ensure results field is not empty if present.

        At least one results field must contain the 'success' key.
        """
        if v is not None and not v:
            raise ValueError(
                "Results cannot be empty dictionary - must contain at least 'success' field"
            )
        return v

    def get_results(self) -> Dict[str, Any]:
        """Get results, preferring 'results' over 'test_results'.

        Returns:
            Results dictionary, preferring 'results' field over 'test_results'.
            Returns empty dict if neither field is present.
        """
        return self.results or self.test_results or {}


def validate_container_output(output_dict: Dict[str, Any]) -> ContainerOutput:
    """
    Validate container output against the schema.

    It supports both 'results' (recommended) and 'test_results' (legacy) field names for backward compatibility.

    Args:
        output_dict: Raw container output dictionary parsed from JSON

    Returns:
        Validated ContainerOutput object

    Raises:
        ValidationError: If output doesn't match schema (missing required fields,
                        invalid types, etc.)
        ValueError: If neither 'results' nor 'test_results' field is present
    """
    validated = ContainerOutput(**output_dict)
    if validated.results is None and validated.test_results is None:
        raise ValueError(
            "Container output must contain 'results' or 'test_results' field. "
            "Both fields are missing."
        )

    return validated
