import pytest
from pydantic import ValidationError

from asqi.response_schemas import (
    ContainerOutput,
    GeneratedDataset,
    GeneratedReport,
    validate_container_output,
)


class TestGeneratedDataset:
    """Tests for GeneratedDataset schema validation."""

    def test_valid_dataset(self):
        """Test creating a valid dataset with all fields."""
        dataset = GeneratedDataset(
            dataset_name="test_data",
            dataset_type="huggingface",
            dataset_path="/output/data.parquet",
            format="parquet",
            metadata={"version": "1.0", "num_rows": 100},
        )
        assert dataset.dataset_name == "test_data"
        assert dataset.dataset_type == "huggingface"
        assert dataset.dataset_path == "/output/data.parquet"
        assert dataset.format == "parquet"
        assert dataset.metadata == {"version": "1.0", "num_rows": 100}

    def test_minimal_dataset(self):
        """Test creating a dataset with only required fields."""
        dataset = GeneratedDataset(
            dataset_name="minimal",
            dataset_type="txt",
            dataset_path="/output/minimal.txt",
        )
        assert dataset.dataset_name == "minimal"
        assert dataset.format is None
        assert dataset.metadata is None

    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GeneratedDataset(
                dataset_name="test",
                dataset_type="huggingface",
                # Missing dataset_path
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("dataset_path",) for e in errors)

    def test_empty_dataset_name(self):
        """Test that empty dataset_name raises ValidationError."""
        with pytest.raises(ValidationError):
            GeneratedDataset(
                dataset_name="",
                dataset_type="huggingface",
                dataset_path="/output/data.parquet",
            )

    def test_whitespace_dataset_name(self):
        """Test that whitespace-only dataset_name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GeneratedDataset(
                dataset_name="   ",
                dataset_type="huggingface",
                dataset_path="/output/data.parquet",
            )
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_empty_dataset_path(self):
        """Test that empty dataset_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GeneratedDataset(
                dataset_name="test",
                dataset_type="huggingface",
                dataset_path="   ",  # Whitespace only
            )
        assert "cannot be empty or whitespace-only" in str(exc_info.value)


class TestGeneratedReport:
    """Tests for GeneratedReport schema validation."""

    def test_valid_report(self):
        """Test creating a valid report with all fields."""
        report = GeneratedReport(
            report_name="security_analysis",
            report_type="html",
            report_path="/output/security.html",
            metadata={"author": "asqi", "file_size_bytes": 52480},
        )
        assert report.report_name == "security_analysis"
        assert report.report_type == "html"
        assert report.report_path == "/output/security.html"
        assert report.metadata == {"author": "asqi", "file_size_bytes": 52480}

    def test_minimal_report(self):
        """Test creating a report with only required fields."""
        report = GeneratedReport(
            report_name="simple",
            report_type="json",
            report_path="/output/simple.json",
        )
        assert report.report_name == "simple"
        assert report.metadata is None

    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GeneratedReport(
                report_name="test",
                report_type="html",
                # Missing report_path
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("report_path",) for e in errors)

    def test_empty_report_name(self):
        """Test that empty report_name raises ValidationError."""
        with pytest.raises(ValidationError):
            GeneratedReport(
                report_name="",
                report_type="html",
                report_path="/output/report.html",
            )

    def test_whitespace_report_path(self):
        """Test that whitespace-only report_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            GeneratedReport(
                report_name="test",
                report_type="html",
                report_path="   ",
            )
        assert "cannot be empty or whitespace-only" in str(exc_info.value)


class TestContainerOutput:
    """Tests for ContainerOutput schema validation."""

    def test_results_field(self):
        """Test new 'results' field."""
        output = ContainerOutput(results={"success": True, "score": 0.95})
        assert output.get_results()["success"] is True
        assert output.get_results()["score"] == 0.95

    def test_test_results_field(self):
        """Test legacy 'test_results' field."""
        output = ContainerOutput(test_results={"success": True, "score": 0.85})
        assert output.get_results()["success"] is True
        assert output.get_results()["score"] == 0.85

    def test_prefers_results_over_test_results(self):
        """When both present, 'results' takes precedence."""
        output = ContainerOutput(
            results={"success": True, "score": 0.95},
            test_results={"success": False, "score": 0.50},
        )
        assert output.get_results()["success"] is True
        assert output.get_results()["score"] == 0.95

    def test_empty_results_dict_fails(self):
        """Empty results dictionary should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContainerOutput(results={})
        assert "cannot be empty dictionary" in str(exc_info.value)

    def test_empty_test_results_dict_fails(self):
        """Empty test_results dictionary should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContainerOutput(test_results={})
        assert "cannot be empty dictionary" in str(exc_info.value)

    def test_with_generated_reports(self):
        """Test output with generated reports."""
        output = ContainerOutput(
            results={"success": True},
            generated_reports=[
                GeneratedReport(
                    report_name="report1",
                    report_type="html",
                    report_path="/output/report1.html",
                )
            ],
        )
        assert len(output.generated_reports) == 1
        assert output.generated_reports[0].report_name == "report1"

    def test_with_generated_datasets(self):
        """Test output with generated datasets."""
        output = ContainerOutput(
            results={"success": True},
            generated_datasets=[
                GeneratedDataset(
                    dataset_name="data1",
                    dataset_type="huggingface",
                    dataset_path="/output/data1.parquet",
                )
            ],
        )
        assert len(output.generated_datasets) == 1
        assert output.generated_datasets[0].dataset_name == "data1"

    def test_backward_compatibility_extra_fields(self):
        """Extra fields are allowed for backward compatibility."""
        output = ContainerOutput(
            results={"success": True}, custom_field="value", another_field=123
        )
        # Access via model_dump to see extra fields
        dumped = output.model_dump()
        assert dumped["custom_field"] == "value"
        assert dumped["another_field"] == 123

    def test_default_empty_lists(self):
        """Test that reports and datasets default to empty lists."""
        output = ContainerOutput(results={"success": True})
        assert output.generated_reports == []
        assert output.generated_datasets == []

    def test_neither_results_field_returns_empty_dict(self):
        """When neither results field is present, get_results() returns empty dict."""
        output = ContainerOutput(
            generated_datasets=[
                GeneratedDataset(
                    dataset_name="data",
                    dataset_type="txt",
                    dataset_path="/output/data.txt",
                )
            ]
        )
        # This passes ContainerOutput validation but validate_container_output will fail
        assert output.get_results() == {}


class TestValidateContainerOutput:
    """Tests for validate_container_output function."""

    def test_valid_output_with_results(self):
        """Test validation with recommended 'results' field."""
        output_dict = {"results": {"success": True, "score": 0.95}}
        validated = validate_container_output(output_dict)
        assert validated.get_results()["success"] is True

    def test_valid_output_with_test_results(self):
        """Test validation with legacy 'test_results' field."""
        output_dict = {"test_results": {"success": True, "score": 0.85}}
        validated = validate_container_output(output_dict)
        assert validated.get_results()["success"] is True

    def test_valid_output_with_both_fields(self):
        """Test that 'results' is preferred when both are present."""
        output_dict = {
            "results": {"success": True, "score": 0.95},
            "test_results": {"success": False, "score": 0.50},
        }
        validated = validate_container_output(output_dict)
        assert validated.get_results()["score"] == 0.95

    def test_missing_results_fields(self):
        """Test that missing both results fields raises ValueError."""
        output_dict = {"generated_datasets": []}
        with pytest.raises(ValueError) as exc_info:
            validate_container_output(output_dict)
        assert "must contain 'results' or 'test_results' field" in str(exc_info.value)

    def test_invalid_generated_dataset(self):
        """Test that invalid nested dataset raises ValidationError."""
        output_dict = {
            "results": {"success": True},
            "generated_datasets": [
                {
                    "dataset_name": "test",
                    # Missing dataset_type and dataset_path
                }
            ],
        }
        with pytest.raises(ValidationError):
            validate_container_output(output_dict)

    def test_invalid_generated_report(self):
        """Test that invalid nested report raises ValidationError."""
        output_dict = {
            "results": {"success": True},
            "generated_reports": [
                {
                    "report_name": "test",
                    # Missing report_type and report_path
                }
            ],
        }
        with pytest.raises(ValidationError):
            validate_container_output(output_dict)

    def test_complete_valid_output(self):
        """Test validation of complete output with all fields."""
        output_dict = {
            "results": {"success": True, "score": 0.95, "details": "All checks passed"},
            "generated_reports": [
                {
                    "report_name": "security",
                    "report_type": "html",
                    "report_path": "/output/security.html",
                    "metadata": {"file_size_bytes": 1024},
                }
            ],
            "generated_datasets": [
                {
                    "dataset_name": "augmented",
                    "dataset_type": "huggingface",
                    "dataset_path": "/output/augmented.parquet",
                    "format": "parquet",
                    "metadata": {"num_rows": 500},
                }
            ],
            "custom_metadata": {"version": "1.0"},  # Extra field
        }
        validated = validate_container_output(output_dict)

        # Verify all fields are correctly validated
        assert validated.get_results()["success"] is True
        assert validated.get_results()["score"] == 0.95
        assert len(validated.generated_reports) == 1
        assert len(validated.generated_datasets) == 1

        # Verify nested objects are properly typed
        report = validated.generated_reports[0]
        assert isinstance(report, GeneratedReport)
        assert report.report_name == "security"

        dataset = validated.generated_datasets[0]
        assert isinstance(dataset, GeneratedDataset)
        assert dataset.dataset_name == "augmented"

    def test_model_dump_for_serialization(self):
        """Test that validated output can be serialized via model_dump."""
        output_dict = {
            "results": {"success": True},
            "generated_reports": [
                {
                    "report_name": "test",
                    "report_type": "json",
                    "report_path": "/output/test.json",
                }
            ],
        }
        validated = validate_container_output(output_dict)

        # model_dump should work for serialization
        dumped = validated.model_dump()
        assert dumped["results"]["success"] is True
        assert len(dumped["generated_reports"]) == 1
        assert dumped["generated_reports"][0]["report_name"] == "test"
