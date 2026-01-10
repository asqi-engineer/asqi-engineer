"""
Unit tests for output.py module, focusing on JSON parsing from container output.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asqi.output import (
    _translate_container_path,
    _verify_and_display_output_item,
    display_generated_datasets,
    display_score_card_reports,
    extract_container_json_output_fields,
    parse_container_json_output,
    translate_dataset_paths,
    translate_report_paths,
)


class TestParseContainerJsonOutput:
    """Test suite for parse_container_json_output function."""

    def test_simple_single_line_json(self):
        """Test parsing simple single-line JSON without formatting."""
        output = '{"success": true, "score": 0.95}'
        result = parse_container_json_output(output)
        assert result == {"success": True, "score": 0.95}

    def test_multiline_formatted_json(self):
        """Test parsing well-formatted multi-line JSON."""
        output = """{
  "success": true,
  "score": 0.95,
  "message": "Test completed"
}"""
        result = parse_container_json_output(output)
        assert result == {"success": True, "score": 0.95, "message": "Test completed"}

    def test_extracts_json_from_noisy_real_log(self):
        """
        Test extracting JSON from real ASQI Engineer logs with noise and newlines.
        """

        output = """
        nction successful on attempt 1\n{\"success\": true, \"test_type\": \"privacy\", \"datasets_tested\": [\"privacy_leakage\"], \"results_summary\": {\"privacy_leakage\": {\"evaluation_score\": {\"RtA\": 0.05, \"TD\": 0.025, \"CD\": 0.041666666666666664}, \"evaluation_error\": null, \"sample_count\": 10}}}\n\rProcessing privacy_leakage.json (10 rows):   0%|          | 0/2 [00:00<?, ?it/s]\rProcessing privacy_leakage.json (10 rows):  50%|\u2588\u2588\u2588\u2588\u2588     | 1/2 [00:05<00:05,  5.09s/it]\rProcessing privacy_leakage.json (10 rows): 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 2/2 [00:08<00:00,  3.87s/it]\r                                                                                        \rTrustLLM generation results copied to: /output/trustllm_privacy/generation_results/bedrock/arn:aws:bedrock:us-east-1:156772879641:inference-profile/us.amazon.nova-lite-v1:0/privacy\n
        """
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert result["test_type"] == "privacy"
        assert (
            result["results_summary"]["privacy_leakage"]["evaluation_score"]["RtA"]
            == 0.05
        )
        assert (
            result["results_summary"]["privacy_leakage"]["evaluation_score"]["TD"]
            == 0.025
        )
        assert (
            result["results_summary"]["privacy_leakage"]["evaluation_score"]["CD"]
            == 0.041666666666666664
        )

    def test_invalid_json_in_real_log_error(self):
        """
        Test error raised when JSON in real ASQI Engineer logs is not properly closed.
        """

        output = """
        {\"success\": true, \"test_type\": \"privacy\", \"datasets_tested\": [\"privacy_leakage\"], \"results_summary\": {\"privacy_leakage\": {\"evaluation_score\": {\"RtA\": 0.05, \"TD\": 0.025, \"CD\": 0.041666666666666664
        """
        with pytest.raises(
            ValueError,
            match="JSON object not properly closed",
        ):
            parse_container_json_output(output)

    def test_json_with_log_prefix(self):
        """Test parsing JSON that appears after log lines."""
        output = """2025-11-04 08:46:46 [INFO] Starting test
Running probes...
Test execution in progress
{
  "success": true,
  "score": 0.85
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert result["score"] == 0.85

    def test_nested_json_with_arrays(self):
        """
        Test parsing complex nested JSON with arrays containing objects.
        This is the key bug case from issue #228 - the old parser would
        greedily return the first object in the array instead of the complete JSON.
        """
        output = """[INFO] Test completed
{
  "success": true,
  "recipe": "singapore-facts",
  "items": [
    {"id": 1, "name": "first"},
    {"id": 2, "name": "second"}
  ],
  "score": 0.95
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert "items" in result
        assert len(result["items"]) == 2
        assert result["items"][0]["id"] == 1
        assert result["items"][1]["id"] == 2
        assert result["score"] == 0.95

    def test_deeply_nested_json(self):
        """Test parsing deeply nested JSON structures (like Moonshot output)."""
        output = """[INFO] Running benchmark
{
  "success": true,
  "run_result": {
    "results": {
      "metadata": {"id": "runner-1", "status": "completed"},
      "results": {
        "recipes": [
          {
            "id": "test-1",
            "details": [
              {
                "model_id": "test-model",
                "data": [
                  {"prompt": "test", "score": 0.9}
                ]
              }
            ]
          }
        ]
      }
    }
  }
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert "run_result" in result
        assert result["run_result"]["results"]["metadata"]["id"] == "runner-1"
        assert len(result["run_result"]["results"]["results"]["recipes"]) == 1

    def test_multiple_json_objects_returns_last(self):
        """
        Test that when multiple complete JSON objects exist,
        the LAST one is returned (new behavior).
        """
        output = """{"first": "object", "success": false}
{"second": "object", "success": true, "final": "result"}"""
        result = parse_container_json_output(output)
        # Should return the LAST complete JSON object
        assert result == {"second": "object", "success": True, "final": "result"}
        assert "first" not in result

    def test_incomplete_json_then_complete(self):
        """Test handling incomplete JSON followed by complete JSON."""
        output = """[DEBUG] Partial: {"temp
[INFO] Final result:
{
  "success": true,
  "actual": "result"
}"""
        result = parse_container_json_output(output)
        assert result == {"success": True, "actual": "result"}

    def test_json_with_curly_braces_in_log_lines(self):
        """Test parsing when log lines contain curly braces."""
        output = """[INFO] Config: {"debug": true, "mode": "test"}
[INFO] Starting test with params {verbose: true}
{
  "success": true,
  "score": 0.95
}"""
        result = parse_container_json_output(output)
        assert result == {"success": True, "score": 0.95}

    def test_json_with_string_containing_braces(self):
        """Test JSON containing strings with curly braces."""
        output = """{
  "success": true,
  "message": "Test {placeholder} completed with {result}",
  "template": "Value: {value}"
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert "{placeholder}" in result["message"]
        assert "{value}" in result["template"]

    def test_json_with_whitespace_variations(self):
        """Test parsing JSON with various whitespace patterns."""
        output = """    {
      "success": true,
      "score": 0.95
    }"""
        result = parse_container_json_output(output)
        assert result == {"success": True, "score": 0.95}

    def test_json_with_trailing_whitespace(self):
        """Test parsing JSON with trailing whitespace and newlines."""
        output = """{
  "success": true,
  "score": 0.95
}

"""
        result = parse_container_json_output(output)
        assert result == {"success": True, "score": 0.95}

    def test_empty_output_raises_error(self):
        """Test that empty output raises ValueError with helpful message."""
        with pytest.raises(
            ValueError,
            match="Empty container output - test container produced no output",
        ):
            parse_container_json_output("")

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only output raises ValueError."""
        with pytest.raises(ValueError, match="Empty container output"):
            parse_container_json_output("   \n\n   ")

    def test_no_json_in_output_raises_error(self):
        """Test that output without JSON raises ValueError."""
        output = """[INFO] Test started
[INFO] Running tests
[ERROR] Something went wrong
Test completed"""
        with pytest.raises(ValueError, match="No valid JSON found in container output"):
            parse_container_json_output(output)

    def test_invalid_json_raises_error(self):
        """Test that malformed JSON raises ValueError."""
        output = """{
  "success": true,
  "score": 0.95
  INVALID SYNTAX
}"""
        with pytest.raises(ValueError, match="No valid JSON found"):
            parse_container_json_output(output)

    def test_error_message_includes_preview(self):
        """Test that error messages include output preview for debugging."""
        long_output = "No JSON here " * 20
        with pytest.raises(ValueError) as exc_info:
            parse_container_json_output(long_output)

        error_msg = str(exc_info.value)
        assert "Output preview:" in error_msg
        assert "..." in error_msg  # Truncation indicator

    def test_json_with_boolean_values(self):
        """Test parsing JSON with various boolean values."""
        output = """{
  "success": true,
  "failed": false,
  "enabled": true
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert result["failed"] is False
        assert result["enabled"] is True

    def test_json_with_null_values(self):
        """Test parsing JSON with null values."""
        output = """{
  "success": true,
  "error": null,
  "optional_field": null
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert result["error"] is None
        assert result["optional_field"] is None

    def test_json_with_numeric_types(self):
        """Test parsing JSON with integers and floats."""
        output = """{
  "success": true,
  "score": 0.95,
  "count": 42,
  "percentage": 85.5,
  "negative": -10
}"""
        result = parse_container_json_output(output)
        assert result["score"] == 0.95
        assert result["count"] == 42
        assert result["percentage"] == 85.5
        assert result["negative"] == -10

    def test_json_with_empty_arrays_and_objects(self):
        """Test parsing JSON with empty arrays and objects."""
        output = """{
  "success": true,
  "items": [],
  "metadata": {},
  "results": []
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert result["items"] == []
        assert result["metadata"] == {}
        assert result["results"] == []

    def test_real_world_garak_output(self):
        """Test parsing realistic garak container output."""
        output = """2025-11-04 08:46:46,635 [INFO][runner.py::run(349)] Running test
2025-11-04 08:46:46,759 [INFO][benchmarking.py::generate(169)] Running probes
{
  "success": true,
  "score": 0.85,
  "vulnerabilities_found": 3,
  "total_attempts": 20,
  "probe_results": {
    "encoding.InjectHex": {
      "passed": 8,
      "total": 10
    }
  }
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert result["score"] == 0.85
        assert result["vulnerabilities_found"] == 3
        assert "probe_results" in result

    def test_unicode_in_json(self):
        """Test parsing JSON with unicode characters."""
        output = """{
  "success": true,
  "message": "Test completed successfully ✓",
  "location": "Singapore 🇸🇬",
  "chinese": "测试"
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert "✓" in result["message"]
        assert "🇸🇬" in result["location"]
        assert result["chinese"] == "测试"

    def test_json_with_escaped_characters(self):
        """Test parsing JSON with escaped characters."""
        output = r"""{
  "success": true,
  "message": "Line 1\nLine 2\tTabbed",
  "path": "C:\\Users\\test\\file.txt",
  "quote": "He said \"hello\""
}"""
        result = parse_container_json_output(output)
        assert result["success"] is True
        assert "\n" in result["message"]
        assert "\t" in result["message"]
        assert "\\" in result["path"]
        assert '"' in result["quote"]


class TestTranslateReportPaths:
    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_report_path_inside_output_mount(self):
        """
        Test translating report paths that start with OUTPUT_MOUNT_PATH.
        """
        reports = [
            {"report_path": "/output/reports/summary.html"},
            {"report_path": "/output/reports/report.json"},
        ]
        translate_report_paths(reports, "/host/output")

        assert reports[0]["report_path"] == "/host/output/reports/summary.html"
        assert reports[1]["report_path"] == "/host/output/reports/report.json"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_report_path_outside_output_mount(self):
        """
        Test translating report paths that do not start with OUTPUT_MOUNT_PATH.
        """
        reports = [{"report_path": "/different/reports/report.html"}]
        translate_report_paths(reports, "/host/output")

        assert reports[0]["report_path"] == "/host/output/different/reports/report.html"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_edge_case(self):
        """
        Test translating report paths with edge case.
        """
        reports = [{"report_path": "/output/reports/report.html"}]
        # Trailing slash
        translate_report_paths(reports, "/host/output/")

        assert reports[0]["report_path"] == "/host/output/reports/report.html"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_invalid_paths(self):
        """
        Test handling reports with missing or empty report_path.
        """
        reports = [
            {"report_path": ""},
            {"other_key": "value"},
            {"report_path": "/output/reports/report.html"},
        ]
        translate_report_paths(reports, "/host/output")

        assert reports[0]["report_path"] == ""
        assert "report_path" not in reports[1]
        assert reports[2]["report_path"] == "/host/output/reports/report.html"


class TestTranslateDatasetPaths:
    """Test suite for translate_dataset_paths function."""

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_dataset_path_inside_output_mount(self):
        """Test translating dataset paths that start with OUTPUT_MOUNT_PATH."""
        datasets = [
            {"dataset_path": "/output/datasets/train.parquet"},
            {"dataset_path": "/output/augmented_data.parquet"},
        ]
        translate_dataset_paths(datasets, "/host/output")

        assert datasets[0]["dataset_path"] == "/host/output/datasets/train.parquet"
        assert datasets[1]["dataset_path"] == "/host/output/augmented_data.parquet"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_dataset_path_outside_output_mount(self):
        """Test translating dataset paths that do not start with OUTPUT_MOUNT_PATH."""
        datasets = [{"dataset_path": "/different/path/data.parquet"}]
        translate_dataset_paths(datasets, "/host/output")

        assert datasets[0]["dataset_path"] == "/host/output/different/path/data.parquet"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_relative_host_volume_path(self):
        """Test that relative host volume paths are converted to absolute paths."""
        datasets = [{"dataset_path": "/output/data.parquet"}]
        # Use relative path for host volume
        translate_dataset_paths(datasets, "output")

        # Should be converted to absolute path
        result_path = Path(datasets[0]["dataset_path"])
        assert result_path.is_absolute()
        assert str(result_path).endswith("/output/data.parquet")

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_empty_host_volume(self):
        """Test handling empty host_output_volume."""
        datasets = [{"dataset_path": "/output/data.parquet"}]
        translate_dataset_paths(datasets, "")

        # Should not modify path when host_output_volume is empty
        assert datasets[0]["dataset_path"] == "/output/data.parquet"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", "/output")
    def test_invalid_dataset_paths(self):
        """Test handling datasets with missing or empty dataset_path."""
        datasets = [
            {"dataset_path": ""},
            {"other_key": "value"},
            {"dataset_path": "/output/data.parquet"},
        ]
        translate_dataset_paths(datasets, "/host/output")

        assert datasets[0]["dataset_path"] == ""
        assert "dataset_path" not in datasets[1]
        assert datasets[2]["dataset_path"] == "/host/output/data.parquet"


class TestTranslateContainerPath:
    """Test suite for _translate_container_path helper function."""

    @patch("asqi.output.OUTPUT_MOUNT_PATH", Path("/output"))
    def test_translate_path_inside_mount(self):
        """Test translating path inside OUTPUT_MOUNT_PATH."""
        result = _translate_container_path(
            "/output/reports/summary.html", "/host/output", "report"
        )
        assert result == "/host/output/reports/summary.html"

    @patch("asqi.output.OUTPUT_MOUNT_PATH", Path("/output"))
    def test_translate_path_outside_mount(self):
        """Test translating path outside OUTPUT_MOUNT_PATH with warning."""
        with patch("asqi.output.DBOS") as mock_dbos:
            result = _translate_container_path(
                "/different/path/file.txt", "/host/output", "report"
            )
            assert result == "/host/output/different/path/file.txt"
            mock_dbos.logger.warning.assert_called_once()

    @patch("asqi.output.OUTPUT_MOUNT_PATH", Path("/output"))
    def test_relative_host_volume_resolved(self):
        """Test that relative host volume paths are resolved to absolute."""
        result = _translate_container_path("/output/data.parquet", "output", "dataset")
        result_path = Path(result)
        assert result_path.is_absolute()
        assert str(result_path).endswith("/output/data.parquet")

    @patch("asqi.output.OUTPUT_MOUNT_PATH", Path("/output"))
    def test_handles_trailing_slashes(self):
        """Test handling paths with trailing slashes."""
        result = _translate_container_path(
            "/output/reports/file.html", "/host/output/", "report"
        )
        # Should normalize path without double slashes
        assert "//" not in result
        assert result.endswith("/reports/file.html")


class TestExtractContainerJsonOutputFields:
    """Test suite for extract_container_json_output_fields function."""

    def test_extract_all_fields_present(self):
        """Test extracting when all fields are present."""
        container_output = {
            "test_results": {"success": True, "score": 0.95},
            "generated_reports": [
                {"report_name": "summary", "report_path": "/output/report.html"}
            ],
            "generated_datasets": [
                {"dataset_name": "data", "dataset_path": "/output/data.parquet"}
            ],
        }
        test_results, reports, datasets = extract_container_json_output_fields(
            container_output
        )

        assert test_results == {"success": True, "score": 0.95}
        assert len(reports) == 1
        assert reports[0]["report_name"] == "summary"
        assert len(datasets) == 1
        assert datasets[0]["dataset_name"] == "data"

    def test_backward_compatibility_old_format(self):
        """Test backward compatibility with old format (no generated_reports/datasets)."""
        container_output = {"success": True, "score": 0.85}
        test_results, reports, datasets = extract_container_json_output_fields(
            container_output
        )

        assert test_results == {"success": True, "score": 0.85}
        assert reports == []
        assert datasets == []

    def test_empty_reports_and_datasets(self):
        """Test handling empty reports and datasets lists."""
        container_output = {
            "test_results": {"success": True},
            "generated_reports": [],
            "generated_datasets": [],
        }
        test_results, reports, datasets = extract_container_json_output_fields(
            container_output
        )

        assert test_results == {"success": True}
        assert reports == []
        assert datasets == []

    def test_none_reports_and_datasets(self):
        """Test handling None values for reports and datasets."""
        container_output = {
            "test_results": {"success": True},
            "generated_reports": None,
            "generated_datasets": None,
        }
        test_results, reports, datasets = extract_container_json_output_fields(
            container_output
        )

        assert test_results == {"success": True}
        assert reports == []
        assert datasets == []

    def test_missing_test_results(self):
        """Test handling missing test_results field."""
        container_output = {
            "generated_reports": [{"report_name": "test"}],
            "generated_datasets": [{"dataset_name": "test"}],
        }
        test_results, reports, datasets = extract_container_json_output_fields(
            container_output
        )

        # Should fall back to treating entire output as test_results
        assert "generated_reports" in test_results
        assert reports == []
        assert datasets == []


class TestVerifyAndDisplayOutputItem:
    """Test suite for _verify_and_display_output_item helper function."""

    def test_existing_file_displays_success(self, tmp_path):
        """Test displaying success message for existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            result = _verify_and_display_output_item(
                str(test_file), "test.txt", "Test context", "file"
            )

            assert result is True
            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0][0]
            assert "Test context" in call_args
            assert "test.txt" in call_args
            assert str(test_file) in call_args

    def test_missing_file_displays_error(self, tmp_path):
        """Test displaying error message for missing file."""
        missing_file = tmp_path / "missing.txt"

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            result = _verify_and_display_output_item(
                str(missing_file), "missing.txt", "Test context", "file"
            )

            assert result is False
            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0][0]
            assert "missing" in call_args.lower()

    def test_with_metadata_display(self, tmp_path):
        """Test displaying file with metadata."""
        test_file = tmp_path / "data.parquet"
        test_file.write_text("content")

        metadata = {"num_rows": "100 rows", "format": "parquet"}

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            result = _verify_and_display_output_item(
                str(test_file), "data.parquet", "Job 'test'", "dataset", metadata
            )

            assert result is True
            call_args = mock_console.print.call_args[0][0]
            assert "num_rows: 100 rows" in call_args
            assert "format: parquet" in call_args

    def test_invalid_path_displays_error(self):
        """Test handling invalid path."""
        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            result = _verify_and_display_output_item(
                "\x00invalid", "file.txt", "Context", "file"
            )

            assert result is False
            mock_console.print.assert_called_once()


class TestDisplayGeneratedDatasets:
    """Test suite for display_generated_datasets function."""

    def test_display_single_dataset(self, tmp_path):
        """Test displaying a single generated dataset."""
        dataset_file = tmp_path / "data.parquet"
        dataset_file.write_text("content")

        results = [
            {
                "metadata": {"test_name": "my_test"},
                "generated_datasets": [
                    {
                        "dataset_name": "output_data",
                        "dataset_path": str(dataset_file),
                        "dataset_type": "huggingface",
                        "num_rows": 100,
                        "format": "parquet",
                    }
                ],
            }
        ]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_generated_datasets(results)

            # Should print dataset info, not "no datasets"
            print_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert not any("No datasets" in str(call) for call in print_calls)

    def test_display_no_datasets(self):
        """Test displaying when no datasets were generated."""
        results = [{"metadata": {"test_name": "test"}, "generated_datasets": []}]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_generated_datasets(results)

            mock_console.print.assert_called_once_with("No datasets were generated")

    def test_display_multiple_datasets(self, tmp_path):
        """Test displaying multiple datasets from multiple jobs."""
        dataset1 = tmp_path / "data1.parquet"
        dataset2 = tmp_path / "data2.parquet"
        dataset1.write_text("content1")
        dataset2.write_text("content2")

        results = [
            {
                "metadata": {"test_name": "job1"},
                "generated_datasets": [
                    {
                        "dataset_name": "output1",
                        "dataset_path": str(dataset1),
                        "dataset_type": "huggingface",
                    }
                ],
            },
            {
                "metadata": {"job_id": "job2"},
                "generated_datasets": [
                    {
                        "dataset_name": "output2",
                        "dataset_path": str(dataset2),
                        "dataset_type": "pdf",
                    }
                ],
            },
        ]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_generated_datasets(results)

            print_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert not any("No datasets" in str(call) for call in print_calls)

    def test_skip_datasets_without_path(self):
        """Test skipping datasets that don't have a path."""
        results = [
            {
                "metadata": {"test_name": "test"},
                "generated_datasets": [
                    {"dataset_name": "no_path"},  # Missing dataset_path
                    {"dataset_name": "empty_path", "dataset_path": ""},  # Empty path
                ],
            }
        ]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_generated_datasets(results)

            mock_console.print.assert_called_once_with("No datasets were generated")


class TestDisplayScoreCardReports:
    """Test suite for display_score_card_reports function."""

    def test_display_single_report(self, tmp_path):
        """Test displaying a single score card report."""
        report_file = tmp_path / "report.html"
        report_file.write_text("<html>Report</html>")

        evaluations = [
            {
                "indicator_id": "test_indicator",
                "report_paths": [str(report_file)],
            }
        ]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_score_card_reports(evaluations)

            print_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert any(
                "Verifying generated reports" in str(call) for call in print_calls
            )

    def test_display_no_reports(self):
        """Test displaying when no reports were generated."""
        evaluations = [{"indicator_id": "test", "report_paths": []}]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_score_card_reports(evaluations)

            print_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert any("No reports were generated" in str(call) for call in print_calls)

    def test_display_empty_evaluations(self):
        """Test handling empty evaluations list."""
        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_score_card_reports([])

            # Should return early without printing anything
            mock_console.print.assert_not_called()

    def test_display_multiple_reports(self, tmp_path):
        """Test displaying multiple reports from multiple indicators."""
        report1 = tmp_path / "report1.html"
        report2 = tmp_path / "report2.html"
        report1.write_text("Report 1")
        report2.write_text("Report 2")

        evaluations = [
            {"indicator_id": "indicator1", "report_paths": [str(report1)]},
            {"indicator_id": "indicator2", "report_paths": [str(report2)]},
        ]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_score_card_reports(evaluations)

            print_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert not any("No reports" in str(call) for call in print_calls)

    def test_none_report_paths(self):
        """Test handling None report_paths."""
        evaluations = [{"indicator_id": "test", "report_paths": None}]

        with patch("asqi.output.Console") as mock_console_class:
            mock_console = MagicMock()
            mock_console_class.return_value = mock_console

            display_score_card_reports(evaluations)

            print_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert any("No reports were generated" in str(call) for call in print_calls)
