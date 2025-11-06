"""
Unit tests for output.py module, focusing on JSON parsing from container output.
"""

import pytest

from asqi.output import parse_container_json_output


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
