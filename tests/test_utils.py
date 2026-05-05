"""Tests for asqi.utils module."""

from __future__ import annotations

import pytest

from asqi.schemas import ExecutionMetadata
from asqi.utils import get_openai_tracking_kwargs


class TestGetOpenAITrackingKwargs:
    """Tests for get_openai_tracking_kwargs function."""

    @pytest.mark.parametrize(
        "metadata,expected",
        [
            (None, {}),
            ({}, {}),
            ({"user_id": ""}, {}),
            ({"tags": {}}, {}),
            ({"tags": None}, {}),
        ],
    )
    def test_empty_inputs_return_empty_dict(self, metadata, expected):
        """Test that empty/None inputs return empty dict (no extra_body created)."""
        result = get_openai_tracking_kwargs(metadata)
        assert result == expected

    def test_user_id_only(self):
        """Test with only user_id, no tags or extra fields."""
        metadata = {"user_id": "user123"}
        result = get_openai_tracking_kwargs(metadata)
        assert result == {"user": "user123"}

    def test_tags_only(self):
        """Test with only tags, no user_id."""
        metadata = {"tags": {"job_id": "test-001", "job_type": "test"}}
        result = get_openai_tracking_kwargs(metadata)
        assert result == {
            "extra_body": {"metadata": {"tags": ["job_id:test-001", "job_type:test"]}}
        }

    def test_tags_with_none_values_skipped(self):
        """Test that tags with None values are skipped."""
        metadata = {
            "tags": {"job_id": "test-001", "parent_id": None, "job_type": "test"}
        }
        result = get_openai_tracking_kwargs(metadata)
        expected_tags = ["job_id:test-001", "job_type:test"]
        assert result["extra_body"]["metadata"]["tags"] == expected_tags

    @pytest.mark.parametrize(
        "tag_value,expected_tag",
        [
            ({"model": "gpt-4"}, "config:{'model': 'gpt-4'}"),
            (["gpt-4", "gpt-3.5"], "models:['gpt-4', 'gpt-3.5']"),
            ((10, 20), "coords:(10, 20)"),
        ],
    )
    def test_complex_tag_values_stringified(self, tag_value, expected_tag):
        """Test that complex tag values (dict/list/tuple) are stringified."""
        tag_key = (
            "config"
            if isinstance(tag_value, dict)
            else ("models" if isinstance(tag_value, list) else "coords")
        )
        metadata = {"tags": {tag_key: tag_value}}
        result = get_openai_tracking_kwargs(metadata)
        assert expected_tag in result["extra_body"]["metadata"]["tags"]

    def test_non_dict_tags_fail_safe(self):
        """Test that non-dict tags are wrapped in a dict safely."""
        metadata = {"tags": "invalid_tags_string"}
        result = get_openai_tracking_kwargs(metadata)
        assert result["extra_body"]["metadata"]["tags"] == ["tags:invalid_tags_string"]

    def test_custom_fields_included_in_extra_body(self):
        """Test that custom metadata fields are included in extra_body.metadata."""
        metadata = {"experiment_id": "exp-001", "model_version": "v2"}
        result = get_openai_tracking_kwargs(metadata)
        assert result == {
            "extra_body": {
                "metadata": {
                    "experiment_id": "exp-001",
                    "model_version": "v2",
                }
            }
        }

    def test_pydantic_execution_metadata_input(self):
        """Test that ExecutionMetadata Pydantic model is converted correctly."""
        metadata = ExecutionMetadata(
            user_id="user456",
            tags={
                "job_id": "test-002",
                "job_type": "integration",
                "parent_id": "parent-456",
            },
        )
        result = get_openai_tracking_kwargs(metadata)
        assert result["user"] == "user456"
        assert "job_id:test-002" in result["extra_body"]["metadata"]["tags"]
        assert "job_type:integration" in result["extra_body"]["metadata"]["tags"]

    def test_full_example_with_all_fields(self):
        """Test complete example with user_id, tags, and custom fields."""
        metadata = {
            "user_id": "user123",
            "custom_field": "experiment_A",
            "tags": {"job_id": "test-001", "job_type": "test"},
        }
        result = get_openai_tracking_kwargs(metadata)
        assert result["user"] == "user123"
        assert result["extra_body"]["metadata"]["custom_field"] == "experiment_A"
        assert "job_id:test-001" in result["extra_body"]["metadata"]["tags"]
        # Verify reserved keys not duplicated
        assert "user_id" not in result["extra_body"]["metadata"]
        assert isinstance(result["extra_body"]["metadata"]["tags"], list)
