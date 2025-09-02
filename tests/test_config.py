import copy
import os

import pytest

from asqi.config import (
    interpolate_env_vars,
    load_config_file,
    merge_defaults_into_suite,
    save_results_to_file,
)


def test_load_config_file_valid(tmp_path):
    yaml_content = "key: value"
    file = tmp_path / "test.yaml"
    file.write_text(yaml_content)
    result = load_config_file(str(file))
    assert result == {"key": "value"}


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config_file("nonexistent.yaml")


def test_load_config_file_invalid_yaml(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text(": bad yaml")
    with pytest.raises(Exception):
        load_config_file(str(file))


def test_save_results_to_file(tmp_path):
    data = {"foo": 123, "bar": [1, 2, 3]}
    out_file = tmp_path / "results.json"
    save_results_to_file(data, str(out_file))
    import json

    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded == data


def test_merge_defaults_into_suite_applies_defaults():
    """Ensure test_suite_default values are merged into each test entry."""

    config = {
        "test_suite_default": {
            "image": "default-image",
            "params": {"timeout": 10},
        },
        "test_suite": [
            {"name": "test1"},
            {"name": "test2", "params": {"timeout": 20}},
        ],
    }

    merged = merge_defaults_into_suite(copy.deepcopy(config))

    # Test1 should inherit all defaults
    t1 = merged["test_suite"][0]
    assert t1["image"] == "default-image"
    assert t1["params"]["timeout"] == 10

    # Test2 should override timeout but still inherit image
    t2 = merged["test_suite"][1]
    assert t2["image"] == "default-image"
    assert t2["params"]["timeout"] == 20


def test_merge_defaults_into_suite_without_defaults():
    """If no test_suite_default exists, config should remain unchanged."""

    config = {"suite_name": "No Defaults", "test_suite": [{"name": "test1"}]}
    merged = merge_defaults_into_suite(copy.deepcopy(config))
    assert merged == config


def test_interpolate_env_vars_direct_substitution():
    """Test direct environment variable substitution ${VAR}."""
    os.environ["TEST_VAR"] = "test_value"
    try:
        data = {"key": "${TEST_VAR}"}
        result = interpolate_env_vars(data)
        assert result == {"key": "test_value"}
    finally:
        del os.environ["TEST_VAR"]


def test_interpolate_env_vars_with_default_non_empty():
    """Test ${VAR:-default} when VAR is set and non-empty."""
    os.environ["TEST_VAR"] = "actual_value"
    try:
        data = {"key": "${TEST_VAR:-default_value}"}
        result = interpolate_env_vars(data)
        assert result == {"key": "actual_value"}
    finally:
        del os.environ["TEST_VAR"]


def test_interpolate_env_vars_with_default_empty():
    """Test ${VAR:-default} when VAR is set but empty."""
    os.environ["TEST_VAR"] = ""
    try:
        data = {"key": "${TEST_VAR:-default_value}"}
        result = interpolate_env_vars(data)
        assert result == {"key": "default_value"}
    finally:
        del os.environ["TEST_VAR"]


def test_interpolate_env_vars_with_default_unset():
    """Test ${VAR:-default} when VAR is not set."""
    data = {"key": "${UNSET_VAR:-default_value}"}
    result = interpolate_env_vars(data)
    assert result == {"key": "default_value"}


def test_interpolate_env_vars_default_if_unset():
    """Test ${VAR-default} when VAR is not set."""
    data = {"key": "${UNSET_VAR-default_value}"}
    result = interpolate_env_vars(data)
    assert result == {"key": "default_value"}


def test_interpolate_env_vars_default_if_unset_empty():
    """Test ${VAR-default} when VAR is set but empty."""
    os.environ["TEST_VAR"] = ""
    try:
        data = {"key": "${TEST_VAR-default_value}"}
        result = interpolate_env_vars(data)
        assert result == {"key": ""}
    finally:
        del os.environ["TEST_VAR"]


def test_interpolate_env_vars_unset_no_default():
    """Test ${VAR} when VAR is not set (should return empty string)."""
    data = {"key": "${UNSET_VAR}"}
    result = interpolate_env_vars(data)
    assert result == {"key": ""}


def test_interpolate_env_vars_nested_structures():
    """Test interpolation in nested dictionaries and lists."""
    os.environ["TEST_VAR"] = "interpolated"
    try:
        data = {
            "dict_key": {"nested": "${TEST_VAR}"},
            "list_key": ["item1", "${TEST_VAR}", "item3"],
            "simple": "${TEST_VAR}",
        }
        result = interpolate_env_vars(data)
        expected = {
            "dict_key": {"nested": "interpolated"},
            "list_key": ["item1", "interpolated", "item3"],
            "simple": "interpolated",
        }
        assert result == expected
    finally:
        del os.environ["TEST_VAR"]


def test_interpolate_env_vars_no_interpolation():
    """Test that data without interpolation placeholders is unchanged."""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    result = interpolate_env_vars(data)
    assert result == data


def test_load_config_file_with_interpolation(tmp_path):
    """Test that load_config_file applies interpolation."""
    os.environ["TEST_REGISTRY"] = "my-registry.com"

    # Store and temporarily unset API_KEY to test default value
    original_api_key = os.environ.pop("API_KEY", None)

    try:
        yaml_content = """
        image: "${TEST_REGISTRY}/my-app:latest"
        params:
          api_key: "${API_KEY:-sk-default}"
        """
        file = tmp_path / "test.yaml"
        file.write_text(yaml_content)
        result = load_config_file(str(file))
        expected = {
            "image": "my-registry.com/my-app:latest",
            "params": {"api_key": "sk-default"},
        }
        assert result == expected
    finally:
        del os.environ["TEST_REGISTRY"]
        # Restore original API_KEY if it existed
        if original_api_key is not None:
            os.environ["API_KEY"] = original_api_key
