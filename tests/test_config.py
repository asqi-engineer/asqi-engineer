import copy

import pytest

from asqi.config import (
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
