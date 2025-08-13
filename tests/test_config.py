import pytest

from asqi.config import load_config_file, save_results_to_file


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
