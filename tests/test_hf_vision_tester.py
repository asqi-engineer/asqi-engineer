import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The entrypoint module imports torch and torchmetrics which are not installed
# in the test environment. We must import asqi.datasets first (which transitively
# imports the HF datasets library that probes for torch), then mock torch/torchmetrics
# before loading the entrypoint module.
import asqi.datasets  # noqa: F401, E402 -- force datasets lib to load first

_MOCK_MODULES = [
    "torch",
    "torchmetrics",
    "torchmetrics.detection",
    "torchmetrics.detection.mean_ap",
]
_mocked_names: list[str] = []
for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
        _mocked_names.append(_mod_name)

# Import entrypoint.py from test_containers/hf_vision_tester/ using importlib
# since it's not a regular package.
_ENTRYPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "test_containers"
    / "hf_vision_tester"
    / "entrypoint.py"
)
_spec = importlib.util.spec_from_file_location("hf_vision_entrypoint", _ENTRYPOINT_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

# Remove mocked modules so they don't pollute other tests (e.g. datasets/dill
# checks "torch" in sys.modules and then calls issubclass(..., torch.Tensor)).
for _mod_name in _mocked_names:
    sys.modules.pop(_mod_name, None)

normalize_label_map = _module.normalize_label_map
write_report = _module.write_report
load_samples = _module.load_samples
parse_config = _module.parse_config


class TestNormalizeLabelMap:
    """Tests for normalize_label_map: converts {id: name} to {name: id}."""

    def test_int_keys_to_name_id(self):
        """Integer keys with string values -> {name: id}."""
        result = normalize_label_map({0: "person", 1: "car"})
        assert result == {"person": 0, "car": 1}

    def test_string_values_passthrough(self):
        """String keys with int values are already {name: id} format."""
        result = normalize_label_map({"person": 0, "car": 1})
        assert result == {"person": 0, "car": 1}

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert normalize_label_map({}) == {}

    def test_string_int_keys(self):
        """String keys like "0" with string values -> int conversion."""
        result = normalize_label_map({"0": "person"})
        assert result == {"person": 0}

    def test_labels_lowercased(self):
        """Label names should be lowercased."""
        result = normalize_label_map({0: "Person", 1: "CAR"})
        assert result == {"person": 0, "car": 1}

    def test_string_key_names_lowercased(self):
        """When input is {name: id}, names should be lowercased."""
        result = normalize_label_map({"Person": 0, "CAR": 1})
        assert result == {"person": 0, "car": 1}


class TestWriteReport:
    """Tests for write_report: writes HTML report to OUTPUT_MOUNT_PATH."""

    def test_returns_none_when_output_path_not_set(self, monkeypatch):
        """Returns None when OUTPUT_MOUNT_PATH env var is not set."""
        monkeypatch.delenv("OUTPUT_MOUNT_PATH", raising=False)
        result = write_report({"map": 0.5}, "model-id", "dataset-name")
        assert result is None

    def test_writes_html_when_output_path_set(self, tmp_path, monkeypatch):
        """Writes HTML report file when OUTPUT_MOUNT_PATH is set."""
        monkeypatch.setenv("OUTPUT_MOUNT_PATH", str(tmp_path))
        metrics = {"map": 0.5432, "map_50": 0.7654}

        result = write_report(metrics, "facebook/detr-resnet-50", "eval_data")

        assert result is not None
        assert result["report_type"] == "html"
        assert result["report_name"] == "detection_report"

        report_path = Path(result["report_path"])
        assert report_path.exists()
        html_content = report_path.read_text()
        assert "facebook/detr-resnet-50" in html_content
        assert "eval_data" in html_content
        assert "map" in html_content
        assert "0.5432" in html_content

    def test_creates_reports_directory(self, tmp_path, monkeypatch):
        """Creates the reports/ subdirectory if it doesn't exist."""
        monkeypatch.setenv("OUTPUT_MOUNT_PATH", str(tmp_path))
        write_report({"map": 0.1}, "model", "ds")
        assert (tmp_path / "reports").is_dir()


class TestLoadSamples:
    """Tests for load_samples: streams dataset samples via load_hf_dataset."""

    @patch.object(_module, "load_hf_dataset")
    def test_sets_streaming_true(self, mock_load):
        """load_samples should set streaming=True on the config."""
        mock_load.return_value = iter([{"image": "a"}, {"image": "b"}])
        config = {"loader_params": {"hub_path": "test/dataset"}}

        list(load_samples(config))

        assert config["loader_params"]["streaming"] is True
        mock_load.assert_called_once_with(config)

    @patch.object(_module, "load_hf_dataset")
    def test_yields_all_samples(self, mock_load):
        """Yields all samples when max_samples is None."""
        samples = [{"image": f"img_{i}"} for i in range(5)]
        mock_load.return_value = iter(samples)

        result = list(load_samples({"loader_params": {}}))
        assert len(result) == 5

    @patch.object(_module, "load_hf_dataset")
    def test_respects_max_samples(self, mock_load):
        """Stops yielding after max_samples items."""
        samples = [{"image": f"img_{i}"} for i in range(10)]
        mock_load.return_value = iter(samples)

        result = list(load_samples({"loader_params": {}}, max_samples=3))
        assert len(result) == 3

    @patch.object(_module, "load_hf_dataset")
    def test_creates_loader_params_key_if_missing(self, mock_load):
        """setdefault creates loader_params if not present in config."""
        mock_load.return_value = iter([])
        config = {}

        list(load_samples(config))

        assert config["loader_params"]["streaming"] is True

    @patch.object(_module, "load_hf_dataset")
    def test_empty_dataset_yields_nothing(self, mock_load):
        """An empty streaming dataset yields zero samples."""
        mock_load.return_value = iter([])

        result = list(load_samples({"loader_params": {}}))
        assert result == []

    @patch.object(_module, "load_hf_dataset")
    def test_max_samples_larger_than_dataset(self, mock_load):
        """max_samples larger than dataset size yields all available samples."""
        samples = [{"image": f"img_{i}"} for i in range(3)]
        mock_load.return_value = iter(samples)

        result = list(load_samples({"loader_params": {}}, max_samples=100))
        assert len(result) == 3

    @patch.object(_module, "load_hf_dataset")
    def test_max_samples_one(self, mock_load):
        """max_samples=1 yields exactly one sample."""
        samples = [{"image": f"img_{i}"} for i in range(5)]
        mock_load.return_value = iter(samples)

        result = list(load_samples({"loader_params": {}}, max_samples=1))
        assert len(result) == 1
        assert result[0] == {"image": "img_0"}

    @patch.object(_module, "load_hf_dataset")
    def test_preserves_existing_loader_params(self, mock_load):
        """Existing loader_params keys are preserved when streaming is set."""
        mock_load.return_value = iter([{"image": "a"}])
        config = {"loader_params": {"hub_path": "org/repo", "split": "test"}}

        list(load_samples(config))

        assert config["loader_params"]["hub_path"] == "org/repo"
        assert config["loader_params"]["split"] == "test"
        assert config["loader_params"]["streaming"] is True


class TestParseConfig:
    """Tests for parse_config: extracts configuration from sys/test params."""

    def _make_sys_params(self, **overrides):
        sut = {
            "type": "hf_inference_api",
            "model_id": "facebook/detr-resnet-50",
            "api_key": "hf_test_key",
            "timeout": 30.0,
        }
        sut.update(overrides)
        return {"system_under_test": sut}

    def _make_test_params(self, **overrides):
        params = {
            "input_datasets": {
                "evaluation_data": {
                    "type": "huggingface",
                    "loader_params": {"hub_path": "test/dataset"},
                }
            },
            "confidence_threshold": 0.5,
            "iou_threshold": 0.5,
        }
        params.update(overrides)
        return params

    def test_extracts_ds_config(self):
        """parse_config extracts ds_config from test_params correctly."""
        sys_params = self._make_sys_params()
        test_params = self._make_test_params()

        config = parse_config(sys_params, test_params)

        assert config["ds_config"] == test_params["input_datasets"]["evaluation_data"]
        assert config["model_id"] == "facebook/detr-resnet-50"
        assert config["api_key"] == "hf_test_key"
        assert config["timeout"] == 30.0
        assert config["conf_threshold"] == 0.5
        assert config["iou_threshold"] == 0.5

    def test_raises_for_wrong_system_type(self):
        """parse_config raises ValueError for non-hf_inference_api type."""
        sys_params = {"system_under_test": {"type": "openai_api"}}
        test_params = self._make_test_params()

        with pytest.raises(ValueError, match="Expected hf_inference_api"):
            parse_config(sys_params, test_params)

    def test_label_map_from_ds_config(self):
        """label_map is extracted from ds_config when present."""
        sys_params = self._make_sys_params()
        test_params = self._make_test_params()
        test_params["input_datasets"]["evaluation_data"]["label_map"] = {
            0: "person",
            1: "car",
        }

        config = parse_config(sys_params, test_params)
        assert config["label_map"] == {0: "person", 1: "car"}

    def test_label_map_fallback_to_test_params(self):
        """label_map falls back to test_params when not in ds_config."""
        sys_params = self._make_sys_params()
        test_params = self._make_test_params()
        test_params["label_map"] = {0: "dog"}

        config = parse_config(sys_params, test_params)
        assert config["label_map"] == {0: "dog"}

    def test_defaults_when_missing(self):
        """Uses defaults for model_id, timeout, bbox_format when not specified."""
        sys_params = {"system_under_test": {"type": "hf_inference_api"}}
        test_params = {"input_datasets": {"evaluation_data": {}}}

        config = parse_config(sys_params, test_params)

        assert config["model_id"] == "facebook/detr-resnet-50"
        assert config["timeout"] == 30.0
        assert config["bbox_format"] == "xyxy"
        assert config["max_samples"] is None

    def test_api_key_from_env(self, monkeypatch):
        """api_key falls back to HF_TOKEN env var when not in config."""
        monkeypatch.setenv("HF_TOKEN", "env_token_123")
        sys_params = {"system_under_test": {"type": "hf_inference_api"}}
        test_params = {"input_datasets": {"evaluation_data": {}}}

        config = parse_config(sys_params, test_params)
        assert config["api_key"] == "env_token_123"
