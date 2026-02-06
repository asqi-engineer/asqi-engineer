"""Tests for load_hf_dataset Hub/local dispatch and helper functions.

These tests mock datasets.load_dataset to verify dispatch logic,
token handling, column mapping, and input_mount_path resolution
without making real API calls.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import Dataset, IterableDataset

from asqi.datasets import load_hf_dataset
from asqi.schemas import DatasetFeature, DatasetLoaderParams, HFDatasetDefinition


def _make_hub_config(mapping=None, **loader_overrides):
    """Helper to create an HFDatasetDefinition with hub_path."""
    params = {"hub_path": "org/dataset", **loader_overrides}
    return HFDatasetDefinition(
        type="huggingface",
        loader_params=DatasetLoaderParams(**params),
        mapping=mapping or {},
    )


def _make_local_config(mapping=None, **loader_overrides):
    """Helper to create an HFDatasetDefinition with builder_name."""
    defaults = {"builder_name": "parquet", "data_files": "data.parquet"}
    defaults.update(loader_overrides)
    return HFDatasetDefinition(
        type="huggingface",
        loader_params=DatasetLoaderParams(**defaults),
        mapping=mapping or {},
    )


def _fake_dataset():
    """Return a minimal Dataset for mocking."""
    return Dataset.from_dict({"col_a": [1], "col_b": [2]})


class TestHubModeDispatch:
    """load_hf_dataset dispatches to _load_from_hub when hub_path is set."""

    @patch("asqi.datasets.load_dataset")
    def test_hub_path_calls_load_dataset_with_hub_path(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config()

        load_hf_dataset(cfg)

        mock_ld.assert_called_once()
        call_kwargs = mock_ld.call_args
        assert call_kwargs.kwargs["path"] == "org/dataset"

    @patch("asqi.datasets.load_dataset")
    def test_hub_mode_passes_name_split_revision(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config(name="v2", split="validation", revision="abc123")

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["name"] == "v2"
        assert kw["split"] == "validation"
        assert kw["revision"] == "abc123"

    @patch("asqi.datasets.load_dataset")
    def test_hub_mode_default_split_is_train(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config()

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["split"] == "train"


class TestLocalModeDispatch:
    """load_hf_dataset dispatches to _load_from_local when builder_name is set."""

    @patch("asqi.datasets.load_dataset")
    def test_local_builder_calls_load_dataset_with_builder_name(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config()

        load_hf_dataset(cfg)

        mock_ld.assert_called_once()
        kw = mock_ld.call_args.kwargs
        assert kw["path"] == "parquet"

    @patch("asqi.datasets.load_dataset")
    def test_local_mode_passes_data_files(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config(data_files="my_data.parquet")

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["data_files"] == "my_data.parquet"

    @patch("asqi.datasets.load_dataset")
    def test_local_mode_passes_data_dir(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config(data_files=None, data_dir="subdir")

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["data_dir"] == "subdir"


class TestHubTokenHandling:
    """Token resolution: config token > HF_TOKEN env var."""

    @patch("asqi.datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "env-token-123"}, clear=False)
    def test_falls_back_to_hf_token_env_var(self, mock_ld):
        """When no token in config, use HF_TOKEN env var."""
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config()

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["token"] == "env-token-123"

    @patch("asqi.datasets.load_dataset")
    @patch.dict("os.environ", {"HF_TOKEN": "env-token-123"}, clear=False)
    def test_config_token_overrides_env_var(self, mock_ld):
        """Config token takes precedence over env var."""
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config(token="config-token-456")

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["token"] == "config-token-456"

    @patch("asqi.datasets.load_dataset")
    @patch.dict("os.environ", {}, clear=False)
    def test_no_token_anywhere_passes_none(self, mock_ld):
        """When neither config nor env has a token, pass None."""
        mock_ld.return_value = _fake_dataset()
        # Remove HF_TOKEN if present
        import os

        os.environ.pop("HF_TOKEN", None)
        cfg = _make_hub_config()

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["token"] is None


class TestColumnMapping:
    """Column mapping is applied only when mapping dict is non-empty."""

    @patch("asqi.datasets.load_dataset")
    def test_empty_mapping_does_not_call_rename_columns(self, mock_ld):
        ds = _fake_dataset()
        ds.rename_columns = MagicMock()
        mock_ld.return_value = ds

        cfg = _make_local_config(mapping={})

        result = load_hf_dataset(cfg)

        ds.rename_columns.assert_not_called()
        assert result is ds

    @patch("asqi.datasets.load_dataset")
    def test_non_empty_mapping_calls_rename_columns(self, mock_ld):
        ds = _fake_dataset()
        renamed_ds = _fake_dataset()
        ds.rename_columns = MagicMock(return_value=renamed_ds)
        mock_ld.return_value = ds

        cfg = _make_local_config(mapping={"col_a": "column_alpha"})

        result = load_hf_dataset(cfg)

        ds.rename_columns.assert_called_once_with({"col_a": "column_alpha"})
        assert result is renamed_ds


class TestStreamingMode:
    """Streaming flag is passed through and returns IterableDataset."""

    @patch("asqi.datasets.load_dataset")
    def test_streaming_flag_passed_to_load_dataset_hub(self, mock_ld):
        mock_ld.return_value = MagicMock(spec=IterableDataset)
        cfg = _make_hub_config(streaming=True)

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["streaming"] is True

    @patch("asqi.datasets.load_dataset")
    def test_streaming_flag_passed_to_load_dataset_local(self, mock_ld):
        mock_ld.return_value = MagicMock(spec=IterableDataset)
        cfg = _make_local_config(streaming=True)

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["streaming"] is True

    @patch("asqi.datasets.load_dataset")
    def test_streaming_hub_returns_iterable_dataset(self, mock_ld):
        """load_hf_dataset with streaming=True returns an IterableDataset."""
        mock_iterable = MagicMock(spec=IterableDataset)
        mock_ld.return_value = mock_iterable
        cfg = _make_hub_config(streaming=True)

        result = load_hf_dataset(cfg)

        assert isinstance(result, IterableDataset)

    @patch("asqi.datasets.load_dataset")
    def test_streaming_local_returns_iterable_dataset(self, mock_ld):
        """load_hf_dataset local mode with streaming=True returns IterableDataset."""
        mock_iterable = MagicMock(spec=IterableDataset)
        mock_ld.return_value = mock_iterable
        cfg = _make_local_config(streaming=True)

        result = load_hf_dataset(cfg)

        assert isinstance(result, IterableDataset)

    @patch("asqi.datasets.validate_dataset_features")
    @patch("asqi.datasets.load_dataset")
    def test_streaming_skips_validate_dataset_features(self, mock_ld, mock_validate):
        """validate_dataset_features is not called for streaming datasets."""
        mock_ld.return_value = MagicMock(spec=IterableDataset)
        cfg = _make_hub_config(streaming=True)
        fake_features = [DatasetFeature(name="col_a", dtype="int64")]

        load_hf_dataset(cfg, expected_features=fake_features)

        mock_validate.assert_not_called()

    @patch("asqi.datasets.validate_dataset_features")
    @patch("asqi.datasets.load_dataset")
    def test_non_streaming_calls_validate_dataset_features(
        self, mock_ld, mock_validate
    ):
        """validate_dataset_features is called for non-streaming datasets."""
        ds = _fake_dataset()
        mock_ld.return_value = ds
        cfg = _make_hub_config(streaming=False)
        fake_features = [DatasetFeature(name="col_a", dtype="int64")]

        load_hf_dataset(cfg, expected_features=fake_features)

        mock_validate.assert_called_once_with(ds, fake_features, "dataset")

    @patch("asqi.datasets.load_dataset")
    def test_streaming_rename_columns_called_on_iterable_dataset(self, mock_ld):
        """rename_columns is called on IterableDataset when mapping is non-empty."""
        mock_iterable = MagicMock(spec=IterableDataset)
        renamed_iterable = MagicMock(spec=IterableDataset)
        mock_iterable.rename_columns.return_value = renamed_iterable
        mock_ld.return_value = mock_iterable
        cfg = _make_hub_config(streaming=True, mapping={"col_a": "alpha"})

        result = load_hf_dataset(cfg)

        mock_iterable.rename_columns.assert_called_once_with({"col_a": "alpha"})
        assert result is renamed_iterable

    @patch("asqi.datasets.load_dataset")
    def test_streaming_default_is_false(self, mock_ld):
        """streaming defaults to False when not specified."""
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config()

        load_hf_dataset(cfg)

        kw = mock_ld.call_args.kwargs
        assert kw["streaming"] is False


class TestInputMountPathHubMode:
    """input_mount_path is ignored in Hub mode."""

    @patch("asqi.datasets.load_dataset")
    def test_input_mount_path_ignored_in_hub_mode(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_hub_config()

        load_hf_dataset(cfg, input_mount_path=Path("/mnt/input"))

        kw = mock_ld.call_args.kwargs
        # Hub mode passes path=hub_path, no data_dir or data_files
        assert kw["path"] == "org/dataset"
        assert "data_dir" not in kw
        assert "data_files" not in kw


class TestInputMountPathLocalMode:
    """input_mount_path resolves data_dir and data_files in local mode."""

    @patch("asqi.datasets.load_dataset")
    def test_resolves_data_dir(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config(data_files=None, data_dir="images/train")

        load_hf_dataset(cfg, input_mount_path=Path("/mnt/input"))

        kw = mock_ld.call_args.kwargs
        assert kw["data_dir"] == "/mnt/input/images/train"

    @patch("asqi.datasets.load_dataset")
    def test_resolves_data_files_str(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config(data_files="data.parquet")

        load_hf_dataset(cfg, input_mount_path=Path("/mnt/input"))

        kw = mock_ld.call_args.kwargs
        assert kw["data_files"] == "/mnt/input/data.parquet"

    @patch("asqi.datasets.load_dataset")
    def test_resolves_data_files_list(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config(data_files=["a.parquet", "b.parquet"])

        load_hf_dataset(cfg, input_mount_path=Path("/mnt/input"))

        kw = mock_ld.call_args.kwargs
        assert kw["data_files"] == [
            "/mnt/input/a.parquet",
            "/mnt/input/b.parquet",
        ]

    @patch("asqi.datasets.load_dataset")
    def test_no_mount_path_leaves_paths_unchanged(self, mock_ld):
        mock_ld.return_value = _fake_dataset()
        cfg = _make_local_config(data_files="data.parquet")

        load_hf_dataset(cfg, input_mount_path=None)

        kw = mock_ld.call_args.kwargs
        assert kw["data_files"] == "data.parquet"
