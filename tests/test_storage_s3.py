"""Unit tests for :mod:`asqi.storage.s3` pure S3 primitives.

All tests mock the boto3 client surface; no network calls or MinIO
container is required. See ``test_storage_s3_minio.py`` (out of scope
for AIP-2474) for the eventual MinIO integration suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from asqi.storage.s3 import (
    S3ClientConfig,
    download_file_to_path,
    download_prefix_to_folder,
    ensure_bucket_exists,
    make_s3_client,
    upload_file,
    upload_folder,
)
from botocore.exceptions import ClientError


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    """Reset the lru_cache between tests so each test sees fresh boto3 mocks."""
    make_s3_client.cache_clear()


class TestS3ClientConfig:
    def test_is_hashable_for_cache_key(self) -> None:
        cfg = S3ClientConfig(endpoint_url="http://m:9000", region="us-east-1")
        # Frozen dataclass; hashable means it can be used as an lru_cache key.
        assert hash(cfg) == hash(S3ClientConfig(endpoint_url="http://m:9000", region="us-east-1"))

    def test_distinct_configs_hash_differently(self) -> None:
        a = S3ClientConfig(endpoint_url="http://m:9000", region="us-east-1")
        b = S3ClientConfig(endpoint_url="http://m:9001", region="us-east-1")
        assert hash(a) != hash(b)

    def test_credentials_default_to_none(self) -> None:
        cfg = S3ClientConfig(endpoint_url="x", region="r")
        assert cfg.access_key is None
        assert cfg.secret_key is None


class TestMakeS3Client:
    def test_caches_clients_per_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        def fake_client(_service: str, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return MagicMock(name=f"s3-{len(calls)}")

        monkeypatch.setattr("asqi.storage.s3.boto3.client", fake_client)
        cfg = S3ClientConfig(endpoint_url="http://m:9000", region="us-east-1")
        a = make_s3_client(cfg)
        b = make_s3_client(cfg)
        assert a is b
        assert len(calls) == 1

    def test_omits_credentials_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_client(_service: str, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("asqi.storage.s3.boto3.client", fake_client)
        make_s3_client(S3ClientConfig(endpoint_url="http://m:9000", region="r"))
        assert "aws_access_key_id" not in captured
        assert "aws_secret_access_key" not in captured

    def test_includes_credentials_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_client(_service: str, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr("asqi.storage.s3.boto3.client", fake_client)
        make_s3_client(S3ClientConfig(endpoint_url="x", region="r", access_key="ak", secret_key="sk"))  # noqa: S106 — test fixture
        assert captured["aws_access_key_id"] == "ak"
        assert captured["aws_secret_access_key"] == "sk"  # noqa: S105 — boto3 kwarg name, not a secret


class TestEnsureBucketExists:
    def test_skips_create_when_head_succeeds(self) -> None:
        client = MagicMock()
        client.head_bucket.return_value = {}
        ensure_bucket_exists(client, "b", "us-east-1")
        client.create_bucket.assert_not_called()

    def test_creates_with_no_location_constraint_in_us_east_1(self) -> None:
        client = MagicMock()
        client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadBucket",
        )
        ensure_bucket_exists(client, "b", "us-east-1")
        client.create_bucket.assert_called_once_with(Bucket="b")

    def test_creates_with_location_constraint_outside_us_east_1(self) -> None:
        client = MagicMock()
        client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Not Found"}},
            "HeadBucket",
        )
        ensure_bucket_exists(client, "b", "ap-southeast-1")
        client.create_bucket.assert_called_once_with(
            Bucket="b",
            CreateBucketConfiguration={"LocationConstraint": "ap-southeast-1"},
        )

    def test_reraises_forbidden_head_bucket_without_create(self) -> None:
        client = MagicMock()
        client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadBucket",
        )
        with pytest.raises(ClientError):
            ensure_bucket_exists(client, "b", "us-east-1")
        client.create_bucket.assert_not_called()


class TestUploadFile:
    def test_uploads_with_guessed_content_type(self, tmp_path: Path) -> None:
        local = tmp_path / "data.json"
        local.write_text('{"k":"v"}')
        client = MagicMock()
        upload_file(client, local, "b", "wf/in/data.json")
        client.put_object.assert_called_once()
        kwargs = client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert kwargs["Key"] == "wf/in/data.json"
        assert kwargs["ContentType"] == "application/json"

    def test_uses_explicit_content_type(self, tmp_path: Path) -> None:
        local = tmp_path / "blob.bin"
        local.write_bytes(b"\x00\x01")
        client = MagicMock()
        upload_file(client, local, "b", "k", content_type="application/x-custom")
        assert client.put_object.call_args.kwargs["ContentType"] == "application/x-custom"

    def test_falls_back_to_octet_stream_for_unknown_extension(self, tmp_path: Path) -> None:
        local = tmp_path / "x.unknownext"
        local.write_text("hi")
        client = MagicMock()
        upload_file(client, local, "b", "k")
        assert client.put_object.call_args.kwargs["ContentType"] == "application/octet-stream"

    def test_raises_when_local_missing(self, tmp_path: Path) -> None:
        client = MagicMock()
        with pytest.raises(FileNotFoundError):
            upload_file(client, tmp_path / "nope", "b", "k")
        client.put_object.assert_not_called()


class TestDownloadFileToPath:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        client = MagicMock()

        def fake_download(*, Bucket: str, Key: str, Fileobj: Any) -> None:
            Fileobj.write(b"hello")

        client.download_fileobj.side_effect = fake_download
        target = tmp_path / "nested" / "deep" / "out.bin"
        download_file_to_path(client, "b", "k", target)
        assert target.read_bytes() == b"hello"


class TestUploadFolder:
    def test_uploads_files_under_prefix_preserving_relative_paths(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("b")
        client = MagicMock()
        keys = upload_folder(client, tmp_path, "bkt", "wf/in")
        assert keys == ["wf/in/a.txt", "wf/in/sub/b.txt"]
        assert client.put_object.call_count == 2

    def test_handles_empty_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        client = MagicMock()
        keys = upload_folder(client, tmp_path, "bkt", "")
        assert keys == ["a.txt"]

    def test_skips_non_files(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        client = MagicMock()
        keys = upload_folder(client, tmp_path, "bkt", "p")
        assert keys == []
        client.put_object.assert_not_called()

    def test_raises_when_local_dir_missing(self, tmp_path: Path) -> None:
        client = MagicMock()
        with pytest.raises(ValueError, match="does not exist"):
            upload_folder(client, tmp_path / "nope", "bkt", "p")


class TestDownloadPrefixToFolder:
    def _make_paginator(self, keys: list[str]) -> MagicMock:
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
        return paginator

    def test_mirrors_prefix_into_local_dir(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get_paginator.return_value = self._make_paginator(
            ["wf/out/a.txt", "wf/out/sub/b.txt"],
        )

        def fake_download(*, Bucket: str, Key: str, Fileobj: Any) -> None:
            Fileobj.write(Key.encode())

        client.download_fileobj.side_effect = fake_download
        written = download_prefix_to_folder(client, "bkt", "wf/out", tmp_path)
        assert written == ["a.txt", "sub/b.txt"]
        assert (tmp_path / "a.txt").read_bytes() == b"wf/out/a.txt"
        assert (tmp_path / "sub" / "b.txt").read_bytes() == b"wf/out/sub/b.txt"

    def test_skips_pseudo_directory_keys(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get_paginator.return_value = self._make_paginator(
            ["wf/out/", "wf/out/a.txt"],
        )
        client.download_fileobj.side_effect = lambda **_: None
        written = download_prefix_to_folder(client, "bkt", "wf/out", tmp_path)
        assert written == ["a.txt"]

    def test_returns_empty_for_missing_prefix(self, tmp_path: Path) -> None:
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{}]
        client.get_paginator.return_value = paginator
        written = download_prefix_to_folder(client, "bkt", "wf/out", tmp_path)
        assert written == []
