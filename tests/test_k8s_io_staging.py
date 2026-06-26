"""Unit tests for :mod:`asqi.storage.k8s_io_staging`.

S3 client is mocked at the boto3 level; no MinIO container required.
The K8s backend is not invoked — these tests verify the args-rewrite
contract that the backend later consumes via ``_extract_io_refs``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from asqi.storage.k8s_io_staging import (
    StagingResult,
    fetch_k8s_outputs,
    s3_config_from_env,
    stage_k8s_io,
)


def _params(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _build_args(
    test_payload: dict[str, Any] | None = None,
    gen_payload: dict[str, Any] | None = None,
) -> list[str]:
    args: list[str] = ["run"]
    if test_payload is not None:
        args.extend(["--test-params", _params(test_payload)])
    if gen_payload is not None:
        args.extend(["--generation-params", _params(gen_payload)])
    return args


class TestStageK8sIoNoVolumes:
    def test_returns_args_unchanged_when_no_volumes(self, tmp_path: Path) -> None:
        args = _build_args({"params": {"k": "v"}})
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        assert result.command_args == args
        assert result.output_bucket is None
        assert result.output_prefix is None
        assert result.local_output_path is None
        assert result.input_keys == []
        client.put_object.assert_not_called()

    def test_skips_non_json_arg_values(self, tmp_path: Path) -> None:
        args = ["run", "--test-params", "not-json"]
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        assert result.command_args == args


class TestStageK8sIoUploads:
    def test_uploads_input_directory_and_emits_inputrefs(self, tmp_path: Path) -> None:
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.txt").write_text("a")
        (tmp_path / "in" / "sub").mkdir()
        (tmp_path / "in" / "sub" / "b.txt").write_text("b")
        (tmp_path / "out").mkdir()

        args = _build_args(
            {
                "__volumes": {
                    "input": str(tmp_path / "in"),
                    "output": str(tmp_path / "out"),
                },
                "volumes": {"input": "/local/in", "output": "/local/out"},
                "params": {"k": "v"},
            }
        )
        client = MagicMock()

        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )

        # put_object called once per input file
        assert client.put_object.call_count == 2
        # Two inputs uploaded under the expected prefix
        assert sorted(result.input_keys) == [
            "wf1/it1/input/a.txt",
            "wf1/it1/input/sub/b.txt",
        ]
        # Output destination captured for post-run fetch
        assert result.output_bucket == "aip-test"
        assert result.output_prefix == "wf1/it1/output"
        assert result.local_output_path == str(tmp_path / "out")

    def test_uploads_single_input_file(self, tmp_path: Path) -> None:
        in_file = tmp_path / "data.json"
        in_file.write_text("{}")
        args = _build_args({"__volumes": {"input": str(in_file)}})
        client = MagicMock()

        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        assert result.input_keys == ["wf1/it1/input/data.json"]

    def test_missing_input_path_raises(self, tmp_path: Path) -> None:
        args = _build_args({"__volumes": {"input": str(tmp_path / "missing")}})
        client = MagicMock()
        with pytest.raises(FileNotFoundError):
            stage_k8s_io(
                workflow_id="wf1",
                item_id="it1",
                command_args=args,
                s3_client=client,
                bucket="aip-test",
            )

    def test_global_key_prefix_is_applied(self, tmp_path: Path) -> None:
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.txt").write_text("a")
        args = _build_args(
            {
                "__volumes": {
                    "input": str(tmp_path / "in"),
                    "output": str(tmp_path / "out"),
                }
            }
        )
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
            key_prefix="customer-a",
        )
        assert result.input_keys == ["customer-a/wf1/it1/input/a.txt"]
        assert result.output_prefix == "customer-a/wf1/it1/output"


class TestStageK8sIoPayloadRewrite:
    def _staged_payload(self, args: list[str], flag: str) -> dict[str, Any]:
        idx = args.index(flag)
        return json.loads(args[idx + 1])

    def test_strips_volumes_reserved_key(self, tmp_path: Path) -> None:
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.txt").write_text("a")
        args = _build_args(
            {"__volumes": {"input": str(tmp_path / "in")}, "params": {"k": "v"}}
        )
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        payload = self._staged_payload(result.command_args, "--test-params")
        assert "__volumes" not in payload
        assert payload["params"] == {"k": "v"}

    def test_adds_inputs_and_output_reserved_keys(self, tmp_path: Path) -> None:
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.txt").write_text("a")
        args = _build_args(
            {
                "__volumes": {
                    "input": str(tmp_path / "in"),
                    "output": str(tmp_path / "out"),
                },
            }
        )
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        payload = self._staged_payload(result.command_args, "--test-params")
        assert payload["__inputs"] == [
            {"bucket": "aip-test", "key": "wf1/it1/input/a.txt", "checksum": None}
        ]
        assert payload["__output"] == {
            "bucket": "aip-test",
            "key_prefix": "wf1/it1/output",
        }

    def test_rewrites_plain_volumes_to_container_paths(self, tmp_path: Path) -> None:
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "a.txt").write_text("a")
        args = _build_args(
            {
                "__volumes": {
                    "input": str(tmp_path / "in"),
                    "output": str(tmp_path / "out"),
                },
                "volumes": {
                    "input": "/host/in",
                    "output": "/host/out",
                    "cache": "/host/cache",
                },
            }
        )
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        payload = self._staged_payload(result.command_args, "--test-params")
        assert payload["volumes"]["input"] == "/input"
        assert payload["volumes"]["output"] == "/output"
        # Custom volume keys are preserved untouched.
        assert payload["volumes"]["cache"] == "/host/cache"

    def test_omits_inputs_key_when_no_input_path(self, tmp_path: Path) -> None:
        # Only output present, no input.
        args = _build_args({"__volumes": {"output": str(tmp_path / "out")}})
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        payload = self._staged_payload(result.command_args, "--test-params")
        assert "__inputs" not in payload
        assert payload["__output"]["key_prefix"] == "wf1/it1/output"

    def test_handles_both_test_and_generation_params(self, tmp_path: Path) -> None:
        (tmp_path / "in_t").mkdir()
        (tmp_path / "in_t" / "t.txt").write_text("t")
        (tmp_path / "in_g").mkdir()
        (tmp_path / "in_g" / "g.txt").write_text("g")
        args = _build_args(
            test_payload={
                "__volumes": {
                    "input": str(tmp_path / "in_t"),
                    "output": str(tmp_path / "out_t"),
                }
            },
            gen_payload={
                "__volumes": {
                    "input": str(tmp_path / "in_g"),
                    "output": str(tmp_path / "out_g"),
                }
            },
        )
        client = MagicMock()
        result = stage_k8s_io(
            workflow_id="wf1",
            item_id="it1",
            command_args=args,
            s3_client=client,
            bucket="aip-test",
        )
        # Both payloads should have been rewritten; second pass overwrites the
        # output destination (matches the single-output K8s sidecar contract).
        test_payload = json.loads(
            result.command_args[result.command_args.index("--test-params") + 1]
        )
        gen_payload = json.loads(
            result.command_args[result.command_args.index("--generation-params") + 1]
        )
        assert "__inputs" in test_payload
        assert "__inputs" in gen_payload
        # Final tracked output prefix is the last one encountered (gen).
        assert result.output_prefix == "wf1/it1/output"


class TestFetchK8sOutputs:
    def test_noop_when_no_output(self) -> None:
        staging = StagingResult(command_args=["run"])
        client = MagicMock()
        assert fetch_k8s_outputs(staging, client) == []
        client.get_paginator.assert_not_called()

    def test_downloads_when_output_present(self, tmp_path: Path) -> None:
        staging = StagingResult(
            command_args=["run"],
            output_bucket="aip-test",
            output_prefix="wf1/it1/output",
            local_output_path=str(tmp_path / "out"),
        )
        client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "wf1/it1/output/result.json"}]}
        ]
        client.get_paginator.return_value = paginator

        def fake_download(*, Bucket: str, Key: str, Fileobj: Any) -> None:
            Fileobj.write(b"{}")

        client.download_fileobj.side_effect = fake_download
        written = fetch_k8s_outputs(staging, client)
        assert written == ["result.json"]
        assert (tmp_path / "out" / "result.json").read_bytes() == b"{}"


class TestS3ConfigFromEnv:
    def test_raises_when_required_envvars_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in (
            "AIP_ASQI_RUNNER_S3_BUCKET",
            "AIP_ASQI_RUNNER_S3_ENDPOINT",
            "AIP_ASQI_RUNNER_S3_REGION",
        ):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(RuntimeError, match="AIP_ASQI_RUNNER_S3_BUCKET"):
            s3_config_from_env()

    def test_builds_config_from_full_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_BUCKET", "b")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_ENDPOINT", "http://minio:9000")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_REGION", "ap-southeast-1")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_ADDRESSING_STYLE", "path")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_ACCESS_KEY", "ak")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_SECRET_KEY", "sk")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_KEY_PREFIX", "/customer/x/")
        cfg, bucket, prefix = s3_config_from_env()
        assert cfg.endpoint_url == "http://minio:9000"
        assert cfg.region == "ap-southeast-1"
        assert cfg.addressing_style == "path"
        assert cfg.access_key == "ak"
        assert cfg.secret_key == "sk"  # noqa: S105 — test fixture, not a secret
        assert bucket == "b"
        assert prefix == "customer/x"

    def test_rejects_invalid_addressing_style(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_BUCKET", "b")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_ENDPOINT", "x")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_REGION", "r")
        monkeypatch.setenv("AIP_ASQI_RUNNER_S3_ADDRESSING_STYLE", "weird")
        with pytest.raises(RuntimeError, match="AIP_ASQI_RUNNER_S3_ADDRESSING_STYLE"):
            s3_config_from_env()
