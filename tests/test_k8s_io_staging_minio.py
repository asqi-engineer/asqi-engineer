"""Integration tests for K8s S3 I/O staging against local MinIO.

These tests exercise AIP-2474 ACs 1 & 2 with a real S3 client:

* AC1 — input upload before Job creation: :func:`stage_k8s_io` uploads files
  found under ``__volumes["input"]`` to MinIO at a workflow/item-scoped prefix;
  test asserts the objects exist via a direct ``list_objects_v2`` call.
* AC2 — output download after Job completion: a fake "sidecar output" object
  is placed under the staging :class:`OutputDestination` prefix, then
  :func:`fetch_k8s_outputs` mirrors it back to the local output directory;
  test asserts the file appears at the expected local path.

The tests auto-skip when MinIO is unreachable so CI without infra passes.
To run locally:

    docker compose -f deploy/compose/infra/minio/compose.yaml up -d
    AIP_ASQI_RUNNER_S3_ENDPOINT=http://localhost:9000 \\
    AIP_ASQI_RUNNER_S3_ACCESS_KEY=miniouser \\
    AIP_ASQI_RUNNER_S3_SECRET_KEY=miniopassword \\
    AIP_ASQI_RUNNER_S3_REGION=us-east-1 \\
    AIP_ASQI_RUNNER_S3_BUCKET=aip-test-aip2474 \\
    AIP_ASQI_RUNNER_S3_ADDRESSING_STYLE=path \\
    uv run --extra k8s --group test pytest tests/test_k8s_io_staging_minio.py -q
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

boto3 = pytest.importorskip("boto3")
botocore = pytest.importorskip("botocore")
from asqi.storage.k8s_io_staging import (  # noqa: E402
    fetch_k8s_outputs,
    stage_k8s_io,
)
from asqi.storage.s3 import (  # noqa: E402
    S3ClientConfig,
    make_s3_client,
)
from botocore.exceptions import (  # noqa: E402  — after importorskip
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
)

# ── MinIO endpoint discovery ──────────────────────────────────────────────────
_ENDPOINT = os.environ.get("AIP_ASQI_RUNNER_S3_ENDPOINT", "http://localhost:9000")
_REGION = os.environ.get("AIP_ASQI_RUNNER_S3_REGION", "us-east-1")
_ACCESS_KEY = os.environ.get("AIP_ASQI_RUNNER_S3_ACCESS_KEY", "miniouser")
_SECRET_KEY = os.environ.get("AIP_ASQI_RUNNER_S3_SECRET_KEY", "miniopassword")
_BUCKET = os.environ.get("AIP_ASQI_RUNNER_S3_BUCKET", "aip-test-aip2474")


def _minio_probe() -> str | None:
    """Return None if MinIO is reachable + can list buckets; else a skip reason."""
    try:
        client = make_s3_client(
            S3ClientConfig(
                endpoint_url=_ENDPOINT,
                region=_REGION,
                addressing_style="path",
                access_key=_ACCESS_KEY,
                secret_key=_SECRET_KEY,
            )
        )
        client.list_buckets()
    except (EndpointConnectionError, BotoCoreError, ClientError, OSError) as exc:
        return f"MinIO probe failed at {_ENDPOINT}: {type(exc).__name__}: {exc}"
    return None


_SKIP_REASON = _minio_probe()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def s3_client() -> Any:
    return make_s3_client(
        S3ClientConfig(
            endpoint_url=_ENDPOINT,
            region=_REGION,
            addressing_style="path",
            access_key=_ACCESS_KEY,
            secret_key=_SECRET_KEY,
        )
    )


@pytest.fixture
def ensured_bucket(s3_client: Any) -> Iterator[str]:
    """Ensure the test bucket exists. Doesn't tear it down (shared MinIO)."""
    try:
        s3_client.head_bucket(Bucket=_BUCKET)
    except ClientError:
        s3_client.create_bucket(Bucket=_BUCKET)
    yield _BUCKET


@pytest.fixture
def isolated_prefix() -> str:
    """A unique per-test S3 key prefix so concurrent runs don't collide."""
    return f"aip-2474-it/{uuid.uuid4().hex[:12]}"


def _list_keys(s3_client: Any, bucket: str, prefix: str) -> list[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(cast(str, obj["Key"]))
    return sorted(keys)


# ── AC1: input upload before Job creation ─────────────────────────────────────
class TestAC1InputUpload:
    def test_stage_uploads_input_directory_to_minio(
        self,
        tmp_path: Path,
        s3_client: Any,
        ensured_bucket: str,
        isolated_prefix: str,
    ) -> None:
        # Arrange: a local input dir like the workload would see
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello-a")
        (input_dir / "nested").mkdir()
        (input_dir / "nested" / "b.json").write_text('{"x": 1}')

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        payload = {
            "__volumes": {"input": str(input_dir), "output": str(output_dir)},
            "volumes": {"input": str(input_dir), "output": str(output_dir)},
            "other": "preserved",
        }
        args = ["--test-params", json.dumps(payload)]

        # Act
        staging = stage_k8s_io(
            workflow_id="wf-it",
            item_id="item-it",
            command_args=args,
            s3_client=s3_client,
            bucket=ensured_bucket,
            key_prefix=isolated_prefix,
        )

        # Assert: objects exist under input prefix
        expected_prefix = f"{isolated_prefix}/wf-it/item-it/input"
        uploaded = _list_keys(s3_client, ensured_bucket, expected_prefix)
        assert uploaded, f"no objects uploaded under {expected_prefix}"
        rel_keys = {k[len(expected_prefix) + 1 :] for k in uploaded}
        assert rel_keys == {"a.txt", "nested/b.json"}

        # Assert: rewritten args carry __inputs + __output, no __volumes, and
        # plain volumes is rewritten to /input + /output
        rewritten = json.loads(staging.command_args[1])
        assert "__volumes" not in rewritten
        assert rewritten["other"] == "preserved"
        assert rewritten["volumes"] == {"input": "/input", "output": "/output"}
        assert len(rewritten["__inputs"]) == 2
        for ref in rewritten["__inputs"]:
            assert ref["bucket"] == ensured_bucket
            assert ref["key"].startswith(expected_prefix + "/")
        assert rewritten["__output"]["bucket"] == ensured_bucket
        assert (
            rewritten["__output"]["key_prefix"]
            == f"{isolated_prefix}/wf-it/item-it/output"
        )

        # Cleanup: best-effort
        for key in uploaded:
            s3_client.delete_object(Bucket=ensured_bucket, Key=key)

    def test_stage_uploads_single_input_file(
        self,
        tmp_path: Path,
        s3_client: Any,
        ensured_bucket: str,
        isolated_prefix: str,
    ) -> None:
        input_file = tmp_path / "single.csv"
        input_file.write_text("col\nval\n")
        payload = {"__volumes": {"input": str(input_file)}}
        args = ["--generation-params", json.dumps(payload)]

        stage_k8s_io(
            workflow_id="wf",
            item_id="it",
            command_args=args,
            s3_client=s3_client,
            bucket=ensured_bucket,
            key_prefix=isolated_prefix,
        )

        expected_prefix = f"{isolated_prefix}/wf/it/input"
        uploaded = _list_keys(s3_client, ensured_bucket, expected_prefix)
        assert uploaded == [f"{expected_prefix}/single.csv"]
        s3_client.delete_object(Bucket=ensured_bucket, Key=uploaded[0])


# ── AC2: output download after Job completion ─────────────────────────────────
class TestAC2OutputDownload:
    def test_fetch_mirrors_output_prefix_to_local_dir(
        self,
        tmp_path: Path,
        s3_client: Any,
        ensured_bucket: str,
        isolated_prefix: str,
    ) -> None:
        # Arrange: run staging to produce a StagingResult with output_prefix set
        local_output = tmp_path / "out"
        local_output.mkdir()
        payload = {"__volumes": {"output": str(local_output)}}
        args = ["--test-params", json.dumps(payload)]
        staging = stage_k8s_io(
            workflow_id="wf",
            item_id="it",
            command_args=args,
            s3_client=s3_client,
            bucket=ensured_bucket,
            key_prefix=isolated_prefix,
        )
        assert staging.output_prefix == f"{isolated_prefix}/wf/it/output"

        # Simulate the sidecar publishing two output objects.
        out_prefix = cast(str, staging.output_prefix)
        s3_client.put_object(
            Bucket=ensured_bucket,
            Key=f"{out_prefix}/result.json",
            Body=b'{"score": 0.42}',
        )
        s3_client.put_object(
            Bucket=ensured_bucket,
            Key=f"{out_prefix}/logs/run.log",
            Body=b"ok\n",
        )

        # Act
        written = fetch_k8s_outputs(staging, s3_client)

        # Assert: files mirrored locally with prefix-relative paths preserved
        assert sorted(written) == sorted(["result.json", "logs/run.log"])
        assert (local_output / "result.json").read_bytes() == b'{"score": 0.42}'
        assert (local_output / "logs" / "run.log").read_bytes() == b"ok\n"

        # Cleanup
        for key in [f"{out_prefix}/result.json", f"{out_prefix}/logs/run.log"]:
            s3_client.delete_object(Bucket=ensured_bucket, Key=key)

    def test_fetch_is_noop_when_no_output_declared(
        self,
        tmp_path: Path,
        s3_client: Any,
        ensured_bucket: str,
        isolated_prefix: str,
    ) -> None:
        payload = {"__volumes": {"input": str(tmp_path)}}
        (tmp_path / "x.txt").write_text("x")
        args = ["--test-params", json.dumps(payload)]
        staging = stage_k8s_io(
            workflow_id="wf",
            item_id="it",
            command_args=args,
            s3_client=s3_client,
            bucket=ensured_bucket,
            key_prefix=isolated_prefix,
        )
        assert fetch_k8s_outputs(staging, s3_client) == []

        # Cleanup uploaded input
        uploaded = _list_keys(
            s3_client, ensured_bucket, f"{isolated_prefix}/wf/it/input"
        )
        for key in uploaded:
            s3_client.delete_object(Bucket=ensured_bucket, Key=key)
