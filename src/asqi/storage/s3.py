"""Pure S3 primitives for both asqi-engineer and asqi-runner.

This module intentionally has no dependency on ``core.config`` (runner
settings), ``core.logging`` (runner structlog), or FastAPI. Callers
inject S3 endpoint / credentials / region via :class:`S3ClientConfig`
and pass the resulting client around.

Runner-side wrappers that read ``settings`` and add ``UploadFile``
support live in ``services/asqi-runner/storage/s3.py`` and delegate
the actual S3 calls back to this module.
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

AddressingStyle = Literal["auto", "path", "virtual"]


@dataclass(frozen=True)
class S3ClientConfig:
    """Caller-injected S3 client configuration.

    Frozen so it can be used as a cache key for :func:`make_s3_client`.

    Attributes:
        endpoint_url: S3 endpoint (e.g. AWS regional endpoint or a MinIO URL).
        region: AWS region name (e.g. ``"ap-southeast-1"``). Some MinIO
            deployments accept any non-empty string here.
        addressing_style: ``"auto"`` (default), ``"path"`` (required for
            most MinIO deployments), or ``"virtual"``.
        access_key: Optional static access key. When both ``access_key``
            and ``secret_key`` are ``None``, boto3's default credential
            chain is used (IRSA, instance profile, env vars, etc.).
        secret_key: Optional static secret key.
    """

    endpoint_url: str
    region: str
    addressing_style: AddressingStyle = "auto"
    access_key: str | None = None
    secret_key: str | None = None


@lru_cache(maxsize=4)
def make_s3_client(config: S3ClientConfig) -> Any:
    """Build (and cache) a boto3 S3 client for ``config``.

    The cache key is the full :class:`S3ClientConfig` instance, so callers
    that rotate credentials must construct a new config. The cache is
    bounded at 4 entries to avoid unbounded growth in test fixtures.
    """
    kwargs: dict[str, Any] = {
        "region_name": config.region,
        "endpoint_url": config.endpoint_url,
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": config.addressing_style},
        ),
    }
    if config.access_key and config.secret_key:
        kwargs["aws_access_key_id"] = config.access_key
        kwargs["aws_secret_access_key"] = config.secret_key

    boto3_any = cast(Any, boto3)
    return boto3_any.client("s3", **kwargs)


def ensure_bucket_exists(s3_client: Any, bucket: str, region: str) -> None:
    """Create *bucket* if it does not already exist.

    A successful ``head_bucket`` short-circuits without creating anything.
    A 404 / missing bucket response triggers creation. Other boto3
    ``ClientError`` responses, such as 403, are re-raised.
    """
    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            logger.warning("Bucket '%s' head_bucket failed with %s", bucket, error_code)
            raise
        logger.debug("Bucket '%s' missing via head_bucket (%s); will attempt create", bucket, error_code)
    except Exception as e:
        logger.debug(
            "Bucket '%s' not reachable via head_bucket (%s); will attempt create",
            bucket,
            type(e).__name__,
        )

    body: dict[str, Any] = {"Bucket": bucket}
    if region and region != "us-east-1":
        body["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3_client.create_bucket(**body)
    logger.info("Created S3 bucket '%s' in region '%s'", bucket, region or "<default>")


def upload_file(
    s3_client: Any,
    local_path: str | Path,
    bucket: str,
    key: str,
    content_type: str | None = None,
) -> None:
    """Upload a local file to ``s3://<bucket>/<key>``.

    Raises:
        FileNotFoundError: If ``local_path`` does not exist.
        Exception: Any boto3 ``ClientError`` raised by ``put_object`` is
            propagated; callers decide whether to retry or fail-closed.
    """
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    if content_type is None:
        guessed, _ = mimetypes.guess_type(str(path))
        content_type = guessed or "application/octet-stream"

    with path.open("rb") as fh:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=fh,
            ContentType=content_type,
        )
    logger.debug("Uploaded '%s' -> s3://%s/%s", path, bucket, key)


def download_file_to_path(
    s3_client: Any,
    bucket: str,
    key: str,
    local_path: str | Path,
) -> None:
    """Download ``s3://<bucket>/<key>`` to ``local_path``.

    Parent directories are created if missing. Raises on any boto3 error.
    """
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=fh)
    logger.debug("Downloaded s3://%s/%s -> '%s'", bucket, key, path)


def upload_folder(
    s3_client: Any,
    local_dir: str | Path,
    bucket: str,
    key_prefix: str,
) -> list[str]:
    """Recursively upload every file under ``local_dir`` to ``key_prefix``.

    Returns the list of S3 keys written, in the same order they were
    uploaded. Relative paths under ``local_dir`` map directly onto S3
    keys under ``key_prefix`` (POSIX separators).

    Raises:
        ValueError: If ``local_dir`` does not exist or is not a directory.
    """
    root = Path(local_dir)
    if not root.is_dir():
        raise ValueError(f"local_dir does not exist or is not a directory: {local_dir}")

    prefix = key_prefix.strip("/")
    written: list[str] = []

    for local_path in sorted(root.rglob("*")):
        if not local_path.is_file():
            continue
        rel = local_path.relative_to(root).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        upload_file(s3_client, local_path, bucket, key)
        written.append(key)

    logger.debug("Uploaded %d file(s) from '%s' to s3://%s/%s/", len(written), root, bucket, prefix)
    return written


def download_prefix_to_folder(
    s3_client: Any,
    bucket: str,
    key_prefix: str,
    local_dir: str | Path,
) -> list[str]:
    """Mirror every object under ``s3://<bucket>/<key_prefix>/`` into ``local_dir``.

    Uses the ``list_objects_v2`` paginator so it handles prefixes with
    thousands of objects. Returns the list of relative paths written
    (relative to ``local_dir``), in the order they were downloaded.
    """
    prefix = key_prefix.strip("/")
    paginated_prefix = f"{prefix}/" if prefix else ""

    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()
    written: list[str] = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=paginated_prefix):
        for obj in page.get("Contents", []) or []:
            key = cast(str, obj["Key"])
            # Strip the prefix to recover the relative path under local_dir.
            rel = key[len(paginated_prefix) :] if paginated_prefix and key.startswith(paginated_prefix) else key
            if not rel or rel.endswith("/"):
                continue
            # Defence in depth: refuse keys that would escape local_dir.
            target = (root / rel).resolve()
            if not target.is_relative_to(root_resolved):
                raise ValueError(f"Refusing to write outside local_dir; suspicious S3 key: {key!r}")
            download_file_to_path(s3_client, bucket, key, target)
            written.append(rel)

    logger.debug("Downloaded %d file(s) from s3://%s/%s -> '%s'", len(written), bucket, paginated_prefix, root)
    return written
