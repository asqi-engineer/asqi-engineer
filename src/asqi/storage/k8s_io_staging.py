"""S3 staging for the K8s execution path.

When ``RUN_BACKEND=k8s``, asqi-engineer uploads the workload's local input
files to S3 *before* the :class:`~asqi.backends.kubernetes_backend.KubernetesBackend`
sees the args. The backend's per-Job ConfigMap then carries
:class:`~asqi.schemas.InputRef` / :class:`~asqi.schemas.OutputDestination`
that the sidecar (AIP-2208) consumes to materialise ``/shared/input`` and
publish ``/shared/output`` back to S3.

This module is the wedge between
:func:`asqi.workflow._execute_container_job` and the K8s backend. It

1. parses ``__volumes`` from the ``--test-params`` / ``--generation-params``
   JSON payloads,
2. uploads ``__volumes["input"]`` to a workflow/item-scoped S3 prefix,
3. emits :class:`~asqi.schemas.InputRef` per uploaded file and a single
   :class:`~asqi.schemas.OutputDestination` for the output prefix,
4. rewrites the payload: adds ``__inputs`` / ``__output`` (consumed by
   :func:`asqi.backends.kubernetes_backend._extract_io_refs`), strips
   ``__volumes`` (legacy Docker key — AIP-2473 rejects it fail-closed),
   and rewrites the plain ``volumes`` key so the workload sees
   ``{"input": "/input", "output": "/output"}`` (the AIP-2473 hardcoded
   sidecar mount points).

Docker remains unchanged: this module is only invoked when the caller
already knows it's on the K8s path.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, cast

from asqi.schemas import InputRef, OutputDestination
from asqi.storage.s3 import (
    S3ClientConfig,
    download_prefix_to_folder,
    ensure_bucket_exists,
    upload_file,
    upload_folder,
)

logger = logging.getLogger(__name__)

# ── Param-flag contract ────────────────────────────────────────────────────────
# Mirror the constants on the K8s backend (intentional copy — we don't want
# asqi.storage to import asqi.backends, which would create an import cycle
# through asqi.backends.kubernetes_backend).
_PARAM_FLAGS = ("--test-params", "--generation-params")
_VOLUMES_KEY = "__volumes"
_VOLUMES_PLAIN_KEY = "volumes"
_INPUTS_KEY = "__inputs"
_OUTPUT_KEY = "__output"

# AIP-2473 Pod spec mounts the shared emptyDir at these workload paths.
_WORKLOAD_INPUT_MOUNT = "/input"
_WORKLOAD_OUTPUT_MOUNT = "/output"

# ── Env-var contract (reuse asqi-runner names — same process) ──────────────────
_ENV_BUCKET = "AIP_ASQI_RUNNER_S3_BUCKET"
_ENV_ENDPOINT = "AIP_ASQI_RUNNER_S3_ENDPOINT"
_ENV_REGION = "AIP_ASQI_RUNNER_S3_REGION"
_ENV_ADDRESSING_STYLE = "AIP_ASQI_RUNNER_S3_ADDRESSING_STYLE"
_ENV_ACCESS_KEY = "AIP_ASQI_RUNNER_S3_ACCESS_KEY"
_ENV_SECRET_KEY = "AIP_ASQI_RUNNER_S3_SECRET_KEY"  # noqa: S105 — env var name, not a secret
_ENV_KEY_PREFIX = "AIP_ASQI_RUNNER_S3_KEY_PREFIX"


@dataclass
class StagingResult:
    """Outcome of :func:`stage_k8s_io`.

    Attributes:
        command_args: Modified args list (caller passes this to the backend
            instead of the original ``command_args``).
        output_bucket: Bucket holding workload output, or ``None`` if the
            workload did not declare a ``__volumes["output"]``.
        output_prefix: Key prefix under ``output_bucket`` reserved for the
            sidecar to publish workload outputs.
        local_output_path: Original host-side path from ``__volumes["output"]``.
            :func:`fetch_k8s_outputs` mirrors ``output_prefix`` back to here
            after the Job completes so downstream code (e.g.
            :func:`_translate_container_output_paths`) sees results at the
            same place Docker leaves them.
        input_keys: S3 keys uploaded for inputs (for diagnostics / cleanup).
    """

    command_args: list[str]
    output_bucket: str | None = None
    output_prefix: str | None = None
    local_output_path: str | None = None
    input_keys: list[str] = field(default_factory=list)


def s3_config_from_env() -> tuple[S3ClientConfig, str, str]:
    """Build an :class:`S3ClientConfig` + bucket + key-prefix from env vars.

    Reads the existing ``AIP_ASQI_RUNNER_S3_*`` env vars (since asqi-engineer
    workflow steps execute in-process inside the asqi-runner Pod, the same
    env is available).

    Returns:
        A 3-tuple ``(config, bucket, key_prefix)`` where ``key_prefix`` may
        be empty.

    Raises:
        RuntimeError: If a required env var is missing.
    """
    missing = [name for name in (_ENV_BUCKET, _ENV_ENDPOINT, _ENV_REGION) if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "K8s S3 staging requires " + ", ".join(missing) + " — set these env vars in the asqi-runner deployment."
        )

    addressing = os.environ.get(_ENV_ADDRESSING_STYLE, "auto").strip().lower()
    if addressing not in ("auto", "path", "virtual"):
        raise RuntimeError(f"{_ENV_ADDRESSING_STYLE}={addressing!r} is invalid; expected 'auto', 'path', or 'virtual'.")

    config = S3ClientConfig(
        endpoint_url=os.environ[_ENV_ENDPOINT],
        region=os.environ[_ENV_REGION],
        addressing_style=cast(Any, addressing),
        access_key=os.environ.get(_ENV_ACCESS_KEY) or None,
        secret_key=os.environ.get(_ENV_SECRET_KEY) or None,
    )
    bucket = os.environ[_ENV_BUCKET]
    key_prefix = os.environ.get(_ENV_KEY_PREFIX, "").strip("/")
    return config, bucket, key_prefix


def _scoped_prefix(global_prefix: str, workflow_id: str, item_id: str, leaf: str) -> str:
    """Build ``<global_prefix>/<workflow_id>/<item_id>/<leaf>`` (skipping empty parts)."""
    parts = [p.strip("/") for p in (global_prefix, workflow_id, item_id, leaf) if p]
    return "/".join(parts)


def _rewrite_payload_volumes(payload: dict[str, Any]) -> None:
    """Rewrite the workload-facing ``volumes`` key to use AIP-2473 mount paths.

    Only ``input`` / ``output`` keys are rewritten; any other ``volumes.*``
    entries are preserved untouched. If ``volumes`` is absent or not a dict,
    do nothing — the workload simply won't see one.
    """
    vols_any = payload.get(_VOLUMES_PLAIN_KEY)
    if not isinstance(vols_any, dict):
        return
    vols = cast(dict[str, Any], vols_any)
    if "input" in vols:
        vols["input"] = _WORKLOAD_INPUT_MOUNT
    if "output" in vols:
        vols["output"] = _WORKLOAD_OUTPUT_MOUNT


def stage_k8s_io(
    *,
    workflow_id: str,
    item_id: str,
    command_args: list[str],
    s3_client: Any,
    bucket: str,
    key_prefix: str = "",
    ensure_bucket: bool = False,
    region: str = "",
) -> StagingResult:
    """Stage K8s inputs/outputs by rewriting ``command_args`` in place.

    Walks ``command_args`` for ``--test-params`` / ``--generation-params``
    JSON payloads. For each payload that carries ``__volumes``:

    * If ``__volumes["input"]`` points at an existing directory, every file
      under it is uploaded to
      ``s3://<bucket>/<key_prefix>/<workflow_id>/<item_id>/input/``, one
      :class:`~asqi.schemas.InputRef` is emitted per file. A single
      non-directory file is also supported.
    * If ``__volumes["output"]`` is set, a single
      :class:`~asqi.schemas.OutputDestination` is emitted pointing at
      ``s3://<bucket>/<key_prefix>/<workflow_id>/<item_id>/output/``. The
      sidecar publishes the workload's ``/output`` back to that prefix.
    * ``__inputs`` and ``__output`` are added to the payload;
      ``__volumes`` is stripped (AIP-2473 rejects it fail-closed); the
      plain ``volumes`` key is rewritten to the AIP-2473 mount paths.

    A workload without ``__volumes`` is left unchanged (the JSON arg is
    not even re-serialised). Args that aren't JSON are also skipped — the
    contract matches :func:`asqi.backends.kubernetes_backend._extract_io_refs`.

    Args:
        workflow_id: DBOS workflow id; used to scope the S3 prefix.
        item_id: Test or generation-job id; used to scope the S3 prefix.
        command_args: The args list as built by ``create_test_execution_plan``
            / ``create_data_generation_plan``. Not mutated; a new list is
            returned via :attr:`StagingResult.command_args`.
        s3_client: A boto3 S3 client (from :func:`asqi.storage.s3.make_s3_client`).
        bucket: Bucket to upload to / read from.
        key_prefix: Optional global S3 prefix beneath which the
            ``<workflow_id>/<item_id>/`` scope is created.
        ensure_bucket: If ``True``, ``head_bucket`` then ``create_bucket`` if
            missing. Leave ``False`` in production where the bucket is
            provisioned by Helm/Terraform.
        region: Required when ``ensure_bucket=True`` to set the
            ``LocationConstraint``.

    Returns:
        A :class:`StagingResult` whose ``command_args`` should be passed to
        the K8s backend in place of the original list. If no ``__volumes``
        is present anywhere, the result still contains a copy of the
        original args; ``output_bucket`` and ``local_output_path`` will be
        ``None`` (and :func:`fetch_k8s_outputs` becomes a no-op).
    """
    new_args = list(command_args)
    output_bucket: str | None = None
    output_prefix: str | None = None
    local_output_path: str | None = None
    all_input_keys: list[str] = []

    for i, arg in enumerate(new_args):
        if arg not in _PARAM_FLAGS or i + 1 >= len(new_args):
            continue
        raw = new_args[i + 1]
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload_dict = cast(dict[str, Any], payload)

        volumes = payload_dict.get(_VOLUMES_KEY)
        if not isinstance(volumes, dict):
            continue
        volumes_dict = cast(dict[str, Any], volumes)

        if ensure_bucket:
            ensure_bucket_exists(s3_client, bucket, region)

        # ── Uploads ────────────────────────────────────────────────────────────
        input_refs: list[InputRef] = []
        local_input = volumes_dict.get("input")
        if isinstance(local_input, str) and local_input:
            input_prefix = _scoped_prefix(key_prefix, workflow_id, item_id, "input")
            uploaded_keys = _upload_input_path(s3_client, local_input, bucket, input_prefix)
            input_refs = [InputRef(bucket=bucket, key=k) for k in uploaded_keys]
            all_input_keys.extend(uploaded_keys)

        output_dest: OutputDestination | None = None
        local_output = volumes_dict.get("output")
        if isinstance(local_output, str) and local_output:
            out_prefix = _scoped_prefix(key_prefix, workflow_id, item_id, "output")
            output_dest = OutputDestination(bucket=bucket, key_prefix=out_prefix)
            output_bucket = bucket
            output_prefix = out_prefix
            local_output_path = local_output

        # ── Payload rewrite ────────────────────────────────────────────────────
        payload_dict.pop(_VOLUMES_KEY, None)
        if input_refs:
            payload_dict[_INPUTS_KEY] = [ref.model_dump(mode="json") for ref in input_refs]
        if output_dest is not None:
            payload_dict[_OUTPUT_KEY] = output_dest.model_dump(mode="json")
        _rewrite_payload_volumes(payload_dict)

        new_args[i + 1] = json.dumps(payload_dict)
        logger.debug(
            "Staged K8s I/O for %s/%s: inputs=%d, output_prefix=%s",
            workflow_id,
            item_id,
            len(input_refs),
            output_prefix,
        )

    return StagingResult(
        command_args=new_args,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        local_output_path=local_output_path,
        input_keys=all_input_keys,
    )


def _upload_input_path(s3_client: Any, local: str, bucket: str, key_prefix: str) -> list[str]:
    """Upload ``local`` (file or directory) and return the uploaded S3 keys."""
    if os.path.isdir(local):
        return upload_folder(s3_client, local, bucket, key_prefix)
    if os.path.isfile(local):
        prefix = key_prefix.strip("/")
        name = os.path.basename(local)
        key = f"{prefix}/{name}" if prefix else name
        upload_file(s3_client, local, bucket, key)
        return [key]
    raise FileNotFoundError(f"__volumes['input'] does not exist locally: {local}")


def fetch_k8s_outputs(staging: StagingResult, s3_client: Any) -> list[str]:
    """Mirror ``staging.output_prefix`` from S3 back to ``staging.local_output_path``.

    No-op when ``staging`` did not declare an output destination (e.g.
    workload without ``__volumes["output"]``). Returns the list of relative
    paths written (for logging / diagnostics).
    """
    if not staging.output_bucket or not staging.output_prefix or not staging.local_output_path:
        return []
    written = download_prefix_to_folder(
        s3_client,
        staging.output_bucket,
        staging.output_prefix,
        staging.local_output_path,
    )
    logger.info(
        "Fetched %d K8s output file(s) from s3://%s/%s/ -> %s",
        len(written),
        staging.output_bucket,
        staging.output_prefix,
        staging.local_output_path,
    )
    return written
