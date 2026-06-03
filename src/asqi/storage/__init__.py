"""Storage primitives shared between asqi-engineer and asqi-runner.

The :mod:`asqi.storage.s3` submodule exposes a small, pure S3 client surface
(config dataclass + boto3 wrappers) with no dependency on runner-side config,
logging, or FastAPI. Both packages import from here; runner-side wrappers
that read ``settings`` from ``core.config`` and add FastAPI ``UploadFile``
support remain in ``services/asqi-runner/storage/s3.py``.
"""
