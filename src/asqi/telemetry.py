"""Shared OpenTelemetry SDK bootstrap for asqi-engineer runtimes (AIP-2890).

Both the CLI entrypoint (``asqi.main``) and the FastAPI service
(``services/asqi-runner/app.py``) instrument code that calls
``opentelemetry.metrics.get_meter()`` / ``opentelemetry.trace.get_tracer()``
(see ``asqi.workflow``). The bare ``opentelemetry-api`` package is a no-op
until a real SDK ``MeterProvider``/``TracerProvider`` has been installed via
``metrics.set_meter_provider()`` / ``trace.set_tracer_provider()`` — this
module is the one place that does that, so both runtimes share a single,
proven-correct implementation instead of two copies that can drift.

Enabled by setting ``OTEL_EXPORTER_OTLP_ENDPOINT``; a no-op otherwise.

Resource attributes (``service.name``, ``deployment.environment.name``,
etc.) are picked up automatically by the SDK from ``OTEL_SERVICE_NAME`` /
``OTEL_RESOURCE_ATTRIBUTES`` per the OpenTelemetry spec when a
``MeterProvider``/``TracerProvider`` is constructed with no explicit
``resource=`` argument — this module intentionally does not duplicate that
parsing.

The OTLP HTTP exporters likewise read ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
append the standard ``/v1/traces`` and ``/v1/metrics`` signal paths
themselves; this module does not construct those URLs manually.
"""

from __future__ import annotations

import os
import threading

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_lock = threading.Lock()
_initialized = False
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def is_enabled() -> bool:
    """Whether OTLP export is configured for this process."""
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))


def bootstrap() -> None:
    """Install real SDK ``MeterProvider``/``TracerProvider``, once per process.

    No-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset. Idempotent: safe to
    call from multiple call sites (e.g. a runtime's own startup path plus
    this shared module) without installing a second provider or emitting
    duplicate exporters.
    """
    global _initialized, _tracer_provider, _meter_provider

    if not is_enabled():
        return

    with _lock:
        if _initialized:
            return

        _meter_provider = MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())]
        )
        metrics.set_meter_provider(_meter_provider)

        _tracer_provider = TracerProvider()
        _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(_tracer_provider)

        _initialized = True


def shutdown() -> None:
    """Flush and shut down the providers installed by :func:`bootstrap`.

    Safe to call even if :func:`bootstrap` was never called or was a no-op.
    """
    global _initialized

    with _lock:
        if not _initialized:
            return

        if _tracer_provider is not None:
            _tracer_provider.shutdown()
        if _meter_provider is not None:
            _meter_provider.shutdown()

        _initialized = False
