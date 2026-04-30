"""Telemetry — in-process counters + OpenTelemetry tracing.

Two layers:

* ``Telemetry`` keeps the lightweight in-process counters that ``/metrics``
  renders (latency, split ratio, V1.5 share). Unchanged from the V1 wiring.
* ``setup_otel`` / ``get_tracer`` wire up OTel tracing when the
  ``observability`` optional extra is installed *and*
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set. Without either of those the
  module returns no-op shims so the call sites can use the same
  ``with tracer.start_as_current_span(...)`` form unconditionally.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


@dataclass
class Counters:
    total: int = 0
    light: int = 0
    heavy: int = 0
    v15: int = 0
    classifier_fallback: int = 0
    latency_samples_ms: list[float] = field(default_factory=list)


class Telemetry:
    def __init__(self) -> None:
        self._counters = Counters()
        self._lock = threading.Lock()

    def record(self, *, path: str, latency_ms: float, classifier_fell_back: bool) -> None:
        with self._lock:
            self._counters.total += 1
            if path == "light":
                self._counters.light += 1
            elif path == "v1.5":
                self._counters.v15 += 1
            else:
                self._counters.heavy += 1
            if classifier_fell_back:
                self._counters.classifier_fallback += 1
            # keep last 1024 samples
            self._counters.latency_samples_ms.append(latency_ms)
            if len(self._counters.latency_samples_ms) > 1024:
                self._counters.latency_samples_ms = self._counters.latency_samples_ms[-1024:]

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            c = self._counters
            samples = sorted(c.latency_samples_ms)
            p50 = samples[len(samples) // 2] if samples else 0.0
            v1_total = c.light + c.heavy
            split_ratio = (c.heavy / v1_total) if v1_total else 0.0
            return {
                "total": c.total,
                "light": c.light,
                "heavy": c.heavy,
                "v1.5": c.v15,
                "heavy_ratio": round(split_ratio, 4),
                "classifier_fallback": c.classifier_fallback,
                "p50_latency_ms": round(p50, 2),
                "ts": time.time(),
            }


class _NoopSpan:
    """No-op span — accepts every call, never raises."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        return None

    def add_event(self, name: str, **kwargs: Any) -> None:  # noqa: D401
        return None

    def record_exception(self, exc: BaseException) -> None:  # noqa: D401
        return None

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _NoopTracer:
    """Returned by ``get_tracer`` when OTel is unavailable or not configured."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


def setup_otel(service_name: str = "gemma4-router") -> "Tracer | None":
    """Initialize the OTel tracer provider when configured.

    Returns the tracer, or ``None`` when the ``observability`` extra is not
    installed *or* ``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset. Callers should
    fall back to ``get_tracer`` (which returns a no-op shim) instead of
    branching on the return value.

    Env-driven config:

    * ``OTEL_EXPORTER_OTLP_ENDPOINT`` — e.g. ``http://otel-collector:4317``.
      Without this, instrumentation is a no-op.
    * ``OTEL_SERVICE_NAME`` — defaults to ``service_name`` arg.
    * ``OTEL_TRACE_SAMPLER`` / ``OTEL_TRACE_SAMPLER_ARG`` — picked up by
      the SDK directly. Default sampler is parent-based, sample-all.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return None

    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def get_tracer(name: str = "gemma4-router"):
    """Return the configured tracer or a no-op shim.

    Safe to call before ``setup_otel`` runs — it just returns the OTel API's
    own no-op tracer in that case (or our ``_NoopTracer`` if the API isn't
    importable at all).
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()
