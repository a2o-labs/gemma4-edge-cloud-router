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


_LATENCY_PER_PATH_CAP = 1000
_PROM_BUCKETS_S = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass
class Counters:
    total: int = 0
    light: int = 0
    heavy: int = 0
    v15: int = 0
    classifier_fallback: int = 0
    latency_samples_ms: list[float] = field(default_factory=list)
    latency_per_path_ms: dict[str, list[float]] = field(default_factory=dict)


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
            per_path = self._counters.latency_per_path_ms.setdefault(path, [])
            per_path.append(latency_ms)
            if len(per_path) > _LATENCY_PER_PATH_CAP:
                del per_path[: len(per_path) - _LATENCY_PER_PATH_CAP]

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

    def format_prometheus(self) -> str:
        """Emit Prometheus exposition format (text/plain; version=0.0.4)."""
        with self._lock:
            c = self._counters
            per_path = {p: list(v) for p, v in c.latency_per_path_ms.items()}
            light = c.light
            heavy = c.heavy
            v15 = c.v15
            classifier_fallback = c.classifier_fallback

        lines = [
            "# HELP gemma4_router_requests_total Total requests by path",
            "# TYPE gemma4_router_requests_total counter",
            f'gemma4_router_requests_total{{path="light"}} {light}',
            f'gemma4_router_requests_total{{path="heavy"}} {heavy}',
            f'gemma4_router_requests_total{{path="v1.5"}} {v15}',
            "",
            "# HELP gemma4_router_classifier_fallback_total Classifier fallbacks",
            "# TYPE gemma4_router_classifier_fallback_total counter",
            f"gemma4_router_classifier_fallback_total {classifier_fallback}",
            "",
            "# HELP gemma4_router_latency_seconds Request latency histogram",
            "# TYPE gemma4_router_latency_seconds histogram",
        ]
        for path in ("light", "heavy", "v1.5"):
            latencies = per_path.get(path, [])
            if not latencies:
                continue
            seconds = [ms / 1000.0 for ms in latencies]
            cumulative = 0
            sorted_s = sorted(seconds)
            i = 0
            for b in _PROM_BUCKETS_S:
                while i < len(sorted_s) and sorted_s[i] <= b:
                    i += 1
                cumulative = i
                lines.append(
                    f'gemma4_router_latency_seconds_bucket{{path="{path}",le="{b}"}} {cumulative}'
                )
            lines.append(
                f'gemma4_router_latency_seconds_bucket{{path="{path}",le="+Inf"}} {len(seconds)}'
            )
            lines.append(
                f'gemma4_router_latency_seconds_sum{{path="{path}"}} {sum(seconds)}'
            )
            lines.append(
                f'gemma4_router_latency_seconds_count{{path="{path}"}} {len(seconds)}'
            )
            lines.append("")
        return "\n".join(lines) + "\n"


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
