"""OTel tracing — verify spans emitted by V15Pipeline.run + setup helpers.

Wires up an in-memory span exporter (no network, no real OTLP collector)
and asserts the manual spans emitted by ``V15Pipeline.run`` and the
``api`` route handlers can be observed with the expected attributes.

Skips if ``opentelemetry-sdk`` is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")


def _setup_in_memory_tracing():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__), exporter


def test_v15_pipeline_emits_encode_and_forward_spans(monkeypatch):
    monkeypatch.setenv("V15_ENABLED", "true")
    monkeypatch.setenv("V15_MOCK_MODE", "true")
    monkeypatch.setenv("V15_DEVICE", "cpu")

    import router.config

    router.config._settings = None

    _, exporter = _setup_in_memory_tracing()

    from router.config import get_settings
    from router.v15_pipeline import V15Pipeline

    pipeline = V15Pipeline(get_settings().v15)
    pipeline.load()
    schema, answer = pipeline.run("Hello world")

    spans = exporter.get_finished_spans()
    span_names = {s.name for s in spans}
    assert "v15.encode" in span_names, f"got: {span_names}"
    assert "v15.forward" in span_names, f"got: {span_names}"

    encode_span = next(s for s in spans if s.name == "v15.encode")
    assert encode_span.attributes.get("complexity") in {"light", "heavy"}
    assert encode_span.attributes.get("embedding_dim") == schema.embedding_dim

    forward_span = next(s for s in spans if s.name == "v15.forward")
    assert forward_span.attributes.get("response_chars") == len(answer)
    assert forward_span.attributes.get("max_new_tokens") is not None

    router.config._settings = None


def test_setup_otel_returns_none_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from router.telemetry import setup_otel

    assert setup_otel() is None


def test_get_tracer_returns_usable_tracer_without_provider(monkeypatch):
    """Even without OTel configured, ``get_tracer`` returns something whose
    ``start_as_current_span`` is a usable context manager. Calling
    ``set_attribute`` on the resulting span must not raise.
    """
    from router.telemetry import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("test") as span:
        span.set_attribute("k", "v")
