"""Prometheus exposition endpoint tests."""

from __future__ import annotations

import os
import re

os.environ["V15_ENABLED"] = "true"
os.environ["V15_MOCK_MODE"] = "true"
os.environ["V15_DEVICE"] = "cpu"

import pytest
from fastapi.testclient import TestClient

import router.config as config_module


@pytest.fixture
def client():
    config_module._settings = None
    from router.api import app

    with TestClient(app) as c:
        yield c


def test_prometheus_endpoint_exposes_metrics(client):
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "gemma4_router_requests_total" in body
    assert 'path="light"' in body
    assert 'path="heavy"' in body
    assert 'path="v1.5"' in body
    assert "# HELP" in body
    assert "# TYPE" in body


def test_prometheus_after_traffic_increments(client):
    client.post("/route/v15", json={"prompt": "Hi"})
    client.post("/route/v15", json={"prompt": "How does X work?"})
    r = client.get("/metrics/prometheus")
    body = r.text
    m = re.search(r'gemma4_router_requests_total\{path="v1\.5"\} (\d+)', body)
    assert m
    assert int(m.group(1)) >= 1
    # histogram lines should appear once we have v1.5 traffic
    assert 'gemma4_router_latency_seconds_bucket{path="v1.5"' in body
    assert 'gemma4_router_latency_seconds_count{path="v1.5"}' in body


def test_prometheus_format_is_parseable(client):
    """Smoke: produced output should parse with prometheus_client.parser."""
    pytest.importorskip("prometheus_client")
    from prometheus_client.parser import text_string_to_metric_families

    client.post("/route/v15", json={"prompt": "hello"})
    r = client.get("/metrics/prometheus")
    families = list(text_string_to_metric_families(r.text))
    names = {f.name for f in families}
    # prometheus_client normalizes counter family names by stripping the
    # `_total` suffix per OpenMetrics convention.
    assert "gemma4_router_requests" in names
    assert "gemma4_router_latency_seconds" in names
