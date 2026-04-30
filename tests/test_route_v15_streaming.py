"""SSE streaming tests for /route/v15/stream — mock-mode only."""

from __future__ import annotations

import json
import os

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


def _parse_sse(line_iter):
    events = []
    current = {"event": None, "data": ""}
    for raw in line_iter:
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if line == "":
            if current["event"]:
                events.append((current["event"], current["data"]))
            current = {"event": None, "data": ""}
            continue
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].lstrip()
    if current["event"]:
        events.append((current["event"], current["data"]))
    return events


def test_v15_stream_endpoint_emits_events(client):
    """Verify SSE frames arrive in order: schema, token+, done."""
    with client.stream("POST", "/route/v15/stream", json={"prompt": "Hello"}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = _parse_sse(r.iter_lines())

    event_types = [e[0] for e in events]
    assert event_types[0] == "schema"
    assert "token" in event_types
    assert event_types[-1] == "done"

    schema = json.loads(events[0][1])
    assert schema["version"] == "1.5"
    assert schema["embedding_dim"] == 4096

    done = json.loads(events[-1][1])
    assert "complexity" in done
    assert done["total_chars"] > 0


def test_v15_stream_disabled_returns_503(client):
    """If V1.5 disabled, /route/v15/stream returns 503 (no streaming starts)."""
    from router.api import app

    app.state.v15_pipeline._loaded = False
    try:
        r = client.post("/route/v15/stream", json={"prompt": "test"})
        assert r.status_code == 503
    finally:
        app.state.v15_pipeline._loaded = True
