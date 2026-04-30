"""HTTP-level tests for the /route/v15 endpoint using mock-mode pipeline."""

from __future__ import annotations

import os

os.environ["V15_ENABLED"] = "true"
os.environ["V15_MOCK_MODE"] = "true"
os.environ["V15_DEVICE"] = "cpu"

import pytest
from fastapi.testclient import TestClient

import router.config as config_module
from router.api import app


@pytest.fixture
def client():
    # Drop any cached Settings so V15 env vars take effect on lifespan.
    config_module._settings = None
    with TestClient(app) as c:
        yield c


def test_v15_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_v15_route_returns_canned_mock(client):
    r = client.post("/route/v15", json={"prompt": "Hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "v1.5"
    assert body["schema_version"] == "1.5"
    assert body["complexity"] in {"light", "heavy"}
    assert "[mock-v15-response" in body["answer"]
    assert body["embedding_dim"] == 4096
    assert body["latency_ms"] is not None


def test_v15_route_long_prompt_classified_heavy(client):
    long_prompt = "explain quantum entanglement in detail" * 50
    r = client.post("/route/v15", json={"prompt": long_prompt})
    assert r.status_code == 200
    assert r.json()["complexity"] == "heavy"


def test_v15_disabled_returns_503(client):
    app.state.v15_pipeline._loaded = False
    try:
        r = client.post("/route/v15", json={"prompt": "test"})
        assert r.status_code == 503
    finally:
        app.state.v15_pipeline._loaded = True
