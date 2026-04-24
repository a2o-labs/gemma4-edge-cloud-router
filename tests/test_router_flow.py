"""End-to-end smoke test for /route with mocked upstream LLMs."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from router.api import app


class _FakeEdge:
    async def answer(self, prompt: str) -> str:
        return f"edge-answer:{prompt[:20]}"


class _FakeCloud:
    async def forward(self, task) -> str:  # noqa: ANN001
        return f"cloud-answer:{task.task_id}"


class _FakeClassifier:
    def __init__(self, complexity: str) -> None:
        self.complexity = complexity

    async def classify(self, prompt: str):  # noqa: ANN001
        class R:
            pass

        r = R()
        r.complexity = self.complexity
        r.confidence = 0.95
        r.task_type = "qa"
        r.fell_back = False
        return r


@pytest.fixture
def client_light():
    with TestClient(app) as c:
        app.state.edge_executor = _FakeEdge()
        app.state.cloud_forwarder = _FakeCloud()
        app.state.classifier = _FakeClassifier("light")
        yield c


@pytest.fixture
def client_heavy():
    with TestClient(app) as c:
        app.state.edge_executor = _FakeEdge()
        app.state.cloud_forwarder = _FakeCloud()
        app.state.classifier = _FakeClassifier("heavy")
        yield c


def test_health(client_light):
    assert client_light.get("/health").json() == {"status": "ok"}


def test_route_light(client_light):
    r = client_light.post("/route", json={"prompt": "hi", "session_id": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "light"
    assert body["answer"].startswith("edge-answer:")


def test_route_heavy_encodes_and_forwards(client_heavy):
    r = client_heavy.post(
        "/route",
        json={"prompt": "please email alice@example.com", "session_id": "s2"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "heavy"
    assert body["answer"].startswith("cloud-answer:")


def test_force_flag_bypasses_classifier(client_light):
    r = client_light.post(
        "/route",
        json={"prompt": "force me", "session_id": "s3", "force": "heavy"},
    )
    assert r.status_code == 200
    assert r.json()["path"] == "heavy"


def test_metrics_increments(client_light):
    client_light.post("/route", json={"prompt": "a", "session_id": "m"})
    m = client_light.get("/metrics").json()
    assert m["total"] >= 1
