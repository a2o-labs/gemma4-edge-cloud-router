"""Standalone tests — build a tiny test app, attach the middleware,
verify it without depending on the main router app."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from router.middleware import PromptSizeMiddleware, RateLimitMiddleware


def _make_test_app(middleware_class, **kwargs):
    app = FastAPI()
    app.add_middleware(middleware_class, **kwargs)

    @app.post("/route")
    async def route(payload: dict):
        return {"ok": True, "echo_len": len(payload.get("prompt", ""))}

    @app.post("/route/v15")
    async def route_v15(payload: dict):
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_prompt_size_under_limit_passes():
    app = _make_test_app(PromptSizeMiddleware, max_chars=100)
    with TestClient(app) as c:
        r = c.post("/route", json={"prompt": "ok"})
        assert r.status_code == 200


def test_prompt_size_over_limit_returns_413():
    app = _make_test_app(PromptSizeMiddleware, max_chars=10)
    with TestClient(app) as c:
        r = c.post("/route", json={"prompt": "x" * 100})
        assert r.status_code == 413
        assert "too long" in r.json()["detail"]


def test_prompt_size_skips_non_route_paths():
    """Non-/route requests are NOT inspected (e.g., /health)."""
    app = _make_test_app(PromptSizeMiddleware, max_chars=10)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200


def test_rate_limit_under_threshold():
    app = _make_test_app(RateLimitMiddleware, max_requests=5, window_seconds=10.0)
    with TestClient(app) as c:
        for _ in range(5):
            r = c.post("/route", json={"prompt": "hi", "session_id": "s1"})
            assert r.status_code == 200


def test_rate_limit_over_threshold_returns_429():
    app = _make_test_app(RateLimitMiddleware, max_requests=3, window_seconds=10.0)
    with TestClient(app) as c:
        for _ in range(3):
            r = c.post("/route", json={"prompt": "hi", "session_id": "s2"})
            assert r.status_code == 200
        r = c.post("/route", json={"prompt": "hi", "session_id": "s2"})
        assert r.status_code == 429
        assert "Retry-After" in r.headers


def test_rate_limit_separate_buckets_per_session():
    app = _make_test_app(RateLimitMiddleware, max_requests=2, window_seconds=10.0)
    with TestClient(app) as c:
        c.post("/route", json={"prompt": "x", "session_id": "A"})
        c.post("/route", json={"prompt": "x", "session_id": "A"})
        r = c.post("/route", json={"prompt": "x", "session_id": "A"})
        assert r.status_code == 429

        r = c.post("/route", json={"prompt": "x", "session_id": "B"})
        assert r.status_code == 200


def test_rate_limit_anon_shared_bucket():
    """Requests without session_id share the _anon bucket."""
    app = _make_test_app(RateLimitMiddleware, max_requests=2, window_seconds=10.0)
    with TestClient(app) as c:
        c.post("/route", json={"prompt": "x"})
        c.post("/route", json={"prompt": "x"})
        r = c.post("/route", json={"prompt": "x"})
        assert r.status_code == 429
