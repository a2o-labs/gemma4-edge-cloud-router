"""Integration: PromptSizeMiddleware + RateLimitMiddleware wired into app.

These tests verify that:
1. The middleware classes are registered on the global ``router.api.app``
   when ``MIDDLEWARE_ENABLED`` is true (the default).
2. Functional behaviour (413 / 429) is exercised on a *separate* FastAPI
   app constructed in-test so we don't pollute the global app's state
   with strict limits that would break other test modules' fixtures.

The standalone middleware-class behaviour is already covered by
``tests/test_middleware.py``; these tests focus on the wire-up.
"""

from __future__ import annotations

import os

os.environ.setdefault("V15_ENABLED", "true")
os.environ.setdefault("V15_MOCK_MODE", "true")
os.environ.setdefault("V15_DEVICE", "cpu")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _build_isolated_app(max_chars: int, max_requests: int) -> FastAPI:
    """Mirror the middleware wiring in router.api on a fresh FastAPI app.

    Done here so we don't mutate the global app's middleware stack and
    spill strict limits into other test modules (test_route_v15 etc.).
    """
    from router.middleware import PromptSizeMiddleware, RateLimitMiddleware

    app = FastAPI()
    # Same registration order as router.api: RateLimit then PromptSize so
    # PromptSize runs first (cheap reject before rate-limit ledger entry).
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=max_requests,
        window_seconds=10.0,
    )
    app.add_middleware(PromptSizeMiddleware, max_chars=max_chars)

    @app.post("/route")
    async def route(payload: dict):
        return {"ok": True}

    @app.post("/route/v15")
    async def route_v15(payload: dict):
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_router_api_app_has_middleware_registered() -> None:
    """router.api.app must register the CombinedRouteGuard when
    MIDDLEWARE_ENABLED is true (the default).
    """
    from router.middleware import CombinedRouteGuard
    from router.api import app

    classes = {m.cls for m in app.user_middleware}
    assert CombinedRouteGuard in classes


def test_oversized_prompt_returns_413_on_route() -> None:
    app = _build_isolated_app(max_chars=100, max_requests=999)
    with TestClient(app) as c:
        r = c.post("/route", json={"prompt": "x" * 200})
        assert r.status_code == 413
        assert "too long" in r.json()["detail"]


def test_oversized_prompt_returns_413_on_route_v15() -> None:
    app = _build_isolated_app(max_chars=100, max_requests=999)
    with TestClient(app) as c:
        r = c.post("/route/v15", json={"prompt": "x" * 200})
        assert r.status_code == 413


def test_health_unaffected_by_middleware() -> None:
    app = _build_isolated_app(max_chars=10, max_requests=1)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200


def test_rate_limit_kicks_in_after_max_requests() -> None:
    app = _build_isolated_app(max_chars=999, max_requests=5)
    with TestClient(app) as c:
        for _ in range(5):
            r = c.post(
                "/route/v15", json={"prompt": "ok", "session_id": "rl-test-sess"}
            )
            assert r.status_code == 200

        r = c.post(
            "/route/v15", json={"prompt": "ok", "session_id": "rl-test-sess"}
        )
        assert r.status_code == 429
        assert "Retry-After" in r.headers
