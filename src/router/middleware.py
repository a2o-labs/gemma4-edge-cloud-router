"""Input validation + rate-limiting middleware for the router.

Standalone — NOT auto-installed on the FastAPI app. Future PR will
wire it via app.add_middleware in src/router/api.py.

Two layers:

1. PromptSizeMiddleware — rejects POST /route* requests whose body's
   `prompt` field is over `max_chars` (default 50_000). Returns 413.

2. RateLimitMiddleware — in-process per-session_id sliding-window rate
   limiter. session_id is taken from the request body. Default
   60 req / 60s. Returns 429 with Retry-After.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


_ROUTE_PREFIXES = ("/route",)


def _is_route_request(req: Request) -> bool:
    return req.method == "POST" and any(req.url.path.startswith(p) for p in _ROUTE_PREFIXES)


async def _read_body_json(req: Request) -> dict:
    """Read + parse JSON body. Returns {} on parse failure.

    Sets request._body so downstream consumers don't read an empty stream.
    """
    body_bytes = await req.body()

    async def receive():
        return {"type": "http.request", "body": body_bytes}

    req._receive = receive  # type: ignore[attr-defined]
    if not body_bytes:
        return {}
    try:
        return json.loads(body_bytes)
    except json.JSONDecodeError:
        return {}


class PromptSizeMiddleware(BaseHTTPMiddleware):
    """Reject /route* POSTs whose `prompt` is over max_chars."""

    def __init__(self, app, max_chars: int = 50_000) -> None:
        super().__init__(app)
        self.max_chars = max_chars

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable],
    ):
        if _is_route_request(request):
            body = await _read_body_json(request)
            prompt = body.get("prompt", "")
            if len(prompt) > self.max_chars:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (f"prompt too long: {len(prompt)} chars > max {self.max_chars}"),
                    },
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-process sliding-window rate limit per session_id.

    Default: 60 requests per 60 seconds. session_id is read from JSON
    body. Anonymous requests (no session_id) share a single bucket
    keyed on '_anon'.
    """

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def _check(self, session_id: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        bucket = self._buckets.setdefault(session_id, deque())
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry_after = self.window - (now - bucket[0])
            return False, max(0.0, retry_after)
        bucket.append(now)
        return True, 0.0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable],
    ):
        if not _is_route_request(request):
            return await call_next(request)

        body = await _read_body_json(request)
        session_id = body.get("session_id") or "_anon"

        allowed, retry_after = self._check(session_id)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"rate limit: {self.max_requests} req / {self.window:.0f}s exceeded"
                    ),
                },
                headers={"Retry-After": f"{retry_after:.1f}"},
            )
        return await call_next(request)
