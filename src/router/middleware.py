"""Input validation + rate-limiting middleware for the router.

Implemented as raw ASGI middleware (not BaseHTTPMiddleware) so the
request body can be consumed for inspection and then replayed to the
downstream app cleanly. ``BaseHTTPMiddleware`` does not let us substitute
the receive callable for downstream consumers, so any approach that
chains body-reading BaseHTTPMiddleware with StreamingResponse routes
ends up tripping Starlette's receive-accounting and raising
``RuntimeError: Unexpected message received: http.request``.

Three classes exposed:

- ``PromptSizeMiddleware`` — size-only check (413 on oversized prompt).
- ``RateLimitMiddleware`` — sliding-window rate limit per ``session_id`` (429).
- ``CombinedRouteGuard`` — both checks in a single body read; preferred
  for production wiring.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Callable

from fastapi.responses import JSONResponse


_ROUTE_PREFIXES = ("/route",)


def _is_route_request(scope: dict) -> bool:
    if scope.get("type") != "http":
        return False
    if scope.get("method") != "POST":
        return False
    path = scope.get("path", "")
    return any(path.startswith(p) for p in _ROUTE_PREFIXES)


async def _read_body(receive: Callable) -> bytes:
    """Drain receive into a single bytes payload."""
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        if message["type"] != "http.request":
            continue
        body += message.get("body", b"")
        more_body = message.get("more_body", False)
    return body


def _replay_receive(body: bytes) -> Callable:
    """Make a receive() that replays ``body`` once then blocks forever.

    Returning ``http.disconnect`` immediately would cause StreamingResponse
    handlers to short-circuit (Starlette interprets disconnect as 'client
    closed connection' and cancels the response generator). Instead we
    block the second-and-later receive() calls indefinitely; the request
    lifecycle terminates naturally when the response handler returns.
    """
    import asyncio

    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # Block forever — connection still open until response completes.
        await asyncio.Event().wait()
        # Unreachable, but satisfy the type checker.
        return {"type": "http.disconnect"}

    return receive


def _parse_json(body: bytes) -> dict:
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


async def _send_json_response(send: Callable, status: int, content: dict, headers: dict | None = None) -> None:
    """Send a JSONResponse via the raw ASGI send callable."""
    response = JSONResponse(status_code=status, content=content, headers=headers or None)
    await response(
        {"type": "http", "method": "POST"},  # minimal scope; response only reads status/headers
        lambda: None,  # type: ignore[arg-type]
        send,
    )


class PromptSizeMiddleware:
    """ASGI middleware: rejects /route* POSTs whose ``prompt`` exceeds ``max_chars``."""

    def __init__(self, app, max_chars: int = 50_000) -> None:
        self.app = app
        self.max_chars = max_chars

    async def __call__(self, scope, receive, send):
        if not _is_route_request(scope):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        prompt = _parse_json(body).get("prompt", "")
        if len(prompt) > self.max_chars:
            await _send_json_response(
                send,
                413,
                {"detail": f"prompt too long: {len(prompt)} chars > max {self.max_chars}"},
            )
            return

        await self.app(scope, _replay_receive(body), send)


class RateLimitMiddleware:
    """ASGI middleware: sliding-window rate limit per ``session_id``."""

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        self.app = app
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def _check(self, session_id: str) -> tuple[bool, float]:
        now = time.monotonic()
        bucket = self._buckets.setdefault(session_id, deque())
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry_after = self.window - (now - bucket[0])
            return False, max(0.0, retry_after)
        bucket.append(now)
        return True, 0.0

    async def __call__(self, scope, receive, send):
        if not _is_route_request(scope):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        session_id = _parse_json(body).get("session_id") or "_anon"

        allowed, retry_after = self._check(session_id)
        if not allowed:
            await _send_json_response(
                send,
                429,
                {"detail": f"rate limit: {self.max_requests} req / {self.window:.0f}s exceeded"},
                headers={"Retry-After": f"{retry_after:.1f}"},
            )
            return

        await self.app(scope, _replay_receive(body), send)


class CombinedRouteGuard:
    """Single ASGI middleware: prompt-size + rate-limit in one body read.

    Prefer over stacking PromptSizeMiddleware + RateLimitMiddleware so the
    body is read once and the replay happens once.
    """

    def __init__(
        self,
        app,
        max_chars: int = 50_000,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        self.app = app
        self.max_chars = max_chars
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def _rate_check(self, session_id: str) -> tuple[bool, float]:
        now = time.monotonic()
        bucket = self._buckets.setdefault(session_id, deque())
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry_after = self.window - (now - bucket[0])
            return False, max(0.0, retry_after)
        bucket.append(now)
        return True, 0.0

    async def __call__(self, scope, receive, send):
        if not _is_route_request(scope):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        parsed = _parse_json(body)

        prompt = parsed.get("prompt", "")
        if len(prompt) > self.max_chars:
            await _send_json_response(
                send,
                413,
                {"detail": f"prompt too long: {len(prompt)} chars > max {self.max_chars}"},
            )
            return

        session_id = parsed.get("session_id") or "_anon"
        allowed, retry_after = self._rate_check(session_id)
        if not allowed:
            await _send_json_response(
                send,
                429,
                {"detail": f"rate limit: {self.max_requests} req / {self.window:.0f}s exceeded"},
                headers={"Retry-After": f"{retry_after:.1f}"},
            )
            return

        await self.app(scope, _replay_receive(body), send)
