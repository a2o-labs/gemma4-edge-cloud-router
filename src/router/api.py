"""FastAPI surface for the router."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

from .classifier import Classifier, ChatClient
from .cloud_forwarder import CloudForwarder
from .config import get_settings
from .decoder import Decoder
from .edge_executor import EdgeExecutor
from .encoder import Encoder, MaskMap
from .schema import (
    RouteRequest,
    RouteResponse,
    V15RouteRequest,
    V15RouteResponse,
)
from .telemetry import Telemetry
from .v15_pipeline import V15Pipeline

log = logging.getLogger("router")


class _EdgeChatClient(ChatClient):
    """Thin wrapper around the edge LLM chat endpoint for the classifier."""

    def __init__(self, executor: EdgeExecutor) -> None:
        self._executor = executor

    async def chat(self, system: str, user: str) -> str:
        base = self._executor.settings.base_url.rstrip("/")
        payload = {
            "model": self._executor.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self._executor.settings.api_key}"}
        async with httpx.AsyncClient(timeout=self._executor.settings.request_timeout_s) as c:
            r = await c.post(f"{base}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    mask_map = MaskMap(ttl_seconds=settings.router.mask_map_ttl_seconds)
    edge_exec = EdgeExecutor(settings.edge)
    app.state.settings = settings
    app.state.mask_map = mask_map
    app.state.edge_executor = edge_exec
    app.state.encoder = Encoder(mask_map)
    app.state.decoder = Decoder(mask_map)
    app.state.cloud_forwarder = CloudForwarder(settings.cloud)
    app.state.classifier = Classifier(_EdgeChatClient(edge_exec), settings.classifier)
    app.state.telemetry = Telemetry()

    v15_pipeline = V15Pipeline(settings.v15)
    try:
        v15_pipeline.load()
    except Exception as e:
        log.warning("V1.5 pipeline load failed (continuing in V1-only mode): %s", e)
    app.state.v15_pipeline = v15_pipeline

    log.info(
        "router ready — edge=%s cloud=%s v1.5=%s",
        settings.edge.base_url,
        settings.cloud.base_url,
        "ready" if v15_pipeline.is_ready() else "off",
    )
    yield


app = FastAPI(title="gemma4-router", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    return app.state.telemetry.snapshot()


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest) -> RouteResponse:
    session_id = req.session_id or uuid.uuid4().hex
    start = time.perf_counter()

    classifier = app.state.classifier
    encoder: Encoder = app.state.encoder
    decoder: Decoder = app.state.decoder
    forwarder: CloudForwarder = app.state.cloud_forwarder
    edge_exec: EdgeExecutor = app.state.edge_executor
    telemetry: Telemetry = app.state.telemetry

    if req.force in {"light", "heavy"}:
        complexity = req.force
        confidence = 1.0
        task_type = "other"
        fell_back = False
    else:
        result = await classifier.classify(req.prompt)
        complexity = result.complexity
        confidence = result.confidence
        task_type = result.task_type
        fell_back = result.fell_back

    try:
        if complexity == "light":
            answer = await edge_exec.answer(req.prompt)
            task_id = uuid.uuid4().hex
        else:
            task = encoder.encode(req.prompt, session_id=session_id, task_type=task_type)
            task_id = task.task_id
            cloud_answer = await forwarder.forward(task)
            answer = decoder.decode(session_id, cloud_answer)
    except httpx.HTTPError as exc:
        log.exception("upstream error on %s path", complexity)
        raise HTTPException(status_code=502, detail=f"upstream {complexity} error: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    telemetry.record(path=complexity, latency_ms=latency_ms, classifier_fell_back=fell_back)

    return RouteResponse(
        task_id=task_id,
        path=complexity,  # type: ignore[arg-type]
        answer=answer,
        classifier_confidence=confidence,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/route/v15", response_model=V15RouteResponse)
async def route_v15(req: V15RouteRequest) -> V15RouteResponse:
    pipeline: V15Pipeline = app.state.v15_pipeline
    if not pipeline.is_ready():
        raise HTTPException(status_code=503, detail="V1.5 pipeline disabled or not loaded")
    start = time.perf_counter()
    try:
        schema, answer = pipeline.run(req.prompt)
    except Exception as exc:
        log.exception("V1.5 pipeline error")
        raise HTTPException(status_code=502, detail=f"v1.5 pipeline error: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000
    app.state.telemetry.record(path="v1.5", latency_ms=latency_ms, classifier_fell_back=False)
    return V15RouteResponse(
        task_id=schema.task_id,
        answer=answer,
        embedding_dim=schema.embedding_dim,
        complexity=schema.complexity,
        latency_ms=round(latency_ms, 2),
    )


def run() -> None:
    import uvicorn

    s = get_settings().router
    uvicorn.run("router.api:app", host=s.host, port=s.port, log_level=s.log_level.lower())
