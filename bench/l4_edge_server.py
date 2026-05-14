"""V1.5 cross-machine bench: edge-side HTTP server on L4.

Wraps EdgeEncoder as a FastAPI service. Listens on :8002.

POST /v15/encode
  request:  {"prompt": "..."}
  response: {"schema_json": "...", "encode_ms": ..., "embedding_dim": ...}

The schema_json is a CompactSchemaV15 with embedding_b64 set; pass it
verbatim to the cloud-side server.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from router.edge_encoder import EdgeEncoder


class EncodeRequest(BaseModel):
    prompt: str


class HealthResponse(BaseModel):
    ok: bool
    edge_model: str
    embedding_dim: int
    cuda_mem_used_gb: float
    cuda_mem_total_gb: float
    cuda_device: str


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edge-model", default="google/gemma-3-1b-it")
    p.add_argument("--embedding-dim", type=int, default=1152)
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--bind", default="0.0.0.0")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[load] HF_TOKEN unset; transformers will fall back to ~/.cache/huggingface/token")

    print(f"[load] edge={args.edge_model} dim={args.embedding_dim}")
    t0 = time.perf_counter()
    enc = EdgeEncoder(
        model_name=args.edge_model,
        embedding_dim=args.embedding_dim,
        hf_token=hf_token,
        device="cuda",
        dtype=torch.float16,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    dev_name = torch.cuda.get_device_name(0)
    print(f"[load] done {load_ms:.0f}ms  mem={mem_used:.2f}/{mem_total:.1f}GB")

    app = FastAPI()

    @app.get("/v15/health")
    def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            edge_model=args.edge_model,
            embedding_dim=args.embedding_dim,
            cuda_mem_used_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
            cuda_mem_total_gb=round(mem_total, 1),
            cuda_device=dev_name,
        )

    @app.post("/v15/encode")
    def encode(req: EncodeRequest) -> dict:
        t0 = time.perf_counter()
        schema, vec = enc.encode(req.prompt)
        encode_ms = (time.perf_counter() - t0) * 1000
        return {
            "schema_json": schema.model_dump_json(),
            "encode_ms": round(encode_ms, 1),
            "embedding_dim": args.embedding_dim,
            "complexity": schema.complexity,
        }

    uvicorn.run(app, host=args.bind, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
