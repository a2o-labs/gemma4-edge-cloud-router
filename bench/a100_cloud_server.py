"""V1.5 cross-machine bench: cloud-side HTTP server on A100.

Wraps SoftPromptAdapter as a FastAPI service. Listens on :8001.

POST /v15/forward
  request:  {"schema_json": "...", "max_new_tokens": 64}
  response: {"output": "...", "forward_ms": 1234.5, "decode_ms": ...}

The schema_json is the CompactSchemaV15 produced by the edge side; we
extract embedding_b64, decode to numpy float16 vector, run forward.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from router.cloud_adapter import SoftPromptAdapter
from router.schema_v15 import CompactSchemaV15
from router.edge_encoder import decode_embedding_from_schema


class ForwardRequest(BaseModel):
    schema_json: str
    max_new_tokens: int = 64


class ForwardBatchRequest(BaseModel):
    schema_jsons: list[str]
    max_new_tokens: int = 64


class HealthResponse(BaseModel):
    ok: bool
    cloud_model: str
    cloud_hidden: int
    edge_dim: int
    prompt_tokens: int
    quantization: str
    cuda_mem_used_gb: float
    cuda_mem_total_gb: float
    cuda_device: str


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cloud-model", default="google/gemma-3-27b-it")
    p.add_argument("--edge-dim", type=int, default=1152)
    p.add_argument("--prompt-tokens", type=int, default=8)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--bind", default="0.0.0.0")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print("[load] HF_TOKEN unset; HF_HUB_OFFLINE=1 (cache must contain model)")

    from transformers import BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print(f"[load] cloud={args.cloud_model} 4bit edge_dim={args.edge_dim} K={args.prompt_tokens}")
    t0 = time.perf_counter()
    adapter = SoftPromptAdapter(
        edge_dim=args.edge_dim,
        prompt_tokens=args.prompt_tokens,
        hf_token=hf_token,
        cloud_model_name=args.cloud_model,
        device="cuda",
        dtype=torch.float16,
        quantization_config=bnb_cfg,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    dev_name = torch.cuda.get_device_name(0)
    print(f"[load] done {load_ms:.0f}ms  mem={mem_used:.2f}/{mem_total:.1f}GB  hidden={adapter.cloud_hidden}")

    app = FastAPI()

    @app.get("/v15/health")
    def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            cloud_model=args.cloud_model,
            cloud_hidden=adapter.cloud_hidden,
            edge_dim=args.edge_dim,
            prompt_tokens=args.prompt_tokens,
            quantization="bnb_4bit_nf4",
            cuda_mem_used_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
            cuda_mem_total_gb=round(mem_total, 1),
            cuda_device=dev_name,
        )

    @app.post("/v15/forward_batch")
    def forward_batch(req: ForwardBatchRequest) -> dict:
        from router.schema_v15 import CompactSchemaV15 as CS
        schemas = [CS.model_validate_json(j) for j in req.schema_jsons]
        for s in schemas:
            if s.embedding_dim != args.edge_dim:
                return {"error": f"edge_dim mismatch server={args.edge_dim} client={s.embedding_dim}"}
        vecs = np.stack(
            [decode_embedding_from_schema(s).astype(np.float32) for s in schemas]
        )
        edge_b = torch.from_numpy(vecs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = adapter.forward_batch(
            edge_b, req.schema_jsons, max_new_tokens=req.max_new_tokens
        )
        torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - t0) * 1000
        return {
            "outputs": outs,
            "n_outputs": len(outs),
            "forward_ms": round(forward_ms, 1),
            "tokens_per_sec": round(
                len(outs) * req.max_new_tokens / (forward_ms / 1000.0), 1
            ),
        }

    @app.post("/v15/forward")
    def forward(req: ForwardRequest) -> dict:
        schema = CompactSchemaV15.model_validate_json(req.schema_json)
        if schema.embedding_dim != args.edge_dim:
            return {"error": f"edge_dim mismatch server={args.edge_dim} client={schema.embedding_dim}"}
        t0 = time.perf_counter()
        vec = decode_embedding_from_schema(schema)
        decode_ms = (time.perf_counter() - t0) * 1000

        edge_t = torch.from_numpy(vec.astype(np.float32))

        t0 = time.perf_counter()
        # Inline what adapter.forward does so we can also return out_ids info
        if edge_t.dim() == 1:
            edge_t = edge_t.unsqueeze(0)
        edge_t = edge_t.to(device=adapter.device, dtype=adapter.dtype)
        soft = adapter.mlp(edge_t).reshape(
            edge_t.shape[0], adapter.prompt_tokens, adapter.cloud_hidden
        )
        text_embeds = adapter._embed_input(req.schema_json)
        if text_embeds.shape[0] != soft.shape[0]:
            text_embeds = text_embeds.expand(soft.shape[0], -1, -1)
        embeds = torch.cat([soft, text_embeds], dim=1)
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=adapter.device)
        with torch.no_grad():
            out_ids = adapter.cloud.generate(
                inputs_embeds=embeds,
                attention_mask=attn,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                pad_token_id=adapter.tokenizer.pad_token_id,
            )
        forward_ms = (time.perf_counter() - t0) * 1000
        out_skip = adapter.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        out_raw = adapter.tokenizer.decode(out_ids[0], skip_special_tokens=False)
        return {
            "output": out_skip,
            "output_raw": out_raw,
            "n_out_tokens": int(out_ids.shape[1]),
            "out_ids_head": out_ids[0, :8].tolist(),
            "soft_prompt_in_tokens": int(embeds.shape[1]),
            "forward_ms": round(forward_ms, 1),
            "decode_ms": round(decode_ms, 2),
        }

    uvicorn.run(app, host=args.bind, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
