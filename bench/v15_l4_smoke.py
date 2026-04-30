"""V1.5 real-weight smoke test on L4 GPU.

Loads a real Gemma-family instruct model via EdgeEncoder + SoftPromptAdapter,
exercises the encode -> wire (base64 round-trip) -> cloud-adapter forward
pipeline against real CUDA weights, and prints latency + shape + dtype
assertions.

Default model is ``google/gemma-3-4b-it`` (~8GB fp16, comfortably fits on
a single L4 24GB). The original V1.5 spec referenced ``google/gemma-4-4b-it``
but that model ID does not exist on the Hugging Face Hub (2026-04 check ->
HTTP 404). Gemma 4 currently ships only as the 26B-A4B Mixture, which at
fp16 is ~52GB and exceeds L4 VRAM. Override with ``--edge-model`` /
``--cloud-model`` once a true Gemma 4 4B SKU is published.

Usage (on L4):
    HF_TOKEN=... python bench/v15_l4_smoke.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from router.cloud_adapter import SoftPromptAdapter
from router.edge_encoder import EdgeEncoder, decode_embedding_from_schema


DEFAULT_MODEL = "google/gemma-3-4b-it"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--edge-model", default=DEFAULT_MODEL)
    p.add_argument("--cloud-model", default=DEFAULT_MODEL)
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Edge projection output dim. Decoupled from base hidden_size; "
        "EdgeEncoder.projection: Linear(base_hidden -> embedding_dim).",
    )
    p.add_argument("--prompt-tokens", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Hello world",
            "Explain photosynthesis briefly",
            "What is 2+2?",
            "Write a haiku about autumn rain",
            "Translate 'good morning' to Japanese",
        ],
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def time_op(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = (time.perf_counter() - t0) * 1000
    print(f"[{label}] {dt:.1f}ms")
    return out, dt


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN env var required")

    print("=== V1.5 L4 smoke test ===")
    print(f"edge_model={args.edge_model}  cloud_model={args.cloud_model}")
    print(f"embedding_dim={args.embedding_dim}  prompt_tokens={args.prompt_tokens}")
    print(f"device={args.device}  cuda_device={torch.cuda.get_device_name(0)}")
    print(f"cuda_mem_total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print()

    enc, edge_load_ms = time_op(
        "edge.load",
        lambda: EdgeEncoder(
            model_name=args.edge_model,
            embedding_dim=args.embedding_dim,
            hf_token=hf_token,
            device=args.device,
            dtype=torch.float16,
        ),
    )
    edge_base_hidden = getattr(
        enc, "base_hidden", getattr(enc, "base_hidden_size", None)
    )
    print(f"  base_hidden_size={edge_base_hidden}")
    print(f"  cuda_mem_used={torch.cuda.memory_allocated()/1e9:.2f}GB")

    adapter, cloud_load_ms = time_op(
        "cloud.load",
        lambda: SoftPromptAdapter(
            edge_dim=args.embedding_dim,
            prompt_tokens=args.prompt_tokens,
            hf_token=hf_token,
            cloud_model_name=args.cloud_model,
            device=args.device,
            dtype=torch.float16,
        ),
    )
    print(f"  cloud_hidden_size={adapter.cloud_hidden}")
    print(f"  cuda_mem_used={torch.cuda.memory_allocated()/1e9:.2f}GB")
    print()

    encode_lat = []
    forward_lat = []
    for i, prompt in enumerate(args.prompts):
        (schema, vec), enc_ms = time_op(f"encode[{i}]", lambda: enc.encode(prompt))
        encode_lat.append(enc_ms)

        assert vec.shape == (args.embedding_dim,), f"vec shape {vec.shape}"
        assert vec.dtype in (np.float16, np.float32), f"vec dtype {vec.dtype}"
        assert schema.version == "1.5"
        assert schema.embedding_dim == args.embedding_dim
        assert schema.embedding_b64 and len(schema.embedding_b64) > 0

        decoded = decode_embedding_from_schema(schema)
        np.testing.assert_allclose(decoded, vec, rtol=1e-3, atol=1e-3)

        edge_t = torch.from_numpy(decoded.astype(np.float32))
        json_text = schema.model_dump_json()
        out, fwd_ms = time_op(
            f"forward[{i}]",
            lambda: adapter.forward(
                edge_t, json_text, max_new_tokens=args.max_new_tokens
            ),
        )
        forward_lat.append(fwd_ms)
        print(
            f"  classify={schema.complexity}  out_chars={len(out)}  "
            f"out_preview={out[:80]!r}"
        )
        print()

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            "n": n,
            "p50": round(xs[n // 2], 1),
            "p95": round(xs[min(n - 1, int(n * 0.95))], 1),
            "min": round(xs[0], 1),
            "max": round(xs[-1], 1),
        }

    summary = {
        "edge_model": args.edge_model,
        "cloud_model": args.cloud_model,
        "embedding_dim": args.embedding_dim,
        "prompt_tokens": args.prompt_tokens,
        "edge_load_ms": round(edge_load_ms, 1),
        "cloud_load_ms": round(cloud_load_ms, 1),
        "encode": stats(encode_lat),
        "forward": stats(forward_lat),
        "cuda_mem_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        ),
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
