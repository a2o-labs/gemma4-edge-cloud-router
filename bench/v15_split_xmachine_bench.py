#!/usr/bin/env python3
"""V1.5 cross-machine split bench: edge on L4, cloud on A100.

Pipeline per prompt:
    bench client --(http)--> L4 EdgeEncoder (gemma-3-1b-it fp16) --(http)--> A100 SoftPromptAdapter (gemma-3-27b-it 4bit nf4) --> response

Measures end-to-end and per-stage latency, decomposing into
encode_compute_ms (server-reported), forward_compute_ms, network_ms
(round-trip minus server work), total_ms.

Run from the bench client (any host on the tailnet that can reach both
endpoints).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any


def post_json(url: str, payload: dict, timeout: float = 120.0) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rtt_ms = (time.perf_counter() - t0) * 1000
    return data, rtt_ms


def get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    return dict(
        n=n,
        p50=round(xs[n // 2], 1),
        p95=round(xs[min(n - 1, int(n * 0.95))], 1),
        min=round(xs[0], 1),
        max=round(xs[-1], 1),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edge-url", default="http://localhost:8002")
    p.add_argument("--cloud-url", default="http://localhost:8001")
    p.add_argument("--max-new-tokens", type=int, default=64)
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
    p.add_argument("--out", default="/tmp/v15_xmachine_bench.json")
    args = p.parse_args()

    print("=== V1.5 cross-machine split bench ===")
    print(f"  edge_url  = {args.edge_url}")
    print(f"  cloud_url = {args.cloud_url}")

    edge_h = get_json(f"{args.edge_url}/v15/health")
    cloud_h = get_json(f"{args.cloud_url}/v15/health")
    print(f"  edge: {edge_h['edge_model']} dim={edge_h['embedding_dim']} mem={edge_h['cuda_mem_used_gb']:.2f}/{edge_h['cuda_mem_total_gb']}GB on {edge_h['cuda_device']}")
    print(f"  cloud: {cloud_h['cloud_model']} hidden={cloud_h['cloud_hidden']} {cloud_h['quantization']} mem={cloud_h['cuda_mem_used_gb']:.2f}/{cloud_h['cuda_mem_total_gb']}GB on {cloud_h['cuda_device']}")
    print()

    if edge_h["embedding_dim"] != cloud_h["edge_dim"]:
        raise SystemExit(
            f"dim mismatch: edge={edge_h['embedding_dim']} cloud={cloud_h['edge_dim']}"
        )

    encode_compute = []
    encode_rtt = []
    forward_compute = []
    forward_decode = []
    forward_rtt = []
    total = []
    answers: list[dict[str, Any]] = []

    for i, prompt in enumerate(args.prompts):
        # 1) edge encode (over the network)
        enc_resp, enc_rtt_ms = post_json(
            f"{args.edge_url}/v15/encode", {"prompt": prompt}
        )
        schema_json = enc_resp["schema_json"]
        encode_compute.append(enc_resp["encode_ms"])
        encode_rtt.append(enc_rtt_ms)

        # 2) cloud forward
        fwd_resp, fwd_rtt_ms = post_json(
            f"{args.cloud_url}/v15/forward",
            {"schema_json": schema_json, "max_new_tokens": args.max_new_tokens},
        )
        forward_compute.append(fwd_resp["forward_ms"])
        forward_decode.append(fwd_resp["decode_ms"])
        forward_rtt.append(fwd_rtt_ms)
        total.append(enc_rtt_ms + fwd_rtt_ms)

        ans = {
            "prompt": prompt,
            "complexity": enc_resp["complexity"],
            "schema_json_bytes": len(schema_json),
            "encode_compute_ms": enc_resp["encode_ms"],
            "encode_rtt_ms": round(enc_rtt_ms, 1),
            "forward_compute_ms": fwd_resp["forward_ms"],
            "forward_decode_ms": fwd_resp["decode_ms"],
            "forward_rtt_ms": round(fwd_rtt_ms, 1),
            "total_ms": round(enc_rtt_ms + fwd_rtt_ms, 1),
            "n_in_tokens": fwd_resp.get("soft_prompt_in_tokens"),
            "n_out_tokens": fwd_resp.get("n_out_tokens"),
            "output_chars": len(fwd_resp["output"]),
            "output_preview": fwd_resp["output"][:200],
            "output_raw_preview": (fwd_resp.get("output_raw") or "")[:200],
        }
        answers.append(ans)
        print(
            f"[{i}] {prompt!r}\n"
            f"    encode rtt={enc_rtt_ms:.1f}ms compute={enc_resp['encode_ms']:.1f}ms net={enc_rtt_ms-enc_resp['encode_ms']:.1f}ms\n"
            f"    forward rtt={fwd_rtt_ms:.1f}ms compute={fwd_resp['forward_ms']:.1f}ms net={fwd_rtt_ms-fwd_resp['forward_ms']:.1f}ms\n"
            f"    total={enc_rtt_ms+fwd_rtt_ms:.1f}ms class={enc_resp['complexity']} out={fwd_resp['output'][:120]!r}\n"
        )

    encode_net = [encode_rtt[i] - encode_compute[i] for i in range(len(encode_rtt))]
    forward_net = [forward_rtt[i] - forward_compute[i] for i in range(len(forward_rtt))]

    summary = {
        "edge": {
            "url": args.edge_url,
            "model": edge_h["edge_model"],
            "embedding_dim": edge_h["embedding_dim"],
            "cuda_device": edge_h["cuda_device"],
            "cuda_mem_used_gb": edge_h["cuda_mem_used_gb"],
        },
        "cloud": {
            "url": args.cloud_url,
            "model": cloud_h["cloud_model"],
            "hidden": cloud_h["cloud_hidden"],
            "quantization": cloud_h["quantization"],
            "prompt_tokens": cloud_h["prompt_tokens"],
            "cuda_device": cloud_h["cuda_device"],
            "cuda_mem_used_gb": cloud_h["cuda_mem_used_gb"],
        },
        "max_new_tokens": args.max_new_tokens,
        "n_prompts": len(args.prompts),
        "encode_compute_ms": stats(encode_compute),
        "encode_rtt_ms": stats(encode_rtt),
        "encode_network_ms": stats(encode_net),
        "forward_compute_ms": stats(forward_compute),
        "forward_rtt_ms": stats(forward_rtt),
        "forward_network_ms": stats(forward_net),
        "total_ms": stats(total),
        "answers": answers,
    }

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
