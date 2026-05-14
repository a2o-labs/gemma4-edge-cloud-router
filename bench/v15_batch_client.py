#!/usr/bin/env python3
"""V1.5 batch sweep via the A100 /v15/forward_batch HTTP endpoint.

Pre-encodes once per prompt on L4, then issues batched forward calls
of size B for each requested batch size.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post_json(url: str, payload: dict, timeout: float = 600.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, (time.perf_counter() - t0) * 1000


def get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--edge-url", default="http://<edge-host>:8002")
    p.add_argument("--cloud-url", default="http://<cloud-host>:8001")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", default="/tmp/v15_batch_throughput.json")
    args = p.parse_args()

    base_prompts = [
        "What is the capital of Japan?",
        "List the planets",
        "Write a poem about the ocean",
        "Solve: 17 * 23",
        "Define the word 'serendipity'",
        "Explain why the sky is blue",
        "Translate hello to French",
        "Name three rivers in Asia",
    ]

    # Pre-encode on L4 (cached after first miss)
    print("[encode] priming edge encoder cache...")
    schemas = []
    for pr in base_prompts:
        r, _ = post_json(f"{args.edge_url}/v15/encode", {"prompt": pr})
        schemas.append(r["schema_json"])
    print(f"[encode] {len(schemas)} schemas ready")

    cloud_h = get_json(f"{args.cloud_url}/v15/health")
    print(f"[cloud] {cloud_h}")

    # Warmup
    print("[warmup] B=1 warmup...")
    post_json(
        f"{args.cloud_url}/v15/forward_batch",
        {"schema_jsons": schemas[:1], "max_new_tokens": args.max_new_tokens},
        timeout=300,
    )

    results = []
    for B in args.batch_sizes:
        per_trial_ms = []
        per_trial_rtt_ms = []
        per_trial_toks = []
        for trial in range(args.trials):
            batch = [schemas[i % len(schemas)] for i in range(B)]
            resp, rtt_ms = post_json(
                f"{args.cloud_url}/v15/forward_batch",
                {"schema_jsons": batch, "max_new_tokens": args.max_new_tokens},
                timeout=600,
            )
            per_trial_ms.append(resp["forward_ms"])
            per_trial_rtt_ms.append(rtt_ms)
            per_trial_toks.append(resp["tokens_per_sec"])
        best_ms = min(per_trial_ms)
        median_ms = sorted(per_trial_ms)[len(per_trial_ms) // 2]
        best_toks_per_sec = max(per_trial_toks)
        per_user_ms = best_ms / B
        out = {
            "batch": B,
            "max_new_tokens": args.max_new_tokens,
            "trials_compute_ms": [round(x, 1) for x in per_trial_ms],
            "trials_rtt_ms": [round(x, 1) for x in per_trial_rtt_ms],
            "best_compute_ms": round(best_ms, 1),
            "median_compute_ms": round(median_ms, 1),
            "best_tokens_per_sec": round(best_toks_per_sec, 1),
            "per_user_latency_ms": round(per_user_ms, 1),
            "speedup_vs_serial_at_B1": None,  # filled below
        }
        results.append(out)
        print(
            f"[B={B}] best_compute={best_ms:.1f}ms  median={median_ms:.1f}ms  "
            f"toks/s={best_toks_per_sec:.1f}  per-user={per_user_ms:.1f}ms"
        )

    # Compute speedup vs serial (B=1)
    serial_ms = next(r["best_compute_ms"] for r in results if r["batch"] == 1)
    for r in results:
        ideal_serial_ms = r["batch"] * serial_ms
        r["speedup_vs_serial_at_B1"] = round(ideal_serial_ms / r["best_compute_ms"], 2)

    summary = {
        "edge_url": args.edge_url,
        "cloud_url": args.cloud_url,
        "cloud_model": cloud_h["cloud_model"],
        "quantization": cloud_h["quantization"],
        "cuda_device": cloud_h["cuda_device"],
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
