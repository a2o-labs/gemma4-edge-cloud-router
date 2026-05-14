#!/usr/bin/env python3
"""V1.5 cross-machine concurrency sweep.

Fires N concurrent end-to-end requests through L4 -> A100 and measures
tail latency.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post_json(url: str, payload: dict, timeout: float = 180.0) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, (time.perf_counter() - t0) * 1000


def one_request(edge_url: str, cloud_url: str, prompt: str, mnt: int) -> dict:
    t_start = time.perf_counter()
    enc, enc_rtt = post_json(f"{edge_url}/v15/encode", {"prompt": prompt})
    fwd, fwd_rtt = post_json(
        f"{cloud_url}/v15/forward",
        {"schema_json": enc["schema_json"], "max_new_tokens": mnt},
    )
    total = (time.perf_counter() - t_start) * 1000
    return {
        "prompt": prompt,
        "encode_rtt_ms": round(enc_rtt, 1),
        "encode_compute_ms": enc["encode_ms"],
        "forward_rtt_ms": round(fwd_rtt, 1),
        "forward_compute_ms": fwd["forward_ms"],
        "total_ms": round(total, 1),
    }


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
    p.add_argument("--edge-url", default="http://<edge-host>:8002")
    p.add_argument("--cloud-url", default="http://<cloud-host>:8001")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--requests-per-level", type=int, default=8)
    p.add_argument("--out", default="/tmp/v15_xmachine_concurrency.json")
    args = p.parse_args()

    base_prompts = [
        f"Concurrency-test prompt {i}: produce an answer about topic {i}"
        for i in range(args.requests_per_level)
    ]

    results = []
    for C in args.concurrency_levels:
        # Always issue requests_per_level requests so different C see same workload
        prompts = base_prompts[: args.requests_per_level]
        print(f"\n=== concurrency={C}  requests={len(prompts)} ===")
        wall_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=C) as ex:
            futs = [
                ex.submit(
                    one_request, args.edge_url, args.cloud_url, pr, args.max_new_tokens
                )
                for pr in prompts
            ]
            per = [f.result() for f in as_completed(futs)]
        wall_ms = (time.perf_counter() - wall_t0) * 1000
        totals = [r["total_ms"] for r in per]
        forward = [r["forward_rtt_ms"] for r in per]
        encode = [r["encode_rtt_ms"] for r in per]
        out = {
            "concurrency": C,
            "n_requests": len(per),
            "wall_clock_ms": round(wall_ms, 1),
            "throughput_rps": round(len(per) / (wall_ms / 1000.0), 3),
            "total_ms": stats(totals),
            "forward_rtt_ms": stats(forward),
            "encode_rtt_ms": stats(encode),
        }
        results.append(out)
        print(json.dumps(out, indent=2))

    summary = {
        "edge_url": args.edge_url,
        "cloud_url": args.cloud_url,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
