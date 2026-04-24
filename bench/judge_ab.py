"""LLM-as-judge offline A/B harness (stub).

Given a JSONL dataset of prompts with gold answers, run each prompt through
the router twice (force=light and force=heavy), then ask a judge model to
prefer one of the two outputs. Emits a CSV with (prompt_id, light_answer,
heavy_answer, judge_verdict, judge_reason).

This is a STUB: it wires up the call graph but uses a trivial heuristic judge
so the harness runs without an external judge model. Swap `_judge()` for a
real OpenAI-compatible call when the evaluation rig lands.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class Sample:
    prompt_id: str
    prompt: str
    gold: str | None = None


async def _route_once(client: httpx.AsyncClient, base_url: str, prompt: str, force: str) -> str:
    r = await client.post(
        f"{base_url.rstrip('/')}/route",
        json={"prompt": prompt, "session_id": f"bench-{force}", "force": force},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["answer"]


def _judge(prompt: str, light: str, heavy: str, gold: str | None) -> tuple[str, str]:
    """Trivial stub judge: longer response wins, tie → heavy.

    Replace with a real LLM-as-judge call keyed off `gold` when available.
    """
    if gold:
        lscore = sum(1 for w in gold.split() if w in light)
        hscore = sum(1 for w in gold.split() if w in heavy)
        if lscore > hscore:
            return "light", f"gold overlap {lscore}>{hscore}"
        if hscore > lscore:
            return "heavy", f"gold overlap {hscore}>{lscore}"
    if len(heavy) >= len(light):
        return "heavy", "longer response (stub)"
    return "light", "longer response (stub)"


async def run(samples: list[Sample], router_url: str, out_path: Path) -> None:
    async with httpx.AsyncClient() as client:
        rows = []
        for s in samples:
            light, heavy = await asyncio.gather(
                _route_once(client, router_url, s.prompt, "light"),
                _route_once(client, router_url, s.prompt, "heavy"),
            )
            verdict, reason = _judge(s.prompt, light, heavy, s.gold)
            rows.append(
                {
                    "prompt_id": s.prompt_id,
                    "light": light,
                    "heavy": heavy,
                    "verdict": verdict,
                    "reason": reason,
                }
            )
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "light", "heavy", "verdict", "reason"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


def _load_jsonl(path: Path) -> list[Sample]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(
            Sample(prompt_id=obj["id"], prompt=obj["prompt"], gold=obj.get("gold"))
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LLM-as-judge A/B harness (stub)")
    p.add_argument("--dataset", type=Path, required=True, help="JSONL with id/prompt/gold")
    p.add_argument("--router-url", default="http://127.0.0.1:8080")
    p.add_argument("--out", type=Path, default=Path("judge_ab_results.csv"))
    args = p.parse_args(argv)

    samples = _load_jsonl(args.dataset)
    asyncio.run(run(samples, args.router_url, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
