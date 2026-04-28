"""V1.5 evaluation harness: success rate, token reduction, Brier, locust load.

Skeleton only. No real LLM calls executed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class EvalRow:
    prompt: str
    baseline_answer: str
    v15_answer: str
    baseline_tokens: int
    v15_prompt_tokens: int
    label_complexity: str
    pred_complexity_prob_heavy: float


def task_success_rate(rows: Iterable[EvalRow], judge=None) -> float:
    # TODO: pass each (prompt, baseline_answer, v15_answer) to LLM judge
    # (re-use bench/judge_ab.py from V1) and compute fraction where v15 wins
    # or ties (>= baseline quality).
    total = 0
    wins = 0
    for _ in rows:
        total += 1
    return float(wins) / total if total else 0.0


def token_reduction_ratio(rows: Iterable[EvalRow]) -> float:
    base = 0
    v15 = 0
    for r in rows:
        base += r.baseline_tokens
        v15 += r.v15_prompt_tokens
    if base == 0:
        return 0.0
    return 1.0 - (v15 / base)


def brier_score(rows: Iterable[EvalRow]) -> float:
    n = 0
    s = 0.0
    for r in rows:
        y = 1.0 if r.label_complexity == "heavy" else 0.0
        s += (r.pred_complexity_prob_heavy - y) ** 2
        n += 1
    return s / n if n else math.nan


def load_rows(path: str) -> List[EvalRow]:
    out: List[EvalRow] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(EvalRow(**d))
    return out


def locust_stub() -> None:
    """Placeholder for locust load test entrypoint.

    Run with: locust -f eval/eval_v15.py --headless -u 50 -r 5 -t 5m
    Once the LocustUser class below is fleshed out.
    """
    # TODO: from locust import HttpUser, task, between
    # class RouterUser(HttpUser):
    #     wait_time = between(0.5, 1.5)
    #     @task
    #     def route(self):
    #         self.client.post("/route", json={"prompt": "...", "session_id": "u1"})
    print("TODO real load test: locust class not implemented")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-jsonl", required=True)
    a = p.parse_args()
    rows = load_rows(a.eval_jsonl)
    print(json.dumps({
        "count": len(rows),
        "task_success_rate": task_success_rate(rows),
        "token_reduction_ratio": token_reduction_ratio(rows),
        "brier_score": brier_score(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
