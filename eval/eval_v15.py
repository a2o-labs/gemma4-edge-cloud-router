"""V1.5 evaluation harness.

Runs a small canned dataset through both V1 and V1.5 paths and reports a
quality + compression delta. Has two modes:

    mock  - in-process, no HTTP, no real LLM. V15Pipeline runs in mock mode.
    http  - hits a running router at /route + /route/v15 over HTTP.

Judge can be `mock` (always answers "yes" - useful for CI) or `llm` (calls a
litellm-compatible endpoint).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import httpx

from router.schema_v15 import CompactSchemaV15

CANNED_DATASET_PATH = Path(__file__).parent / "data" / "eval_v15.jsonl"


@dataclass
class EvalSample:
    id: str
    prompt: str
    expected_answer: str
    expected_complexity: str
    task_type: str = "other"


@dataclass
class EvalArgs:
    """Lightweight stand-in for argparse.Namespace, used by tests."""

    mode: str = "mock"
    dataset: str = "canned"
    judge: str = "mock"
    router_base: str = "http://localhost:8080"
    judge_base: str = "http://localhost:4000/v1"
    judge_model: str = "claude-haiku-4-5"
    out: str | None = None


# ---------------------------------------------------------------------------
# dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path_or_keyword: str) -> list[EvalSample]:
    """Load a JSONL dataset. Pass `"canned"` for the bundled 10-sample set."""
    if path_or_keyword == "canned":
        path = CANNED_DATASET_PATH
    else:
        path = Path(path_or_keyword)
    rows: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                EvalSample(
                    id=d["id"],
                    prompt=d["prompt"],
                    expected_answer=d.get("expected_answer", ""),
                    expected_complexity=d.get("expected_complexity", "heavy"),
                    task_type=d.get("task_type", "other"),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _get_tokenizer():
    """Return a tiktoken encoder, lazily imported.

    cl100k_base is used as a *proxy* for Gemma's tokenizer. The absolute counts
    will differ from real Gemma tokens, but the relative reduction ratio is
    stable enough for A/B comparisons.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def token_reduction_ratio(
    full_prompt: str,
    schema_v15: CompactSchemaV15,
    tokenizer: Any | None = None,
) -> float:
    """Fraction of tokens saved by sending the compact schema instead of the
    raw prompt.

    Returns a value in roughly [-inf, 1.0]: 1.0 = perfect compression, 0.0 =
    same size, negative = compact form is *larger* (which can happen for very
    short prompts where the JSON envelope dominates).

    The embedding (`embedding_b64`) is excluded from the count because it ships
    out-of-band as a binary blob in the real wire format.
    """
    enc = tokenizer or _get_tokenizer()
    full_tokens = len(enc.encode(full_prompt))
    compact_json = schema_v15.model_dump_json(exclude={"embedding_b64"})
    compact_tokens = len(enc.encode(compact_json))
    return 1.0 - compact_tokens / max(full_tokens, 1)


def brier_score(predictions: Iterable[float], labels: Iterable[int]) -> float:
    """Mean squared error of probabilistic predictions.

    p in [0,1] is the predicted probability of "heavy"; y in {0,1} is the
    true label. Lower is better. Returns 0.0 for an empty input.
    """
    preds = list(predictions)
    labs = list(labels)
    if len(preds) != len(labs):
        raise ValueError(f"length mismatch: {len(preds)} preds vs {len(labs)} labels")
    if not preds:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(preds, labs)) / len(preds)


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


class ChatClient(Protocol):
    async def chat(self, system: str, user: str, model: str) -> str: ...


class MockJudge:
    """Always answers 'yes'. Lets CI exercise the harness without an LLM."""

    async def chat(self, system: str, user: str, model: str) -> str:  # noqa: ARG002
        return "yes"


class HttpJudge:
    """Calls an OpenAI-compatible /chat/completions endpoint (e.g. litellm)."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", "no-key")

    async def chat(self, system: str, user: str, model: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def build_judge_client(name: str, base_url: str | None = None) -> ChatClient:
    if name == "mock":
        return MockJudge()
    if name == "llm":
        if not base_url:
            raise ValueError("judge=llm requires --judge-base")
        return HttpJudge(base_url)
    raise ValueError(f"unknown judge: {name}")


JUDGE_SYSTEM = "You are an evaluator. Reply with only 'yes' or 'no'."

JUDGE_USER_TEMPLATE = (
    "Prompt: {prompt}\n"
    "Expected key facts: {expected}\n"
    "Answer: {answer}\n"
    "Does the answer correctly address the prompt and contain the expected "
    "key facts? Reply 'yes' or 'no'."
)


async def task_success_rate(
    samples: list[EvalSample],
    answers: list[str],
    judge_client: ChatClient,
    judge_model: str = "claude-haiku-4-5",
) -> dict[str, Any]:
    """Run an LLM-as-judge over (sample, answer) pairs.

    Returns dict(rate=float, per_sample=list[bool]).
    """
    if len(samples) != len(answers):
        raise ValueError(f"samples/answers length mismatch: {len(samples)} vs {len(answers)}")

    async def _judge(s: EvalSample, a: str) -> bool:
        user = JUDGE_USER_TEMPLATE.format(prompt=s.prompt, expected=s.expected_answer, answer=a)
        try:
            verdict = await judge_client.chat(JUDGE_SYSTEM, user, judge_model)
        except Exception:
            return False
        return verdict.strip().lower().startswith("yes")

    results = await asyncio.gather(*(_judge(s, a) for s, a in zip(samples, answers)))
    rate = sum(results) / len(results) if results else 0.0
    return {"rate": rate, "per_sample": list(results)}


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def _run_pipeline_mock(samples: list[EvalSample]) -> tuple[list[str], list[CompactSchemaV15]]:
    """Run V15Pipeline locally in mock mode. No HTTP, no GPU, no weights."""
    from router.config import V15Settings
    from router.v15_pipeline import V15Pipeline

    settings = V15Settings(enabled=True, mock_mode=True, device="cpu")
    pipeline = V15Pipeline(settings)
    pipeline.load()

    answers: list[str] = []
    schemas: list[CompactSchemaV15] = []
    for s in samples:
        schema, answer = pipeline.run(s.prompt)
        answers.append(answer)
        schemas.append(schema)
    return answers, schemas


async def _call_v1_http(prompt: str, base: str, client: httpx.AsyncClient) -> str:
    r = await client.post(f"{base.rstrip('/')}/route", json={"prompt": prompt}, timeout=60)
    r.raise_for_status()
    return r.json()["answer"]


async def _call_v15_http(
    prompt: str, base: str, client: httpx.AsyncClient
) -> tuple[str, CompactSchemaV15]:
    r = await client.post(f"{base.rstrip('/')}/route/v15", json={"prompt": prompt}, timeout=120)
    r.raise_for_status()
    body = r.json()
    schema = CompactSchemaV15(
        task_id=body["task_id"],
        task_type="other",
        complexity=body["complexity"],
        embedding_dim=body["embedding_dim"],
        confidence=0.0,
    )
    return body["answer"], schema


async def _run_pipeline_http(
    samples: list[EvalSample], router_base: str
) -> tuple[list[str], list[str], list[CompactSchemaV15]]:
    async with httpx.AsyncClient() as client:
        v1_task = asyncio.gather(*(_call_v1_http(s.prompt, router_base, client) for s in samples))
        v15_task = asyncio.gather(*(_call_v15_http(s.prompt, router_base, client) for s in samples))
        v1_answers, v15_pairs = await asyncio.gather(v1_task, v15_task)
    v15_answers = [a for a, _ in v15_pairs]
    v15_schemas = [s for _, s in v15_pairs]
    return v1_answers, v15_answers, v15_schemas


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------


async def run_eval(args: EvalArgs | argparse.Namespace) -> dict[str, Any]:
    samples = load_dataset(args.dataset)

    if args.mode == "mock":
        v1_answers = ["[mock-v1-answer]" for _ in samples]
        v15_answers, v15_schemas = _run_pipeline_mock(samples)
    elif args.mode == "http":
        v1_answers, v15_answers, v15_schemas = await _run_pipeline_http(samples, args.router_base)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    judge_base = getattr(args, "judge_base", None)
    judge_client = build_judge_client(args.judge, base_url=judge_base)

    v1_success = await task_success_rate(samples, v1_answers, judge_client, args.judge_model)
    v15_success = await task_success_rate(samples, v15_answers, judge_client, args.judge_model)

    trr = [token_reduction_ratio(s.prompt, sc) for s, sc in zip(samples, v15_schemas)]
    trr_sorted = sorted(trr)
    trr_mean = sum(trr) / len(trr) if trr else 0.0
    trr_p50 = trr_sorted[len(trr_sorted) // 2] if trr_sorted else 0.0

    preds = [1.0 if sc.complexity == "heavy" else 0.0 for sc in v15_schemas]
    labels = [1 if s.expected_complexity == "heavy" else 0 for s in samples]
    brier = brier_score(preds, labels)

    return {
        "n": len(samples),
        "v1_success_rate": round(v1_success["rate"], 4),
        "v15_success_rate": round(v15_success["rate"], 4),
        "delta": round(v15_success["rate"] - v1_success["rate"], 4),
        "token_reduction_ratio_mean": round(trr_mean, 4),
        "token_reduction_ratio_p50": round(trr_p50, 4),
        "classifier_brier_score": round(brier, 4),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval.eval_v15", description="V1 vs V1.5 eval harness")
    p.add_argument("--dataset", default="canned", help="'canned' or path to a .jsonl file")
    p.add_argument("--mode", default="mock", choices=["mock", "http"])
    p.add_argument("--judge", default="mock", choices=["mock", "llm"])
    p.add_argument("--router-base", default="http://localhost:8080")
    p.add_argument("--judge-base", default="http://localhost:4000/v1")
    p.add_argument("--judge-model", default="claude-haiku-4-5")
    p.add_argument("--out", default=None, help="optional path to write the JSON result")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    result = asyncio.run(run_eval(args))
    blob = json.dumps(result, indent=2, sort_keys=True)
    print(blob)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Keep backwards-compatible re-export so any callers importing the old struct
# continue to work without a rename. The new EvalSample is the recommended type.
EvalRow = EvalSample  # type: ignore[misc]


# ---------------------------------------------------------------------------
# defaults exposed for tests / external callers
# ---------------------------------------------------------------------------

__all__ = [
    "CANNED_DATASET_PATH",
    "EvalArgs",
    "EvalSample",
    "MockJudge",
    "HttpJudge",
    "build_judge_client",
    "brier_score",
    "load_dataset",
    "run_eval",
    "task_success_rate",
    "token_reduction_ratio",
]


# preserve a tiny eval-row dataclass field default sentinel for any old caller
_field = field  # noqa: F841
