"""Tests for the V1.5 eval harness.

All tests must run on CPU in mock mode - no GPU, no HF weights, no real LLM.
"""

from __future__ import annotations

import os

os.environ.setdefault("V15_ENABLED", "true")
os.environ.setdefault("V15_MOCK_MODE", "true")
os.environ.setdefault("V15_DEVICE", "cpu")

import pytest

from eval.eval_v15 import (
    EvalArgs,
    brier_score,
    load_dataset,
    run_eval,
    token_reduction_ratio,
)
from router.schema_v15 import CompactSchemaV15


def test_token_reduction_ratio_basic():
    schema = CompactSchemaV15(
        task_id="t1",
        task_type="qa",
        complexity="light",
        embedding_b64="x" * 100,
        embedding_dim=4096,
    )
    long_prompt = "hello " * 200
    ratio = token_reduction_ratio(long_prompt, schema)
    # exact value depends on tokenizer; for a long prompt we expect positive
    # compression, but the contract is just that the value is bounded in [0,1].
    assert 0.0 <= ratio <= 1.0


def test_brier_score_perfect():
    preds = [1.0, 0.0, 1.0, 0.0]
    labels = [1, 0, 1, 0]
    assert brier_score(preds, labels) == 0.0


def test_brier_score_inverted():
    preds = [0.0, 1.0, 0.0, 1.0]
    labels = [1, 0, 1, 0]
    assert brier_score(preds, labels) == 1.0


def test_load_dataset_canned():
    samples = load_dataset("canned")
    assert len(samples) >= 5
    assert all(hasattr(s, "prompt") for s in samples)
    assert all(s.expected_complexity in {"light", "heavy"} for s in samples)


@pytest.mark.asyncio
async def test_run_eval_mock_end_to_end():
    """Smoke: full eval pipeline in mock mode without HTTP or real judge."""
    args = EvalArgs(mode="mock", dataset="canned", judge="mock")
    result = await run_eval(args)
    assert "v1_success_rate" in result
    assert "v15_success_rate" in result
    assert "token_reduction_ratio_mean" in result
    assert 0.0 <= result["classifier_brier_score"] <= 1.0
    assert result["n"] >= 5
