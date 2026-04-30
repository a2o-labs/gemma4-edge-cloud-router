"""LRU cache for ``EdgeEncoder.encode`` — hit/miss, eviction, disable.

Inference-only cache: identical prompt → identical vector, fresh task_id.
``cache_size=0`` must hard-disable the cache so training-time forward
passes see no stale state. The training-loop test in
``tests/test_train_v15.py`` covers the gradient-path side.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from router.edge_encoder import EdgeEncoder  # noqa: E402

TINY_MODEL = "trl-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def encoder() -> EdgeEncoder:
    return EdgeEncoder(
        model_name=TINY_MODEL,
        embedding_dim=64,
        device="cpu",
        dtype=torch.float32,
        cache_size=8,
    )


def test_cache_hit_on_repeated_prompt(encoder: EdgeEncoder) -> None:
    encoder.clear_cache()
    schema1, vec1 = encoder.encode("Hello world")
    schema2, vec2 = encoder.encode("Hello world")
    np.testing.assert_array_equal(vec1, vec2)
    stats = encoder.cache_stats()
    assert stats is not None
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_cache_distinct_prompt_is_miss(encoder: EdgeEncoder) -> None:
    encoder.clear_cache()
    encoder.encode("Hello world")
    encoder.encode("Goodbye world")
    stats = encoder.cache_stats()
    assert stats is not None
    assert stats["hits"] == 0
    assert stats["misses"] == 2


def test_cache_lru_evicts_oldest(encoder: EdgeEncoder) -> None:
    encoder.clear_cache()
    for i in range(9):
        encoder.encode(f"prompt-{i}")
    stats = encoder.cache_stats()
    assert stats is not None
    assert stats["size"] == 8
    assert stats["misses"] == 9
    assert stats["hits"] == 0


def test_cache_size_zero_disables_cache() -> None:
    enc = EdgeEncoder(
        model_name=TINY_MODEL,
        embedding_dim=64,
        device="cpu",
        dtype=torch.float32,
        cache_size=0,
    )
    assert enc.cache_stats() is None
    enc.encode("anything")
    enc.encode("anything")
    assert enc.cache_stats() is None
    enc.clear_cache()


def test_task_id_differs_on_cache_hit(encoder: EdgeEncoder) -> None:
    encoder.clear_cache()
    s1, _ = encoder.encode("same prompt")
    s2, _ = encoder.encode("same prompt")
    assert s1 is not None and s2 is not None
    assert s1.task_id != s2.task_id
