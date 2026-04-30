"""Batch encoding + adapter forward tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")  # noqa: F841


@pytest.fixture
def encoder():
    from router.edge_encoder import EdgeEncoder

    return EdgeEncoder(
        model_name="trl-internal-testing/tiny-random-LlamaForCausalLM",
        embedding_dim=64,
        device="cpu",
        dtype=torch.float32,
        cache_size=8,
    )


def test_encode_batch_size_1_matches_encode(encoder):
    """Single-item batch must match plain encode()."""
    encoder.clear_cache()
    s_single, v_single = encoder.encode("Hello")
    encoder.clear_cache()
    schemas, vecs = encoder.encode_batch(["Hello"])
    assert schemas[0].embedding_dim == s_single.embedding_dim
    np.testing.assert_allclose(vecs[0], v_single, rtol=1e-3)


def test_encode_batch_returns_correct_length(encoder):
    encoder.clear_cache()
    schemas, vecs = encoder.encode_batch(["a", "b", "c"])
    assert len(schemas) == 3
    assert len(vecs) == 3
    assert all(v.shape == (64,) for v in vecs)


def test_encode_batch_uses_cache_for_repeats(encoder):
    encoder.clear_cache()
    encoder.encode("first")
    schemas, vecs = encoder.encode_batch(["first", "second", "third"])
    stats = encoder.cache_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 2


def test_encode_batch_results_are_valid_schemas(encoder):
    encoder.clear_cache()
    schemas, _ = encoder.encode_batch(["x", "y"])
    for s in schemas:
        assert s.version == "1.5"
        assert s.embedding_dim == 64


@pytest.fixture
def adapter():
    from router.cloud_adapter import SoftPromptAdapter

    return SoftPromptAdapter(
        edge_dim=64,
        prompt_tokens=4,
        cloud_model_name="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device="cpu",
        dtype=torch.float32,
    )


def test_forward_batch_returns_list_of_strings(adapter):
    vecs = torch.randn(3, 64)
    out = adapter.forward_batch(
        vecs, ['{"a":1}', '{"a":2}', '{"a":3}'], max_new_tokens=4
    )
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(s, str) for s in out)


def test_forward_batch_size_mismatch_raises(adapter):
    vecs = torch.randn(2, 64)
    with pytest.raises(ValueError):
        adapter.forward_batch(vecs, ['{"a":1}'], max_new_tokens=4)
