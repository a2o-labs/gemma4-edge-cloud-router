"""Chunked encoding tests for EdgeEncoder.encode_chunked."""

from __future__ import annotations

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
        cache_size=0,
    )


def test_short_text_returns_single_chunk(encoder):
    """Text below chunk_size should yield one chunk."""
    schemas, vecs = encoder.encode_chunked("Hello world", chunk_size=4096)
    assert len(schemas) == 1
    assert len(vecs) == 1
    assert vecs[0].shape == (64,)


def test_long_text_returns_multiple_chunks(encoder):
    """Force chunking by setting tiny chunk_size."""
    text = " ".join(f"word{i}" for i in range(200))
    schemas, vecs = encoder.encode_chunked(text, chunk_size=32, overlap=4)
    assert len(schemas) >= 2
    assert len(vecs) == len(schemas)
    for v in vecs:
        assert v.shape == (64,)


def test_each_chunk_schema_has_v15_version(encoder):
    text = " ".join(f"word{i}" for i in range(100))
    schemas, _ = encoder.encode_chunked(text, chunk_size=32, overlap=4)
    for s in schemas:
        assert s.version == "1.5"
        assert s.embedding_dim == 64


def test_overlap_must_be_less_than_chunk_size(encoder):
    with pytest.raises(ValueError):
        encoder.encode_chunked("test", chunk_size=10, overlap=10)


def test_return_schema_false_yields_only_vecs(encoder):
    text = " ".join(f"w{i}" for i in range(80))
    schemas, vecs = encoder.encode_chunked(
        text, chunk_size=20, overlap=2, return_schema=False
    )
    assert all(s is None for s in schemas)
    assert all(v.shape == (64,) for v in vecs)
