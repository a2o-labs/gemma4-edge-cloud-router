"""CompactSchemaV15 round-trip + embedding-bridge invariants.

Covers wire-format guarantees the edge encoder must satisfy and the cloud
adapter must rely on:
  * version literal pinned to "1.5"
  * embedding_b64 round-trips through base64 to a float16 numpy array of
    embedding_dim length
  * float32 fallback works (training-time / dev mode)
  * task_type / complexity enums are enforced
  * embedding-less envelope (V1-style) still validates (graceful degradation)
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from pydantic import ValidationError

from router.schema_v15 import CompactSchemaV15


def _encode_vec(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_vec(b64: str, dim: int, dtype: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    np_dtype = {"float16": np.float16, "float32": np.float32}[dtype]
    arr = np.frombuffer(raw, dtype=np_dtype)
    assert arr.shape == (dim,), f"decoded shape {arr.shape} != ({dim},)"
    return arr


def _baseline_envelope(**overrides):
    base = dict(
        task_id="abc123",
        task_type="qa",
        complexity="heavy",
    )
    base.update(overrides)
    return base


def test_v15_minimal_envelope():
    e = CompactSchemaV15(**_baseline_envelope())
    assert e.version == "1.5"
    assert e.embedding_b64 is None
    assert e.embedding_dim == 4096
    assert e.embedding_dtype == "float16"


def test_v15_embedding_round_trip_float16():
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(4096).astype(np.float16)
    e = CompactSchemaV15(
        **_baseline_envelope(embedding_b64=_encode_vec(vec))
    )

    wire = e.model_dump_json()
    back = CompactSchemaV15.model_validate_json(wire)
    assert back.embedding_b64 == e.embedding_b64

    decoded = _decode_vec(back.embedding_b64, back.embedding_dim, back.embedding_dtype)
    assert decoded.shape == (4096,)
    np.testing.assert_array_equal(decoded, vec)


def test_v15_embedding_round_trip_float32():
    rng = np.random.default_rng(seed=43)
    vec = rng.standard_normal(4096).astype(np.float32)
    e = CompactSchemaV15(
        **_baseline_envelope(
            embedding_b64=_encode_vec(vec),
            embedding_dtype="float32",
        )
    )
    decoded = _decode_vec(e.embedding_b64, e.embedding_dim, e.embedding_dtype)
    np.testing.assert_array_equal(decoded, vec)


def test_v15_embedding_dim_override():
    """Edge encoder may use a non-default projection dim (e.g. 2048)."""
    vec = np.zeros(2048, dtype=np.float16)
    e = CompactSchemaV15(
        **_baseline_envelope(
            embedding_b64=_encode_vec(vec),
            embedding_dim=2048,
        )
    )
    decoded = _decode_vec(e.embedding_b64, e.embedding_dim, e.embedding_dtype)
    assert decoded.shape == (2048,)


def test_v15_rejects_invalid_complexity():
    with pytest.raises(ValidationError):
        CompactSchemaV15(**_baseline_envelope(complexity="medium"))


def test_v15_rejects_invalid_dtype():
    with pytest.raises(ValidationError):
        CompactSchemaV15(
            **_baseline_envelope(embedding_dtype="float64")
        )


def test_v15_envelope_without_embedding_validates():
    """V1-compat path: cloud adapter falls back to JSON-only."""
    e = CompactSchemaV15(
        **_baseline_envelope(
            action_graph={"nodes": [{"id": "n1", "op": "summarize"}]},
            semantic_tags=["finance"],
        )
    )
    wire = e.model_dump_json()
    back = CompactSchemaV15.model_validate_json(wire)
    assert back.embedding_b64 is None
    assert back.action_graph == {"nodes": [{"id": "n1", "op": "summarize"}]}


def test_v15_version_pinned_to_1_5():
    """Schema bump must be intentional; version literal blocks accidental drift."""
    with pytest.raises(ValidationError):
        CompactSchemaV15(
            **_baseline_envelope(),
            **{"version": "2.0"},  # type: ignore[arg-type]
        )


def test_v15_action_graph_passthrough():
    """action_graph is opaque dict — preserve nested structure across round-trip."""
    graph = {
        "nodes": [
            {"id": "n1", "op": "retrieve", "args": {"query": "@symbol_0"}},
            {"id": "n2", "op": "summarize", "args": {"length": "short"}},
        ],
        "edges": [{"from": "n1", "to": "n2"}],
    }
    e = CompactSchemaV15(**_baseline_envelope(action_graph=graph))
    back = CompactSchemaV15.model_validate_json(e.model_dump_json())
    assert back.action_graph == graph
