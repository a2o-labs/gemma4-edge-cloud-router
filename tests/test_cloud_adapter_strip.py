"""Tests for the schema-strip helper in cloud_adapter."""
from __future__ import annotations

import json

from router.cloud_adapter import _strip_schema_embedding


def test_strips_embedding_b64_field() -> None:
    schema = {
        "version": "1.5",
        "task_id": "abc",
        "task_type": "qa",
        "complexity": "light",
        "embedding_b64": "A" * 3072,
        "embedding_dim": 1152,
    }
    out = _strip_schema_embedding(json.dumps(schema))
    parsed = json.loads(out)
    assert "embedding_b64" not in parsed
    assert parsed["task_id"] == "abc"
    assert parsed["embedding_dim"] == 1152
    # The whole point: dramatic size reduction.
    assert len(out) < 200


def test_passthrough_when_field_absent() -> None:
    schema = {"version": "1.5", "task_id": "x", "task_type": "qa", "complexity": "light"}
    text = json.dumps(schema)
    assert _strip_schema_embedding(text) == text


def test_passthrough_for_non_json_text() -> None:
    plain = "Hello, world!"
    assert _strip_schema_embedding(plain) == plain


def test_passthrough_for_json_array() -> None:
    arr = json.dumps([1, 2, 3])
    assert _strip_schema_embedding(arr) == arr
