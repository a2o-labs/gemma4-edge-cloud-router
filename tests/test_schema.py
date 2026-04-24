"""CompactTask v1.0 round-trip tests."""

from __future__ import annotations

import json

from router.encoder import Encoder, MaskMap
from router.schema import CompactTask, SymbolPacket


def test_schema_json_round_trip():
    task = CompactTask(
        task_id="abc123",
        task_type="qa",
        instruction="hello @email_0",
        symbols=[SymbolPacket(symbol="@email_0", kind="pii", hint="email-like")],
        context_refs=["mem://foo"],
    )
    wire = task.model_dump_json()
    back = CompactTask.model_validate_json(wire)
    assert back == task
    assert back.schema_version == "1.0"


def test_schema_rejects_unknown_field():
    import pydantic

    payload = {
        "schema_version": "1.0",
        "task_id": "x",
        "task_type": "qa",
        "instruction": "hi",
        "unknown": True,
    }
    try:
        CompactTask.model_validate(payload)
    except pydantic.ValidationError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError("CompactTask should reject unknown fields")


def test_encoder_masks_and_unmasks():
    mask = MaskMap(ttl_seconds=60)
    enc = Encoder(mask)
    task = enc.encode(
        "email me at alice@example.com later",
        session_id="s1",
        task_type="qa",
    )
    assert "alice@example.com" not in task.instruction
    assert any(p.kind == "pii" for p in task.symbols)
    assert task.privacy.mask_applied is True

    # simulate cloud echoing the symbol back
    raw_cloud = json.dumps({"reply": f"ok, will email {task.symbols[0].symbol} soon"})
    restored = mask.unmask("s1", raw_cloud)
    assert "alice@example.com" in restored
