"""CLI smoke tests in mock mode."""

from __future__ import annotations

import os

os.environ["V15_ENABLED"] = "true"
os.environ["V15_MOCK_MODE"] = "true"
os.environ["V15_DEVICE"] = "cpu"

import json

import router.config

router.config._settings = None


def test_cli_health_prints_settings(capsys):
    from router.cli import main

    rc = main(["health"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["v15_enabled"] is True
    assert parsed["mock_mode"] is True
    assert parsed["device"] == "cpu"


def test_cli_encode_emits_schema_v15_json(capsys):
    from router.cli import main

    rc = main(["encode", "Hello world"])
    assert rc == 0
    out = capsys.readouterr().out
    schema = json.loads(out)
    assert schema["version"] == "1.5"
    assert schema["complexity"] in {"light", "heavy"}
    assert schema["embedding_dim"] == 4096


def test_cli_encode_no_embedding_strips_b64(capsys):
    from router.cli import main

    rc = main(["encode", "Hello", "--no-embedding"])
    assert rc == 0
    schema = json.loads(capsys.readouterr().out)
    assert "embedding_b64" not in schema or schema.get("embedding_b64") is None


def test_cli_pipeline_v15_returns_mock_answer(capsys):
    from router.cli import main

    rc = main(["pipeline-v15", "Hello"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == "1.5"
    assert "[mock-v15-response" in out["answer"]
    assert out["embedding_dim"] == 4096
