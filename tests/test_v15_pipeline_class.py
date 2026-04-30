"""Direct unit tests for V15Pipeline (no FastAPI involved)."""

from __future__ import annotations


def test_v15_pipeline_mock_load_and_run():
    from router.v15_pipeline import V15Pipeline, V15Settings

    p = V15Pipeline(V15Settings(enabled=True, mock_mode=True, device="cpu"))
    p.load()
    assert p.is_ready()
    schema, answer = p.run("Hello world")
    assert schema.version == "1.5"
    assert schema.embedding_dim == 4096
    assert "[mock-v15-response" in answer


def test_v15_pipeline_disabled_load_is_noop():
    from router.v15_pipeline import V15Pipeline, V15Settings

    p = V15Pipeline(V15Settings(enabled=False))
    p.load()
    assert not p.is_ready()
