"""Tests for the V1.0/V1.5 adaptive routing decision."""
from __future__ import annotations

from router.adaptive_router import decide_path, make_token_count_fn


def test_routes_to_v1_when_pipeline_not_ready() -> None:
    d = decide_path("anything", v15_ready=False, threshold=92)
    assert d.path == "v1"
    assert d.reason == "v15-pipeline-not-ready"
    assert d.v15_ready is False


def test_routes_to_v1_when_prompt_below_threshold() -> None:
    d = decide_path(
        "What is 2+2?",
        token_count_fn=lambda s: 7,
        threshold=92,
        v15_ready=True,
    )
    assert d.path == "v1"
    assert d.prompt_tokens == 7
    assert d.reason == "prompt-below-threshold"


def test_routes_to_v15_when_prompt_above_threshold() -> None:
    long_prompt = "summarize the following: " + "context " * 200
    d = decide_path(
        long_prompt,
        token_count_fn=lambda s: 250,
        threshold=92,
        v15_ready=True,
    )
    assert d.path == "v15"
    assert d.prompt_tokens == 250
    assert d.reason == "prompt-exceeds-threshold"


def test_at_threshold_stays_on_v1() -> None:
    """Boundary: a prompt exactly at the threshold stays on V1.0 to avoid
    flapping; V1.5 only kicks in when prompt is strictly larger."""
    d = decide_path(
        "x", token_count_fn=lambda s: 92, threshold=92, v15_ready=True
    )
    assert d.path == "v1"


def test_token_count_fallback_when_callable_fails() -> None:
    def explode(_s: str) -> int:
        raise RuntimeError("tokenizer crashed")

    long_prompt = "x" * 500  # chars/4 fallback => 125 tokens
    d = decide_path(long_prompt, token_count_fn=explode, threshold=92, v15_ready=True)
    assert d.path == "v15"
    assert d.prompt_tokens == 125


def test_chars_over_4_fallback_when_no_tokenizer_supplied() -> None:
    long_prompt = "y" * 500
    d = decide_path(long_prompt, token_count_fn=None, threshold=92, v15_ready=True)
    assert d.path == "v15"
    assert d.prompt_tokens == 125


def test_make_token_count_fn_returns_none_when_pipeline_unloaded() -> None:
    class FakePipeline:
        def is_ready(self) -> bool:
            return False

    assert make_token_count_fn(FakePipeline()) is None


def test_make_token_count_fn_returns_none_when_no_tokenizer() -> None:
    class FakeAdapter:
        tokenizer = None

    class FakePipeline:
        _adapter = FakeAdapter()

        def is_ready(self) -> bool:
            return True

    assert make_token_count_fn(FakePipeline()) is None


def test_make_token_count_fn_uses_adapter_tokenizer_when_present() -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [0] * len(text.split())

    class FakeAdapter:
        tokenizer = FakeTokenizer()

    class FakePipeline:
        _adapter = FakeAdapter()

        def is_ready(self) -> bool:
            return True

    fn = make_token_count_fn(FakePipeline())
    assert fn is not None
    assert fn("hello world from tokenizer") == 4
