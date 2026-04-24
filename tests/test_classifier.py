"""Classifier tests with a mocked edge LLM chat client."""

from __future__ import annotations

import pytest

from router.classifier import Classifier
from router.config import ClassifierSettings


class FakeClient:
    def __init__(self, reply: str, *, raise_exc: Exception | None = None) -> None:
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, str]] = []

    async def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.raise_exc:
            raise self.raise_exc
        return self.reply


@pytest.mark.asyncio
async def test_classifier_parses_heavy():
    fc = FakeClient('{"task_type": "reasoning", "complexity": "heavy", "confidence": 0.9}')
    c = Classifier(fc, ClassifierSettings())
    r = await c.classify("prove the halting problem")
    assert r.complexity == "heavy"
    assert r.task_type == "reasoning"
    assert r.confidence == pytest.approx(0.9)
    assert r.fell_back is False


@pytest.mark.asyncio
async def test_classifier_parses_light_with_code_fences():
    raw = '```json\n{"task_type":"qa","complexity":"light","confidence":0.8}\n```'
    c = Classifier(FakeClient(raw), ClassifierSettings())
    r = await c.classify("what is 2+2")
    assert r.complexity == "light"


@pytest.mark.asyncio
async def test_classifier_low_confidence_demotes_to_heavy():
    fc = FakeClient('{"task_type": "qa", "complexity": "light", "confidence": 0.2}')
    c = Classifier(fc, ClassifierSettings(confidence_threshold=0.6))
    r = await c.classify("ambiguous ask")
    assert r.complexity == "heavy"


@pytest.mark.asyncio
async def test_classifier_falls_back_on_parse_failure():
    c = Classifier(FakeClient("not json at all"), ClassifierSettings(fallback="heavy"))
    r = await c.classify("anything")
    assert r.complexity == "heavy"
    assert r.fell_back is True


@pytest.mark.asyncio
async def test_classifier_falls_back_on_edge_error():
    c = Classifier(
        FakeClient("", raise_exc=RuntimeError("edge down")),
        ClassifierSettings(fallback="heavy"),
    )
    r = await c.classify("anything")
    assert r.complexity == "heavy"
    assert r.fell_back is True
