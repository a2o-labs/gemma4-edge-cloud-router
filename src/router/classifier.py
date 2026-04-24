"""Light/heavy task classifier backed by the edge LLM.

The classifier expects strictly JSON-parseable output from the local model.
On any parse failure, we fall back to the configured value (default: heavy) so
ambiguous cases are never silently answered by a weak local model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ClassifierSettings

log = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a task complexity classifier. Classify the user request as either "
    '"light" (a local small model can answer in under 2s) or "heavy" (needs a '
    "frontier cloud model). Respond with a single JSON object, no prose:\n"
    '{"task_type": "qa|code|chat|reasoning|other", '
    '"complexity": "light|heavy", "confidence": 0.0-1.0}'
)


@dataclass
class ClassifierResult:
    task_type: str
    complexity: str  # "light" | "heavy"
    confidence: float
    raw: str
    fell_back: bool = False


class ChatClient(Protocol):
    async def chat(self, system: str, user: str) -> str: ...


class Classifier:
    def __init__(self, client: ChatClient, settings: ClassifierSettings) -> None:
        self.client = client
        self.settings = settings

    async def classify(self, prompt: str) -> ClassifierResult:
        try:
            raw = await self.client.chat(CLASSIFIER_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001 — edge LLM unreachable → fallback
            log.warning("classifier edge call failed: %s", exc)
            return self._fallback(raw="", reason=f"edge_error:{exc}")

        parsed = _parse_json(raw)
        if parsed is None:
            return self._fallback(raw=raw, reason="parse_fail")

        complexity = parsed.get("complexity")
        if complexity not in {"light", "heavy"}:
            return self._fallback(raw=raw, reason="bad_complexity")

        confidence = float(parsed.get("confidence", 0.0))
        if complexity == "light" and confidence < self.settings.confidence_threshold:
            log.info("confidence %.2f below threshold — demoting to heavy", confidence)
            complexity = "heavy"

        return ClassifierResult(
            task_type=str(parsed.get("task_type", "other")),
            complexity=complexity,
            confidence=confidence,
            raw=raw,
        )

    def _fallback(self, *, raw: str, reason: str) -> ClassifierResult:
        log.info("classifier fallback (%s) → %s", reason, self.settings.fallback)
        return ClassifierResult(
            task_type="other",
            complexity=self.settings.fallback,
            confidence=0.0,
            raw=raw,
            fell_back=True,
        )


def _parse_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
