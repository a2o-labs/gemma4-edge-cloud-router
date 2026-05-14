"""Adaptive V1.0 / V1.5 routing gate.

Crossover analysis (see docs/v15-1b-baseline-results.md and the 2026-05-08
cross-machine bench): the V1.5 IR adds a fixed wire cost of ~K + S tokens
where S is the stripped-schema text size (typically ~84 tokens of metadata
JSON) and K is the soft-prompt count. Below that crossover the V1.0
raw-prompt path is cheaper to send AND to attend over; above it the V1.5
path wins. This module decides per-request which path to dispatch.

Usage::

    decision = decide_path(
        prompt,
        token_count_fn=tok.encode_count,  # any callable[str -> int]
        threshold=92,
        v15_ready=True,
    )
    if decision == "v15":
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


Path = Literal["v1", "v15"]


@dataclass
class RouteDecision:
    path: Path
    prompt_tokens: int
    threshold: int
    v15_ready: bool
    reason: str


def decide_path(
    prompt: str,
    *,
    token_count_fn: Callable[[str], int] | None = None,
    threshold: int = 92,
    v15_ready: bool = False,
) -> RouteDecision:
    """Decide whether to dispatch a prompt over the V1.0 raw-prompt path or
    the V1.5 IR path.

    The decision rule is: V1.5 only when the V1.5 pipeline is loaded AND
    the prompt is long enough that the schema overhead is amortized. When
    ``token_count_fn`` is None we fall back to a chars/4 estimate, which
    is a coarse but stable proxy for tokenizer output across BPE-family
    models.
    """
    if not v15_ready:
        return RouteDecision(
            path="v1",
            prompt_tokens=-1,
            threshold=threshold,
            v15_ready=False,
            reason="v15-pipeline-not-ready",
        )
    if token_count_fn is not None:
        try:
            n = int(token_count_fn(prompt))
        except Exception:
            n = max(1, len(prompt) // 4)
    else:
        n = max(1, len(prompt) // 4)
    if n > threshold:
        return RouteDecision(
            path="v15",
            prompt_tokens=n,
            threshold=threshold,
            v15_ready=True,
            reason="prompt-exceeds-threshold",
        )
    return RouteDecision(
        path="v1",
        prompt_tokens=n,
        threshold=threshold,
        v15_ready=True,
        reason="prompt-below-threshold",
    )


def make_token_count_fn(pipeline) -> Callable[[str], int] | None:
    """Build a token-count callable from a loaded V15Pipeline.

    Returns ``None`` when the pipeline isn't loaded or doesn't expose a
    tokenizer; callers fall back to the chars/4 estimate.
    """
    if pipeline is None or not getattr(pipeline, "is_ready", lambda: False)():
        return None
    adapter = getattr(pipeline, "_adapter", None)
    tok = getattr(adapter, "tokenizer", None) if adapter is not None else None
    if tok is None:
        return None

    def _count(text: str) -> int:
        return len(tok.encode(text))

    return _count
