"""CompactTask v1.0 — wire schema between edge router and cloud LLM.

Fields frozen per §B.3 of the router design doc. Do not mutate without bumping
`schema_version`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class SymbolPacket(BaseModel):
    """Minimal typed reference passed alongside a CompactTask.

    The cloud side never sees raw PII — only the symbol id. The edge keeps a
    session-scoped mask_map that maps symbol -> original value.
    """

    symbol: str = Field(..., description="Opaque symbolic id, e.g. '@user_0'")
    kind: Literal["pii", "file", "url", "code", "other"] = "other"
    hint: str | None = Field(
        default=None,
        description="Non-sensitive hint (e.g. 'email-like', 'python file') to help the cloud model",
    )


class Privacy(BaseModel):
    """Privacy envelope. mask_applied=true means symbols replaced PII."""

    mask_applied: bool = False
    mask_version: str = "v1"
    redaction_level: Literal["none", "low", "medium", "high"] = "none"


class CompactTask(BaseModel):
    """Payload forwarded to the cloud LLM.

    Strict v1.0 shape. Extra fields rejected so drift is caught at parse time.
    """

    model_config = {"extra": "forbid"}

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    task_id: str
    task_type: Literal["qa", "code", "chat", "reasoning", "other"] = "other"
    instruction: str = Field(..., description="Masked user request")
    symbols: list[SymbolPacket] = Field(default_factory=list)
    context_refs: list[str] = Field(
        default_factory=list,
        description="Opaque refs (memory ids, file hashes) the cloud can resolve by callback",
    )
    privacy: Privacy = Field(default_factory=Privacy)
    meta: dict[str, Any] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    force: Literal["light", "heavy"] | None = None


class RouteResponse(BaseModel):
    task_id: str
    path: Literal["light", "heavy"]
    answer: str
    classifier_confidence: float | None = None
    latency_ms: float | None = None


class V15RouteRequest(BaseModel):
    prompt: str
    session_id: str | None = None


class V15RouteResponse(BaseModel):
    task_id: str
    path: Literal["v1.5"] = "v1.5"
    answer: str
    schema_version: str = "1.5"
    embedding_dim: int
    complexity: Literal["light", "heavy"]
    latency_ms: float | None = None
