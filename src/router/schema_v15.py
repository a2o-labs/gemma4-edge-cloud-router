"""CompactSchema V1.5: V1.0 fields + edge-encoder embedding bridge."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CompactSchemaV15(BaseModel):
    version: Literal["1.5"] = "1.5"
    task_id: str
    task_type: Literal[
        "summarize", "qa", "code", "translate", "search", "reason", "other"
    ]
    complexity: Literal["light", "heavy"]
    semantic_tags: List[str] = []
    action_graph: Optional[Dict[str, Any]] = None
    symbol_packet: Optional[Dict[str, Any]] = None
    privacy: Optional[Dict[str, Any]] = None
    max_tokens_hint: int = 512

    embedding_b64: Optional[str] = Field(
        None,
        description="base64-encoded float16 4096-d vector from edge model last hidden state",
    )
    embedding_dim: int = 4096
    embedding_dtype: Literal["float16", "float32"] = "float16"
    confidence: float = 0.0
