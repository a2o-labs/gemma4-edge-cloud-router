"""V1.5 pipeline orchestrator: edge_encoder -> wire schema_v15 -> cloud_adapter.

Wraps EdgeEncoder + SoftPromptAdapter into a single callable that takes a raw
prompt and returns a generated string. Lazy-imports torch / transformers via
the underlying modules so the FastAPI app stays importable without GPU deps.

Heavy initialization (model loads) happens once at startup via .load(). If
V15_ENABLED env var is false (default), .load() is a no-op and .run() raises.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import V15Settings
from .schema_v15 import CompactSchemaV15
from .telemetry import get_tracer

log = logging.getLogger("v15-pipeline")

__all__ = ["V15Pipeline", "V15Settings"]


class _MockEncoder:
    """Stub encoder: returns a fake embedding + canned schema."""

    def __init__(self, embedding_dim: int) -> None:
        import numpy as np

        self.embedding_dim = embedding_dim
        self._np = np

    def encode(
        self, text: str, return_schema: bool = True
    ) -> tuple[CompactSchemaV15 | None, Any]:
        import base64
        import uuid

        vec = self._np.zeros(self.embedding_dim, dtype=self._np.float16)
        if not return_schema:
            return None, vec
        b64 = base64.b64encode(vec.tobytes()).decode("ascii")
        schema = CompactSchemaV15(
            task_id=str(uuid.uuid4()),
            task_type="other",
            complexity="light" if len(text) < 200 else "heavy",
            embedding_b64=b64,
            embedding_dim=self.embedding_dim,
            embedding_dtype="float16",
        )
        return schema, vec


class _MockAdapter:
    """Stub adapter: returns canned response."""

    def forward(self, edge_vec: Any, json_text: str, max_new_tokens: int = 256) -> str:
        return f"[mock-v15-response prompt_chars={len(json_text)}]"


class V15Pipeline:
    """End-to-end V1.5 orchestrator."""

    def __init__(self, settings: V15Settings) -> None:
        self.settings = settings
        self._encoder: Any = None
        self._adapter: Any = None
        self._loaded = False

    def load(self) -> None:
        if not self.settings.enabled:
            log.info("V1.5 disabled - pipeline.load() no-op")
            return
        if self.settings.mock_mode:
            self._encoder = _MockEncoder(self.settings.embedding_dim)
            self._adapter = _MockAdapter()
            self._loaded = True
            log.info("V1.5 pipeline loaded (mock mode)")
            return
        from .cloud_adapter import SoftPromptAdapter
        from .edge_encoder import EdgeEncoder

        self._encoder = EdgeEncoder(
            model_name=self.settings.edge_model_name,
            embedding_dim=self.settings.embedding_dim,
            hf_token=self.settings.hf_token,
            device=self.settings.device,
        )
        self._adapter = SoftPromptAdapter(
            edge_dim=self.settings.embedding_dim,
            prompt_tokens=self.settings.prompt_tokens,
            hf_token=self.settings.hf_token,
            cloud_model_name=self.settings.cloud_model_name,
            device=self.settings.device,
        )
        self._loaded = True
        log.info("V1.5 pipeline loaded (real mode)")

    def is_ready(self) -> bool:
        return self._loaded

    def run(self, prompt: str) -> tuple[CompactSchemaV15, str]:
        if not self._loaded:
            raise RuntimeError(
                "V1.5 pipeline not loaded. Set V15_ENABLED=true and call .load() at startup."
            )
        tracer = get_tracer()
        with tracer.start_as_current_span("v15.encode") as span:
            schema, vec = self._encoder.encode(prompt, return_schema=True)
            span.set_attribute("complexity", schema.complexity)
            span.set_attribute("embedding_dim", schema.embedding_dim)
        with tracer.start_as_current_span("v15.forward") as span:
            json_text = schema.model_dump_json()
            response = self._adapter.forward(
                vec, json_text, max_new_tokens=self.settings.max_new_tokens
            )
            span.set_attribute("max_new_tokens", self.settings.max_new_tokens)
            span.set_attribute("response_chars", len(response))
        return schema, response
