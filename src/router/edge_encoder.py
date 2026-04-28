"""Edge encoder: Gemma4-26B-A4B Q4 frozen base + linear projection head.

V1.5 stub. Real model loading is intentionally elided; the hooks expose the
shape and contract that downstream training and serving code must respect.
"""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .schema_v15 import CompactSchemaV15


@dataclass
class EdgeEncoderConfig:
    base_model_id: str = "google/gemma-4-26b-a4b"
    quantization: str = "Q4_K_M"
    embedding_dim: int = 4096
    proj_in_dim: int = 4096
    proj_out_dim: int = 4096
    device: str = "cuda:0"
    dtype: str = "float16"


class EdgeEncoder:
    """Frozen Gemma4-26B-A4B last-hidden-state -> linear projection head.

    encode(prompt) returns (CompactSchemaV15, np.ndarray[float16, 4096]).
    The schema is populated by the existing V1 classifier path; the embedding
    is the projection-head output.
    """

    def __init__(self, cfg: EdgeEncoderConfig | None = None) -> None:
        self.cfg = cfg or EdgeEncoderConfig()
        self._model = None
        self._tokenizer = None
        self._proj = None
        print("TODO real model: EdgeEncoder.__init__ has not loaded weights")

    def load(self) -> None:
        # TODO: from transformers import AutoModelForCausalLM, AutoTokenizer
        # TODO: self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model_id)
        # TODO: self._model = AutoModelForCausalLM.from_pretrained(... quantization_config=...)
        # TODO: freeze base; init nn.Linear(proj_in_dim, proj_out_dim)
        print("TODO real model: EdgeEncoder.load skipped")

    def _last_hidden(self, prompt: str) -> np.ndarray:
        # TODO: run model forward, collect last token hidden state
        return np.zeros(self.cfg.embedding_dim, dtype=np.float16)

    def _project(self, hidden: np.ndarray) -> np.ndarray:
        # TODO: apply trained linear projection head
        return hidden.astype(np.float16)

    def _classify(self, prompt: str) -> CompactSchemaV15:
        # Stub: real classifier will be wired to existing router.classifier.
        return CompactSchemaV15(
            task_id=str(uuid.uuid4()),
            task_type="other",
            complexity="light",
            semantic_tags=[],
            confidence=0.0,
            embedding_dim=self.cfg.embedding_dim,
            embedding_dtype="float16",
        )

    def encode(self, prompt: str) -> Tuple[CompactSchemaV15, np.ndarray]:
        schema = self._classify(prompt)
        vec = self._project(self._last_hidden(prompt))
        schema.embedding_b64 = base64.b64encode(vec.tobytes()).decode("ascii")
        return schema, vec
