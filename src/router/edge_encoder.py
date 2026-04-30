"""Edge encoder: frozen Gemma 4 base + trainable Linear projection head.

Wraps a causal LM, extracts the last-token hidden state of the last layer,
projects it to ``embedding_dim``, and packages the vector into a
``CompactSchemaV15`` envelope.

Heavy deps (torch, transformers) are lazy-imported inside methods so that
plain ``import router.edge_encoder`` does not pull them for callers that
only need the schema or config.
"""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Tuple

import numpy as np

from .schema_v15 import CompactSchemaV15

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@dataclass
class EdgeEncoderConfig:
    """Backwards-compat config object retained from the V1.5 stub."""

    base_model_id: str = "google/gemma-4-26B-A4B-it"
    quantization: str = "Q4_K_M"
    embedding_dim: int = 4096
    proj_in_dim: int = 4096
    proj_out_dim: int = 4096
    device: str = "cuda:0"
    dtype: str = "float16"


def _load_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(
            "edge_encoder requires torch + transformers. Install the "
            "'training' extra (uv pip install -e '.[training]') or deploy "
            "on the L4 inference image with the training stack baked in."
        ) from e
    return torch, nn


def _load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "edge_encoder requires transformers>=4.45. Install the 'training' "
            "extra; on L4 deployment images this is preinstalled."
        ) from e
    return AutoModelForCausalLM, AutoTokenizer


class EdgeEncoder:
    """Frozen base LM + trainable nn.Linear projection head.

    encode(prompt) returns (CompactSchemaV15, np.ndarray[float16, embedding_dim]).
    The base model is frozen; only ``self.projection`` is trainable.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-26B-A4B-it",
        embedding_dim: int = 4096,
        hf_token: str | None = None,
        device: str = "cuda",
        dtype: "Any" = None,
    ) -> None:
        torch, nn = _load_torch()
        AutoModelForCausalLM, AutoTokenizer = _load_transformers()

        if dtype is None:
            dtype = torch.float16

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=hf_token
            )
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=dtype,
                output_hidden_states=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load base model '{model_name}'. Hint: real Gemma "
                f"4 weights require the L4 24GB inference deployment with "
                f"HF_TOKEN configured. Underlying error: {e}"
            ) from e

        try:
            base = base.to(device)
        except Exception:
            pass

        for p in base.parameters():
            p.requires_grad_(False)
        base.eval()
        self.base = base

        base_hidden = base.config.hidden_size
        self.base_hidden = base_hidden
        self.projection = nn.Linear(base_hidden, embedding_dim).to(device=device, dtype=dtype)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def trainable_parameters(self) -> Iterator["nn.Parameter"]:
        return self.projection.parameters()

    def _tokenize(self, text: str):
        torch, _ = _load_torch()
        if getattr(self.tokenizer, "chat_template", None):
            out = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            # transformers 5.x returns a BatchEncoding (dict-like) from
            # apply_chat_template even when return_tensors="pt"; older 4.x with
            # return_dict=True also returns a dict; some 4.x paths return a raw
            # Tensor. Normalize all three to a Tensor of input_ids.
            if torch.is_tensor(out):
                ids = out
            elif isinstance(out, dict):
                ids = out["input_ids"]
            else:
                ids = out.input_ids
        else:
            ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    def _last_hidden(self, text: str) -> "torch.Tensor":
        torch, _ = _load_torch()
        input_ids = self._tokenize(text)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        last_layer = out.hidden_states[-1]
        return last_layer[:, -1, :].squeeze(0)

    def classify(self, text: str) -> Literal["light", "heavy"]:
        """Length-based placeholder classifier.

        Production classifier lives in external-pipeline data-harvest (TL's domain);
        this is a deterministic stand-in so the schema envelope can be
        populated without a separate model load.
        """
        if len(text) < 200 and "```" not in text and "\n\n\n" not in text:
            return "light"
        return "heavy"

    def encode(
        self, text: str, return_schema: bool = True
    ) -> Tuple[CompactSchemaV15 | None, np.ndarray]:
        torch, _ = _load_torch()
        hidden = self._last_hidden(text)
        proj = self.projection(hidden.to(dtype=self.projection.weight.dtype))
        np_dtype = np.float16 if self.dtype == torch.float16 else np.float32
        vec = proj.detach().to("cpu").to(torch.float32).numpy().astype(np_dtype)

        if not return_schema:
            return None, vec

        complexity = self.classify(text)
        dtype_label = "float16" if np_dtype == np.float16 else "float32"
        schema = CompactSchemaV15(
            task_id=str(uuid.uuid4()),
            task_type="other",
            complexity=complexity,
            semantic_tags=[],
            confidence=0.0,
            embedding_dim=self.embedding_dim,
            embedding_dtype=dtype_label,
            embedding_b64=base64.b64encode(vec.tobytes()).decode("ascii"),
        )
        return schema, vec


def decode_embedding_from_schema(schema: CompactSchemaV15) -> np.ndarray:
    """Inverse of ``EdgeEncoder.encode``'s base64 packing."""
    if schema.embedding_b64 is None:
        raise ValueError("schema.embedding_b64 is None")
    raw = base64.b64decode(schema.embedding_b64.encode("ascii"))
    np_dtype = {"float16": np.float16, "float32": np.float32}[schema.embedding_dtype]
    arr = np.frombuffer(raw, dtype=np_dtype)
    if arr.shape != (schema.embedding_dim,):
        raise ValueError(
            f"decoded shape {arr.shape} != ({schema.embedding_dim},)"
        )
    return arr
