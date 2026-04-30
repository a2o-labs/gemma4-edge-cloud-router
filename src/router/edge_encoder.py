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
import hashlib
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Tuple

import numpy as np

from .schema_v15 import CompactSchemaV15

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class _EncodeCache:
    """Thread-safe LRU cache mapping sha256(prompt) -> (schema, vec).

    Inference-only. During training, set ``cache_size=0`` so each forward
    pass goes through the trainable projection layer with fresh autograd
    state.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._d: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str):
        with self._lock:
            k = self._key(text)
            if k in self._d:
                self._d.move_to_end(k)
                self.hits += 1
                return self._d[k]
            self.misses += 1
            return None

    def put(self, text: str, value) -> None:
        with self._lock:
            k = self._key(text)
            self._d[k] = value
            self._d.move_to_end(k)
            if len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._d),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / total if total else 0.0,
            }


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

    The optional inference-time LRU cache keyed on sha256(prompt) skips the
    forward pass when the same prompt is seen again. Pass ``cache_size=0``
    to disable — required during training so each step's gradient path
    runs the trainable projection layer afresh; or call ``clear_cache()``
    between optimizer steps.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-26B-A4B-it",
        embedding_dim: int = 4096,
        hf_token: str | None = None,
        device: str = "cuda",
        dtype: "Any" = None,
        quantization_config: "Any" = None,
        cache_size: int = 256,
    ) -> None:
        torch, nn = _load_torch()
        AutoModelForCausalLM, AutoTokenizer = _load_transformers()

        if dtype is None:
            dtype = torch.float16

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=hf_token
            )
            load_kwargs: dict[str, Any] = {
                "token": hf_token,
                "output_hidden_states": True,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                # device_map='auto' is required when using quantization_config;
                # bnb places weights itself, so don't also pass torch_dtype.
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = dtype
            base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load base model '{model_name}'. Hint: real Gemma "
                f"4 weights require the L4 24GB inference deployment with "
                f"HF_TOKEN configured. Underlying error: {e}"
            ) from e

        if quantization_config is None:
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
        if quantization_config is not None:
            proj_device = "cuda" if torch.cuda.is_available() else device
        else:
            proj_device = device
        self.projection = nn.Linear(base_hidden, embedding_dim).to(
            device=proj_device, dtype=dtype
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._cache = _EncodeCache(maxsize=cache_size) if cache_size > 0 else None

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
        if self._cache is not None and return_schema:
            cached = self._cache.get(text)
            if cached is not None:
                schema, vec = cached
                # Re-issue a fresh task_id per call — the embedding is
                # cached but each request has its own id.
                schema = schema.model_copy(update={"task_id": str(uuid.uuid4())})
                return schema, vec
        elif self._cache is not None:
            # return_schema=False: still cache by reusing the schema slot's
            # vec. The cache miss branch below will populate; on hit we
            # only need the vec.
            cached = self._cache.get(text)
            if cached is not None:
                _, vec = cached
                return None, vec

        schema, vec = self._encode_uncached(text, return_schema=return_schema)

        if self._cache is not None:
            # Store the canonical (schema_with_some_task_id, vec). When
            # return_schema=False we synthesize a minimal schema so the
            # cache can serve subsequent return_schema=True hits without
            # rerunning the model.
            cache_schema = schema if schema is not None else self._build_schema(text, vec)
            self._cache.put(text, (cache_schema, vec))
        return schema, vec

    def cache_stats(self) -> dict | None:
        return self._cache.stats() if self._cache is not None else None

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    def _encode_uncached(
        self, text: str, return_schema: bool = True
    ) -> Tuple[CompactSchemaV15 | None, np.ndarray]:
        torch, _ = _load_torch()
        hidden = self._last_hidden(text)
        proj = self.projection(hidden.to(dtype=self.projection.weight.dtype))
        np_dtype = np.float16 if self.dtype == torch.float16 else np.float32
        vec = proj.detach().to("cpu").to(torch.float32).numpy().astype(np_dtype)

        if not return_schema:
            return None, vec

        return self._build_schema(text, vec), vec

    def _build_schema(self, text: str, vec: np.ndarray) -> CompactSchemaV15:
        complexity = self.classify(text)
        dtype_label = "float16" if vec.dtype == np.float16 else "float32"
        return CompactSchemaV15(
            task_id=str(uuid.uuid4()),
            task_type="other",
            complexity=complexity,
            semantic_tags=[],
            confidence=0.0,
            embedding_dim=self.embedding_dim,
            embedding_dtype=dtype_label,
            embedding_b64=base64.b64encode(vec.tobytes()).decode("ascii"),
        )


    def encode_chunked(
        self,
        text: str,
        chunk_size: int = 4096,
        overlap: int = 256,
        return_schema: bool = True,
    ) -> Tuple[list, list]:
        """Encode a long text by splitting into chunks at token boundaries.

        Args:
            text: full input text
            chunk_size: max tokens per chunk
            overlap: token overlap between chunks for context preservation
            return_schema: emit a schema per chunk

        Returns:
            ``(schemas, vecs)`` — both length N where
            ``N = ceil(total_tokens / (chunk_size - overlap))`` for long texts,
            or ``N = 1`` for texts that fit in a single chunk.

        Each chunk is run through ``encode()`` so the LRU cache and any
        instrumentation apply per-chunk; overlapping prefixes between
        consecutive chunks therefore reuse cached entries when their text
        slices match exactly.
        """
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )

        full_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        total = full_ids.shape[0]

        if total <= chunk_size:
            schema, vec = self.encode(text, return_schema=return_schema)
            return [schema], [vec]

        schemas: list = []
        vecs: list = []
        step = chunk_size - overlap

        pos = 0
        while pos < total:
            end = min(pos + chunk_size, total)
            chunk_ids = full_ids[pos:end]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if not chunk_text.strip():
                if end >= total:
                    break
                pos += step
                continue
            schema, vec = self.encode(chunk_text, return_schema=return_schema)
            if return_schema and schema is not None:
                schema = schema.model_copy(
                    update={"max_tokens_hint": min(schema.max_tokens_hint, chunk_size)}
                )
            schemas.append(schema)
            vecs.append(vec)
            if end >= total:
                break
            pos += step
        return schemas, vecs


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
