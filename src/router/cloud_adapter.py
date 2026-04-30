"""Cloud adapter: trainable MLP soft-prompt -> frozen Gemma 4 31B.

Maps an edge-side embedding vector to ``K`` soft-prompt token embeddings
which are prepended to the frozen Gemma 4 31B input embedding stream
before generation. Inspired by the BLIP-2 Q-Former pattern: only the small
adapter MLP is trainable, the cloud base model stays frozen.

Heavy deps (torch, transformers) are lazy-imported inside methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


@dataclass
class CloudAdapterConfig:
    """Backwards-compat config retained from the V1.5 stub."""

    base_model_id: str = "google/gemma-4-31B-it"
    quantization: str = "Q4_K_M"
    soft_prompt_k: int = 8
    soft_prompt_emb_dim: int = 4096
    mlp_hidden_dim: int = 8192
    device: str = "cuda:0"
    dtype: str = "float16"
    max_new_tokens: int = 1024


def _load_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(
            "cloud_adapter requires torch + transformers. Install the "
            "'training' extra or deploy on the A100 80GB cloud image."
        ) from e
    return torch, nn


def _load_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "cloud_adapter requires transformers>=4.45. Install the "
            "'training' extra; on cloud images it is preinstalled."
        ) from e
    return AutoModelForCausalLM, AutoTokenizer


class SoftPromptAdapter:
    """edge_vec(D_e) -> MLP -> (K, D_c) soft prompt -> frozen cloud LM.

    forward(edge_vec, json_text, max_new_tokens) returns the decoded string.
    Only ``self.mlp`` is trainable; the cloud base is frozen.
    """

    def __init__(
        self,
        edge_dim: int = 4096,
        cloud_hidden: int | None = None,
        prompt_tokens: int = 8,
        hf_token: str | None = None,
        cloud_model_name: str = "google/gemma-4-31B-it",
        device: str = "cuda",
        dtype: "Any" = None,
        quantization_config: "Any" = None,
    ) -> None:
        torch, nn = _load_torch()
        AutoModelForCausalLM, AutoTokenizer = _load_transformers()

        if dtype is None:
            dtype = torch.float16

        self.edge_dim = edge_dim
        self.prompt_tokens = prompt_tokens
        self.cloud_model_name = cloud_model_name
        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cloud_model_name, token=hf_token
            )
            load_kwargs: dict[str, Any] = {
                "token": hf_token,
                "output_hidden_states": False,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = dtype
            cloud = AutoModelForCausalLM.from_pretrained(cloud_model_name, **load_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cloud model '{cloud_model_name}'. Hint: "
                f"production deployment runs Gemma 4 31B on A100 80GB "
                f"(vast.ai per SRE plan) with HF_TOKEN. Underlying error: {e}"
            ) from e

        if quantization_config is None:
            try:
                cloud = cloud.to(device)
            except Exception:
                pass

        for p in cloud.parameters():
            p.requires_grad_(False)
        cloud.eval()
        self.cloud = cloud

        detected_hidden = cloud.config.hidden_size
        if cloud_hidden is None:
            cloud_hidden = detected_hidden
        elif cloud_hidden != detected_hidden:
            cloud_hidden = detected_hidden
        self.cloud_hidden = cloud_hidden

        if quantization_config is not None:
            mlp_device = "cuda" if torch.cuda.is_available() else device
        else:
            mlp_device = device
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Linear(edge_dim * 2, prompt_tokens * cloud_hidden),
        ).to(device=mlp_device, dtype=dtype)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def freeze_base(self) -> None:
        for p in self.cloud.parameters():
            p.requires_grad_(False)
        self.cloud.eval()

    def trainable_parameters(self) -> Iterator["nn.Parameter"]:
        return self.mlp.parameters()

    def _embed_input(self, json_text: str) -> "torch.Tensor":
        torch, _ = _load_torch()
        ids = self.tokenizer(json_text, return_tensors="pt").input_ids.to(self.device)
        embed_layer = self.cloud.get_input_embeddings()
        return embed_layer(ids)

    def forward(
        self,
        edge_vec: "torch.Tensor | np.ndarray",
        json_text: str,
        max_new_tokens: int = 256,
    ) -> str:
        torch, _ = _load_torch()

        if isinstance(edge_vec, np.ndarray):
            edge_vec = torch.from_numpy(edge_vec.astype(np.float32))
        edge_vec = edge_vec.to(device=self.device, dtype=self.dtype)
        if edge_vec.dim() == 1:
            edge_vec = edge_vec.unsqueeze(0)

        soft = self.mlp(edge_vec).reshape(
            edge_vec.shape[0], self.prompt_tokens, self.cloud_hidden
        )

        text_embeds = self._embed_input(json_text)
        if text_embeds.shape[0] != soft.shape[0]:
            text_embeds = text_embeds.expand(soft.shape[0], -1, -1)
        embeds = torch.cat([soft, text_embeds], dim=1)
        attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out_ids = self.cloud.generate(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def forward_stream(
        self,
        edge_vec: "torch.Tensor | np.ndarray",
        json_text: str,
        max_new_tokens: int = 256,
    ):
        """Yield decoded tokens as they're produced by ``cloud.generate``.

        Uses ``transformers.TextIteratorStreamer`` driven from a background
        thread so the consumer can pull chunks lazily.
        """
        import threading

        from transformers import TextIteratorStreamer

        torch, _ = _load_torch()

        if isinstance(edge_vec, np.ndarray):
            edge_vec = torch.from_numpy(edge_vec.astype(np.float32))
        edge_vec = edge_vec.to(device=self.device, dtype=self.dtype)
        if edge_vec.dim() == 1:
            edge_vec = edge_vec.unsqueeze(0)

        soft = self.mlp(edge_vec).reshape(
            edge_vec.shape[0], self.prompt_tokens, self.cloud_hidden
        )
        text_embeds = self._embed_input(json_text)
        if text_embeds.shape[0] != soft.shape[0]:
            text_embeds = text_embeds.expand(soft.shape[0], -1, -1)
        embeds = torch.cat([soft, text_embeds], dim=1)
        attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        def _run() -> None:
            with torch.no_grad():
                self.cloud.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=streamer,
                )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join(timeout=1)
