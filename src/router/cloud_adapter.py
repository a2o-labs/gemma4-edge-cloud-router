"""Cloud adapter: soft-prompt MLP that projects edge embedding to K=8 prompt
tokens, prepended into Gemma4-31B (frozen) at inference time.

V1.5 stub. Real model + MLP wiring are TODO; this module fixes the shape
contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CloudAdapterConfig:
    base_model_id: str = "google/gemma-4-31b"
    quantization: str = "Q4_K_M"
    soft_prompt_k: int = 8
    soft_prompt_emb_dim: int = 4096
    mlp_hidden_dim: int = 8192
    device: str = "cuda:0"
    dtype: str = "float16"
    max_new_tokens: int = 1024


class CloudAdapter:
    """vec(4096) -> MLP[4096, 8192, 8*4096] -> reshape to (K=8, 4096)
    soft prompt prepended into frozen Gemma4-31B forward.
    """

    def __init__(self, cfg: CloudAdapterConfig | None = None) -> None:
        self.cfg = cfg or CloudAdapterConfig()
        self._mlp = None
        self._llm = None
        self._tokenizer = None
        print("TODO real model: CloudAdapter.__init__ has not loaded weights")

    def load(self) -> None:
        # TODO: build nn.Sequential(Linear(4096, 8192), GELU, Linear(8192, K*4096))
        # TODO: from transformers import AutoModelForCausalLM, AutoTokenizer
        # TODO: load Gemma4-31B Q4 GGUF / HF, freeze all params
        print("TODO real model: CloudAdapter.load skipped")

    def _vec_to_soft_prompt(self, vec: np.ndarray) -> np.ndarray:
        # TODO: forward MLP, reshape to (K, emb_dim).
        K, D = self.cfg.soft_prompt_k, self.cfg.soft_prompt_emb_dim
        return np.zeros((K, D), dtype=np.float16)

    def forward(self, vec: np.ndarray, json_text: str) -> str:
        """Run frozen Gemma4-31B with soft prompt prepended to json_text input.

        Returns generated string. Stub returns empty string.
        """
        _ = self._vec_to_soft_prompt(vec)
        # TODO: tokenize json_text, prepend soft prompt embeddings into
        # input_embeds, call model.generate(...)
        return ""
