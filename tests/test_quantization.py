"""Verify bnb 4-bit kwarg plumbing through EdgeEncoder / SoftPromptAdapter
constructors. Skip the load-path tests when bitsandbytes / CUDA aren't
available — production is GPU-only, so CI on a CPU host skips them by
design and only runs the always-on signature check."""

from __future__ import annotations

import inspect
import os

os.environ["V15_MOCK_MODE"] = "true"

import pytest


def test_quantization_kwarg_signature_exists():
    """Always-on: confirm both constructors accept quantization_config."""
    from router.cloud_adapter import SoftPromptAdapter
    from router.edge_encoder import EdgeEncoder

    sig = inspect.signature(EdgeEncoder.__init__)
    assert "quantization_config" in sig.parameters

    sig = inspect.signature(SoftPromptAdapter.__init__)
    assert "quantization_config" in sig.parameters


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_edge_encoder_accepts_quantization_config():
    pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig

    from router.edge_encoder import EdgeEncoder

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    enc = EdgeEncoder(
        model_name="trl-internal-testing/tiny-random-LlamaForCausalLM",
        embedding_dim=64,
        device="cuda",
        quantization_config=bnb_config,
    )
    schema, vec = enc.encode("Hello")
    assert schema.version == "1.5"
    assert vec.shape == (64,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cloud_adapter_accepts_quantization_config():
    pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig

    from router.cloud_adapter import SoftPromptAdapter

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    adapter = SoftPromptAdapter(
        edge_dim=64,
        prompt_tokens=4,
        cloud_model_name="trl-internal-testing/tiny-random-LlamaForCausalLM",
        device="cuda",
        quantization_config=bnb_config,
    )
    out = adapter.forward(torch.randn(64), json_text='{"task":"test"}', max_new_tokens=4)
    assert isinstance(out, str)
