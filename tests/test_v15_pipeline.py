"""End-to-end V1.5 pipeline smoke tests on a tiny CPU model.

Uses ``trl-internal-testing/tiny-random-LlamaForCausalLM`` so CI runs on
CPU without GPU. Real Gemma 4 weights are exercised in the L4/A100
deployment, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from router.cloud_adapter import SoftPromptAdapter  # noqa: E402
from router.edge_encoder import EdgeEncoder, decode_embedding_from_schema  # noqa: E402

TINY_MODEL = "trl-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def edge_encoder() -> EdgeEncoder:
    return EdgeEncoder(
        model_name=TINY_MODEL,
        embedding_dim=64,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def cloud_adapter() -> SoftPromptAdapter:
    return SoftPromptAdapter(
        edge_dim=64,
        prompt_tokens=4,
        cloud_model_name=TINY_MODEL,
        device="cpu",
        dtype=torch.float32,
    )


def test_edge_encoder_loads_tiny_model_and_encodes(edge_encoder: EdgeEncoder) -> None:
    schema, vec = edge_encoder.encode("Hello world")
    assert schema is not None
    assert schema.version == "1.5"
    assert vec.shape == (64,)
    assert schema.embedding_b64 is not None
    assert schema.embedding_dim == 64


def test_edge_encoder_classifier_light_vs_heavy(edge_encoder: EdgeEncoder) -> None:
    assert edge_encoder.classify("Hi") == "light"
    long = "x" * 250
    assert edge_encoder.classify(long) == "heavy"


def test_edge_encoder_trainable_only_projection(edge_encoder: EdgeEncoder) -> None:
    base_grad = [p.requires_grad for p in edge_encoder.base.parameters()]
    proj_grad = [p.requires_grad for p in edge_encoder.projection.parameters()]
    assert not any(base_grad)
    assert all(proj_grad)


def test_cloud_adapter_forward_round_trip(cloud_adapter: SoftPromptAdapter) -> None:
    fake_vec = torch.randn(64)
    out = cloud_adapter.forward(
        fake_vec, json_text='{"task": "test"}', max_new_tokens=8
    )
    assert isinstance(out, str)
    assert len(out) >= 0


def test_cloud_adapter_only_mlp_trainable(cloud_adapter: SoftPromptAdapter) -> None:
    cloud_grad = [p.requires_grad for p in cloud_adapter.cloud.parameters()]
    mlp_grad = [p.requires_grad for p in cloud_adapter.mlp.parameters()]
    assert not any(cloud_grad)
    assert all(mlp_grad)


def test_v15_end_to_end_pipeline(
    edge_encoder: EdgeEncoder, cloud_adapter: SoftPromptAdapter
) -> None:
    schema, vec = edge_encoder.encode("How does photosynthesis work?")
    assert schema is not None

    decoded_vec = decode_embedding_from_schema(schema)
    np.testing.assert_allclose(decoded_vec, vec, rtol=1e-3)

    out = cloud_adapter.forward(
        torch.from_numpy(decoded_vec).float(),
        json_text=schema.model_dump_json(),
        max_new_tokens=8,
    )
    assert isinstance(out, str)
