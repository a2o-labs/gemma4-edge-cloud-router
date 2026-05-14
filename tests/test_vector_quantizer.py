"""CPU-only smoke tests for the V2.5 vector quantizer."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router.vector_quantizer import VectorQuantizer, VQConfig  # noqa: E402


def test_quantize_returns_same_shape() -> None:
    q = VectorQuantizer(VQConfig(num_codes=64, code_dim=16))
    x = torch.randn(4, 8, 16)
    out, info = q(x, training=False)
    assert out.shape == x.shape
    assert info["indices"].shape == (4, 8)


def test_indices_are_in_range() -> None:
    q = VectorQuantizer(VQConfig(num_codes=32, code_dim=8))
    x = torch.randn(2, 8)
    _, info = q(x, training=False)
    assert info["indices"].min() >= 0
    assert info["indices"].max() < 32


def test_straight_through_gradient_flows_to_input() -> None:
    """The forward output is q, but backward should pass d(loss)/d(q) → d(loss)/d(x)."""
    q = VectorQuantizer(VQConfig(num_codes=16, code_dim=4))
    x = torch.randn(3, 4, requires_grad=True)
    out, info = q(x, training=False)
    loss = out.sum() + info["commit_loss"]
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_codebook_updates_under_ema() -> None:
    """After many training steps with the same data, codebook should drift toward inputs."""
    cfg = VQConfig(num_codes=16, code_dim=4, decay=0.5, revive_every=0)
    q = VectorQuantizer(cfg)
    initial = q.codebook.clone()
    x = torch.randn(64, 4) * 5.0  # large-magnitude inputs
    for _ in range(50):
        q(x, training=True)
    drift = (q.codebook - initial).abs().mean().item()
    assert drift > 0.1, f"codebook didn't move enough: drift={drift}"


def test_dead_code_revival_reinits_unused_codes() -> None:
    cfg = VQConfig(num_codes=4, code_dim=4, decay=0.99, revive_every=10, dead_threshold=0.5)
    q = VectorQuantizer(cfg)
    # Single-cluster inputs → only one code ever gets used.
    x = torch.zeros(8, 4) + 1.0
    cb_before = q.codebook.clone()
    for _ in range(50):
        q(x, training=True)
    cb_after = q.codebook
    # At least one of the dead codes should have been revived (changed).
    diff = (cb_after - cb_before).abs().sum(dim=-1)
    assert (diff > 1e-6).any(), "no code was revived"


def test_utilization_one_code_at_start() -> None:
    """With random init and a single tight cluster, utilization is low."""
    q = VectorQuantizer(VQConfig(num_codes=128, code_dim=16))
    x = torch.zeros(4, 8, 16)
    _, info = q(x, training=False)
    # All zeros map to a single code.
    assert info["utilization"] <= 1.0 / 128


def test_state_dict_round_trip() -> None:
    cfg = VQConfig(num_codes=16, code_dim=8, revive_every=0)
    q = VectorQuantizer(cfg)
    x = torch.randn(32, 8)
    for _ in range(5):
        q(x, training=True)
    state = q.state_dict()
    q2 = VectorQuantizer(VQConfig(num_codes=16, code_dim=8, revive_every=0))
    q2.load_state_dict(state)
    assert torch.allclose(q.codebook, q2.codebook)
    assert torch.allclose(q.cluster_size, q2.cluster_size)
    assert q._step == q2._step
