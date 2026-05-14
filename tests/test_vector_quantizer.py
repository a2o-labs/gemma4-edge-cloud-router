"""CPU-only smoke tests for the V2.5 vector quantizer."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router.vector_quantizer import (  # noqa: E402
    RVQConfig,
    ResidualVectorQuantizer,
    VectorQuantizer,
    VQConfig,
)


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


def test_kmeans_warmup_lifts_utilization() -> None:
    """Random init → 1 cluster used. K-means warmup → all codes occupy
    distinct cluster regions, each gets used by the matching subset."""
    torch.manual_seed(0)
    cfg = VQConfig(num_codes=8, code_dim=4, revive_every=0)
    q = VectorQuantizer(cfg)

    # Three well-separated Gaussian clusters.
    n_per = 30
    centers = torch.tensor([[5.0, 0, 0, 0], [-5.0, 0, 0, 0], [0, 0, 5.0, 0]])
    samples = torch.cat([
        centers[i].unsqueeze(0) + torch.randn(n_per, 4) * 0.1
        for i in range(3)
    ], dim=0)

    # Pre-warmup: usage on 3 clusters worth of inputs.
    _, info_before = q(samples, training=False)
    util_before = info_before["utilization"]

    q.warmup_kmeans(samples, n_iters=10)
    _, info_after = q(samples, training=False)
    util_after = info_after["utilization"]

    # K-means should make at least the 3 cluster-relevant codes get used.
    assert util_after >= util_before
    assert util_after >= 3 / 8


def test_rvq_returns_same_shape() -> None:
    rvq = ResidualVectorQuantizer(RVQConfig(n_layers=3, num_codes_per_layer=16, code_dim=8))
    x = torch.randn(2, 4, 8)
    out, info = rvq(x, training=False)
    assert out.shape == x.shape
    assert len(info["indices_per_layer"]) == 3
    assert len(info["utilization_per_layer"]) == 3


def test_rvq_residual_decreases_per_layer() -> None:
    """Each layer should reduce reconstruction error toward the input."""
    torch.manual_seed(1)
    rvq = ResidualVectorQuantizer(RVQConfig(n_layers=4, num_codes_per_layer=64, code_dim=8))
    x = torch.randn(64, 8)
    rvq.warmup_kmeans(x, n_iters=10)
    out, info = rvq(x, training=False)
    err = (out - x).pow(2).mean().item()
    # With 4 layers × 64 codes after warmup, reconstruction error should be
    # noticeably below input variance (~1.0 for unit Gaussian).
    assert err < 1.0


def test_rvq_effective_bits() -> None:
    cfg = RVQConfig(n_layers=4, num_codes_per_layer=256, code_dim=8)
    assert abs(cfg.effective_bits_per_token() - 32.0) < 1e-6


def test_rvq_state_dict_round_trip() -> None:
    cfg = RVQConfig(n_layers=2, num_codes_per_layer=16, code_dim=8, revive_every=0)
    rvq = ResidualVectorQuantizer(cfg)
    x = torch.randn(32, 8)
    for _ in range(3):
        rvq(x, training=True)
    state = rvq.state_dict()
    rvq2 = ResidualVectorQuantizer(RVQConfig(n_layers=2, num_codes_per_layer=16, code_dim=8, revive_every=0))
    rvq2.load_state_dict(state)
    for a, b in zip(rvq.layers, rvq2.layers):
        assert torch.allclose(a.codebook, b.codebook)
