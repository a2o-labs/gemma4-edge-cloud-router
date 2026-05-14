"""Tests for quantization-aware IR helpers."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from router.quantization_aware import (  # noqa: E402
    dequantize_per_token,
    estimate_wire_bytes,
    fake_quantize_per_token,
    real_quantize_per_token,
)


def test_int8_round_trip_error_bounded() -> None:
    """For random Gaussian inputs the int8 round-trip error per element
    should be below the theoretical worst case of (max/127)."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 64)
    x_hat, scale = fake_quantize_per_token(x, bits=8)
    assert x_hat.shape == x.shape
    err = (x_hat - x).abs()
    # Theoretical max per element is scale/2 (rounding); allow generous slack.
    assert (err <= scale.squeeze(-1).unsqueeze(-1)).all()


def test_int4_round_trip_error_grows_but_bounded() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 4, 64)
    x_hat, scale = fake_quantize_per_token(x, bits=4)
    assert x_hat.shape == x.shape
    err = (x_hat - x).abs()
    # int4 has a coarser grid; error still bounded by scale.
    assert (err <= scale.squeeze(-1).unsqueeze(-1)).all()
    # And int4 error should be larger on average than int8 (sanity).
    x_hat8, _ = fake_quantize_per_token(x, bits=8)
    assert (x_hat - x).abs().mean() > (x_hat8 - x).abs().mean()


def test_straight_through_gradient_flows() -> None:
    x = torch.randn(3, 8, 16, requires_grad=True)
    x_hat, _ = fake_quantize_per_token(x, bits=4)
    loss = x_hat.sum()
    loss.backward()
    assert x.grad is not None
    # Straight-through means grad ones.
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_real_quantize_returns_int_payload() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 16)
    with torch.no_grad():
        x_q_int, scale = real_quantize_per_token(x, bits=8)
    assert x_q_int.dtype == torch.int8
    assert x_q_int.shape == x.shape
    assert scale.shape == (2, 1)
    assert (x_q_int.abs() <= 127).all()


def test_real_quantize_int4_packed_in_int8_range() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 16)
    with torch.no_grad():
        x_q_int, _ = real_quantize_per_token(x, bits=4)
    assert (x_q_int.abs() <= 7).all()


def test_dequantize_inverse() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 16)
    with torch.no_grad():
        x_q_int, scale = real_quantize_per_token(x, bits=8)
        x_hat = dequantize_per_token(x_q_int, scale)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_estimate_wire_bytes_at_typical_settings() -> None:
    # Match the V1.5 production shape: K=32, cloud_hidden=1152.
    fp16 = 32 * 1152 * 2  # 73 728
    int8 = estimate_wire_bytes(32, 1152, 8)
    int4 = estimate_wire_bytes(32, 1152, 4)
    assert int8 < fp16 / 1.9  # ~2× saving
    assert int4 < fp16 / 3.8  # ~4× saving


def test_invalid_bits_raises() -> None:
    x = torch.randn(2, 8)
    with pytest.raises(ValueError):
        fake_quantize_per_token(x, bits=16)
