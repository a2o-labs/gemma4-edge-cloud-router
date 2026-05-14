"""V1.5 quantization-aware IR.

Different attack angle than the VQ-VAE / RVQ track of PR #32 / #33: the
soft prompt stays *continuous* through the MLP, but we transmit it as
int8 or int4 on the wire and dequantize at the cloud. Training does
fake-quantization with a straight-through estimator so the MLP learns
to produce values that round-trip cleanly through the quantizer's
limited range.

Compared to the VQ track:

  * VQ failure mode 1 (low codebook util → mode collapse) doesn't
    apply: this is per-dim scalar quantization, every dimension carries
    information independently.
  * VQ failure mode 2 (residual sums off-manifold) doesn't apply:
    int8/int4 is a regular grid in MLP-output space, the dequantized
    vector is the same continuous shape as the original, just rounded.
  * Wire payload at K=32 cloud_hidden=1152: fp16 = 73 KB →
    int8 = 36 KB → int4 = 18 KB (with one fp16 scale per soft-prompt
    slot, ~16 bytes overhead).

Two public functions for the inference path::

    x_hat, scales = fake_quantize_per_token(x, bits=4)
    # x_hat: same shape as x, but rounded through the int4 grid.
    # scales: (..., 1) fp tensor; needed for real wire encode/decode.

    x_hat = dequantize_per_token(x_q_int, scales, bits=4)
    # x_q_int: int-typed (..., D) tensor (the wire payload).

During training the straight-through estimator routes the gradient
through ``x`` unchanged so the MLP learns to live within the rounded
grid; during inference the same ``fake_quantize`` is called inside
``torch.no_grad`` and gives the dequantized vector directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _load_torch():
    import torch

    return torch


def _qmax_for(bits: int) -> int:
    if bits not in (4, 8):
        raise ValueError(f"only int4 and int8 supported, got bits={bits}")
    return (1 << (bits - 1)) - 1


def fake_quantize_per_token(
    x: "torch.Tensor",
    bits: int = 8,
    eps: float = 1e-8,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Symmetric per-token fake quantization with straight-through gradient.

    Each row of ``x`` (along the last-but-one axis) gets its own scale:
    ``s = max(|x|) / qmax`` where ``qmax = 2^(bits-1) - 1`` (= 127 for
    int8, 7 for int4). The forward returns the dequantized tensor
    (rounded and clipped through the int grid then rescaled); the
    backward passes the gradient through ``x`` unchanged via the
    straight-through estimator.

    ``x`` is expected to be ``(B, K, D)`` or ``(K, D)`` — the last axis
    is the per-token feature dim, the second-to-last is the per-token
    slot. Each (B, K) slot gets its own scale.
    """
    torch = _load_torch()
    qmax = _qmax_for(bits)
    # Per-(batch, slot) scale: reduce over the feature dim.
    # max over last axis with keepdim to broadcast.
    abs_max = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / qmax).clamp(min=eps)
    x_q_int = torch.round(x / scale).clamp(-qmax, qmax)
    x_hat = x_q_int * scale
    # Straight-through: forward is x_hat, backward routes ∂L/∂x_hat → ∂L/∂x.
    x_st = x + (x_hat - x).detach()
    return x_st, scale


def real_quantize_per_token(
    x: "torch.Tensor",
    bits: int = 8,
    eps: float = 1e-8,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize for wire transmission. Returns ``(x_q_int, scale)``.

    Same math as ``fake_quantize_per_token`` but returns the integer
    tensor (so the caller can pack it into bytes) plus the fp scale
    needed for the inverse. No gradient routing — call inside
    ``torch.no_grad()``.
    """
    torch = _load_torch()
    qmax = _qmax_for(bits)
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / qmax).clamp(min=eps)
    x_q_int = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8 if bits == 8 else torch.int8)
    return x_q_int, scale


def dequantize_per_token(
    x_q_int: "torch.Tensor",
    scale: "torch.Tensor",
) -> "torch.Tensor":
    """Inverse of ``real_quantize_per_token``."""
    return x_q_int.to(scale.dtype) * scale


def estimate_wire_bytes(K: int, D: int, bits: int) -> int:
    """Bytes per request for the soft-prompt block in this quant scheme.

    Per-token scale overhead: one fp16 scale (2 B) per (B, K) slot.
    Feature payload: K * D * (bits/8) bytes.
    """
    feature_bytes = K * D * bits // 8
    scale_bytes = K * 2  # fp16 scale per K slot, single sample
    return feature_bytes + scale_bytes
