"""V2.5 vector quantizer for V1.5 soft prompts.

Replaces the K continuous soft-prompt vectors emitted by
``SoftPromptAdapter.mlp`` with K discrete codebook indices, dropping
wire payload from ``K * hidden * 2`` bytes (~18 KB at K=8, hidden=1152)
to ``K * 2`` bytes (16 B). The codebook is learned end-to-end with the
rest of the adapter via the standard VQ-VAE EMA update.

Implementation notes:

- EMA codebook update (van den Oord 2017 §3.2). Avoids gradient flow
  through codebook entries; tracks running stats of cluster-assignment
  counts and embedding sums.
- Straight-through estimator: forward returns the quantized vector,
  but backward passes the gradient through the MLP output unchanged.
- Dead-code revival: every ``revive_every`` steps, codes whose usage
  counter is below ``dead_threshold`` get re-initialised to a high-
  loss training input, preventing codebook collapse.
- Commitment loss: encourages MLP outputs to stay near their nearest
  code. Returned alongside the quantized vector so callers add it to
  the joint training loss.

Usage::

    quantizer = VectorQuantizer(num_codes=4096, code_dim=cloud_hidden)
    q, info = quantizer(soft)            # soft: (B, K, cloud_hidden)
    loss = ce_loss + 0.25 * info["commit_loss"]
    # info["indices"]: (B, K) int64 — what to put on the wire

Lazy-imports torch so the module remains importable without GPU deps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _load_torch():
    import torch
    import torch.nn as nn

    return torch, nn


@dataclass
class VQConfig:
    num_codes: int = 4096
    code_dim: int = 1152
    decay: float = 0.99  # EMA decay for codebook updates
    eps: float = 1e-5  # numerical stability for cluster size normalisation
    commit_weight: float = 0.25  # β in the VQ-VAE paper
    revive_every: int = 200  # steps between dead-code revivals
    dead_threshold: float = 0.5  # below this cluster size → revive


class VectorQuantizer:
    """EMA-updated VQ-VAE quantizer with dead-code revival.

    Not a ``torch.nn.Module`` because we lazy-import torch at construct
    time; the codebook tensors live as plain attributes on the instance.
    """

    def __init__(self, cfg: VQConfig | None = None, **kwargs: Any) -> None:
        torch, _ = _load_torch()
        self.cfg = cfg or VQConfig(**kwargs)
        # Codebook: (num_codes, code_dim).
        # Initialise with small random values; EMA updates take over once
        # training begins.
        self.codebook = torch.randn(self.cfg.num_codes, self.cfg.code_dim) * 0.01
        # EMA stats — running cluster-assignment count and weighted sum.
        self.cluster_size = torch.zeros(self.cfg.num_codes)
        self.embed_sum = self.codebook.clone()
        self._step = 0

    # ------------------------------------------------------------------
    # device / dtype management
    # ------------------------------------------------------------------

    def to(self, *, device: Any, dtype: Any | None = None) -> "VectorQuantizer":
        self.codebook = self.codebook.to(device=device, dtype=dtype)
        self.cluster_size = self.cluster_size.to(device=device)
        self.embed_sum = self.embed_sum.to(device=device, dtype=dtype)
        return self

    def state_dict(self) -> dict:
        return {
            "codebook": self.codebook,
            "cluster_size": self.cluster_size,
            "embed_sum": self.embed_sum,
            "step": self._step,
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        self.codebook = state["codebook"]
        self.cluster_size = state["cluster_size"]
        self.embed_sum = state["embed_sum"]
        self._step = state.get("step", 0)

    # ------------------------------------------------------------------
    # core forward
    # ------------------------------------------------------------------

    def __call__(self, x: "torch.Tensor", training: bool = True) -> tuple["torch.Tensor", dict]:
        """Quantize ``x`` of shape ``(..., code_dim)``.

        Returns ``(q, info)`` where ``q`` has the same shape as ``x``
        (with straight-through gradient routing) and ``info`` contains
        ``indices`` (``(...,)`` int64), ``commit_loss``, ``codebook_loss``,
        and ``utilization`` (fraction of codes actually used in the call).
        """
        torch, _ = _load_torch()
        flat = x.reshape(-1, self.cfg.code_dim)  # (B*K, code_dim)

        # Distance: ‖x‖² + ‖e‖² − 2 x·eᵀ
        codebook = self.codebook.to(dtype=flat.dtype)
        x_sq = (flat * flat).sum(dim=-1, keepdim=True)
        e_sq = (codebook * codebook).sum(dim=-1)
        dist = x_sq + e_sq - 2.0 * flat @ codebook.t()
        indices = dist.argmin(dim=-1)  # (B*K,)
        q_flat = codebook[indices]
        flat_indices = indices.reshape(x.shape[:-1])

        # Losses (commitment is the only one we backprop through ‖x − sg(q)‖²;
        # codebook loss is for monitoring — EMA handles updates).
        commit_loss = self.cfg.commit_weight * (
            (flat.detach() - q_flat) ** 2
        ).mean() if False else self.cfg.commit_weight * (
            (flat - q_flat.detach()) ** 2
        ).mean()
        with torch.no_grad():
            codebook_loss = ((flat.detach() - q_flat.detach()) ** 2).mean()

        # Straight-through estimator.
        q_st = flat + (q_flat - flat).detach()
        q_st = q_st.reshape(x.shape)

        # EMA codebook update (training only).
        if training:
            with torch.no_grad():
                onehot = torch.zeros(
                    flat.shape[0], self.cfg.num_codes, device=flat.device, dtype=flat.dtype
                )
                onehot.scatter_(1, indices.unsqueeze(1), 1.0)
                cluster_sum = onehot.sum(dim=0)  # (num_codes,)
                embed_sum_step = onehot.t() @ flat.detach()  # (num_codes, code_dim)

                self.cluster_size = self.cluster_size.to(cluster_sum.dtype)
                self.embed_sum = self.embed_sum.to(embed_sum_step.dtype)

                self.cluster_size.mul_(self.cfg.decay).add_(
                    cluster_sum, alpha=1.0 - self.cfg.decay
                )
                self.embed_sum.mul_(self.cfg.decay).add_(
                    embed_sum_step, alpha=1.0 - self.cfg.decay
                )
                # Normalise: e_k = embed_sum_k / (cluster_size_k + eps)
                n = self.cluster_size.sum()
                normaliser = (
                    (self.cluster_size + self.cfg.eps)
                    / (n + self.cfg.num_codes * self.cfg.eps)
                    * n
                )
                self.codebook = self.embed_sum / normaliser.unsqueeze(1)
            self._step += 1
            if self.cfg.revive_every > 0 and self._step % self.cfg.revive_every == 0:
                self._revive_dead_codes(flat)

        with torch.no_grad():
            n_used = int(torch.unique(indices).numel())
            utilization = n_used / self.cfg.num_codes

        return q_st, {
            "indices": flat_indices,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "utilization": utilization,
        }

    def _revive_dead_codes(self, flat: "torch.Tensor") -> None:
        """Re-initialise unused codes to randomly chosen training inputs.

        Counters dropouts / collapse where most codes never get assigned.
        """
        torch, _ = _load_torch()
        with torch.no_grad():
            dead = self.cluster_size < self.cfg.dead_threshold
            n_dead = int(dead.sum().item())
            if n_dead == 0:
                return
            # Sample n_dead inputs at random from the current batch.
            B = flat.shape[0]
            idx = torch.randint(0, B, (n_dead,), device=flat.device)
            self.codebook[dead] = flat[idx].to(self.codebook.dtype)
            # Reset their EMA stats so they participate fresh.
            self.cluster_size[dead] = 1.0
            self.embed_sum[dead] = self.codebook[dead]
