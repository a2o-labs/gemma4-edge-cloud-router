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

    def warmup_kmeans(self, samples: "torch.Tensor", n_iters: int = 10) -> None:
        """K-means warmup: replace random codebook init with cluster centers
        of ``samples``.

        Solves the cold-start collapse where random codebook entries are far
        from any MLP output, leading to all assignments concentrating on a
        single nearest code. Call once before training begins, with a batch
        of representative MLP outputs.

        ``samples`` should be ``(N, code_dim)`` with ``N >> num_codes``.
        """
        torch, _ = _load_torch()
        flat = samples.reshape(-1, self.cfg.code_dim).detach()
        if flat.shape[0] < self.cfg.num_codes:
            # Not enough samples to seed all codes — fall back to random
            # plus replicate.
            reps = (self.cfg.num_codes + flat.shape[0] - 1) // flat.shape[0]
            flat = flat.repeat(reps, 1)[: self.cfg.num_codes]
        with torch.no_grad():
            # Init: pick num_codes random samples as initial centers.
            perm = torch.randperm(flat.shape[0], device=flat.device)
            centers = flat[perm[: self.cfg.num_codes]].to(self.codebook.dtype).clone()
            for _ in range(n_iters):
                # Assignment step.
                x_sq = (flat * flat).sum(dim=-1, keepdim=True)
                e_sq = (centers * centers).sum(dim=-1)
                dist = x_sq + e_sq - 2.0 * flat @ centers.t()
                idx = dist.argmin(dim=-1)
                # Update step: new center = mean of assigned points (with
                # fallback to old center for empty clusters).
                onehot = torch.zeros(
                    flat.shape[0], self.cfg.num_codes, device=flat.device, dtype=flat.dtype
                )
                onehot.scatter_(1, idx.unsqueeze(1), 1.0)
                cluster_count = onehot.sum(dim=0)
                cluster_sum = onehot.t() @ flat
                # Avoid division by zero — keep old center for empty clusters.
                mask = cluster_count > 0
                new_centers = centers.clone()
                new_centers[mask] = (
                    cluster_sum[mask] / cluster_count[mask].unsqueeze(1)
                ).to(centers.dtype)
                centers = new_centers
            self.codebook = centers
            # Seed EMA stats so existing-code regions don't get wiped out by
            # the first training step's update.
            self.cluster_size = torch.ones(self.cfg.num_codes, device=centers.device, dtype=flat.dtype)
            self.embed_sum = self.codebook.clone()


@dataclass
class RVQConfig:
    """Residual VQ: ``n_layers`` stacked quantizers, each on the residual
    of the previous. Effective bandwidth = sum of layer-bits.
    """

    n_layers: int = 4
    num_codes_per_layer: int = 256
    code_dim: int = 1152
    decay: float = 0.99
    eps: float = 1e-5
    commit_weight: float = 0.25
    revive_every: int = 200
    dead_threshold: float = 0.5

    def effective_bits_per_token(self) -> float:
        import math

        return self.n_layers * math.log2(self.num_codes_per_layer)


class ResidualVectorQuantizer:
    """Stack of ``n_layers`` VectorQuantizers, each operating on the residual
    of the previous layer's output.

    Effective channel size is ``n_layers * log2(num_codes_per_layer)`` bits
    per token, vs ``log2(num_codes)`` for a single VQ. Standard in audio
    codecs (SoundStream, Encodec) for the same reason: a single 8-bit code
    is too coarse to reconstruct natural distributions.
    """

    def __init__(self, cfg: RVQConfig | None = None, **kwargs) -> None:
        self.cfg = cfg or RVQConfig(**kwargs)
        self.layers = [
            VectorQuantizer(VQConfig(
                num_codes=self.cfg.num_codes_per_layer,
                code_dim=self.cfg.code_dim,
                decay=self.cfg.decay,
                eps=self.cfg.eps,
                commit_weight=self.cfg.commit_weight,
                revive_every=self.cfg.revive_every,
                dead_threshold=self.cfg.dead_threshold,
            ))
            for _ in range(self.cfg.n_layers)
        ]

    def to(self, *, device, dtype=None) -> "ResidualVectorQuantizer":
        for layer in self.layers:
            layer.to(device=device, dtype=dtype)
        return self

    def state_dict(self) -> dict:
        return {
            "layers": [layer.state_dict() for layer in self.layers],
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        for layer, layer_state in zip(self.layers, state["layers"]):
            layer.load_state_dict(layer_state)

    def __call__(self, x, training: bool = True):
        torch, _ = _load_torch()
        residual = x
        accumulated = torch.zeros_like(x)
        all_indices = []
        commit_total = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        codebook_total = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        utilization_per_layer = []
        for layer in self.layers:
            q_layer, info = layer(residual, training=training)
            # Standard RVQ: residual_{t+1} = residual_t - q_t (using true q,
            # not straight-through, for the residual flow).
            with torch.no_grad():
                residual_update = residual - q_layer
            accumulated = accumulated + q_layer
            residual = residual_update
            all_indices.append(info["indices"])
            commit_total = commit_total + info["commit_loss"]
            codebook_total = codebook_total + info["codebook_loss"]
            utilization_per_layer.append(info["utilization"])
        return accumulated, {
            "indices_per_layer": all_indices,
            "commit_loss": commit_total / max(1, len(self.layers)),
            "codebook_loss": codebook_total / max(1, len(self.layers)),
            "utilization_per_layer": utilization_per_layer,
            "utilization": sum(utilization_per_layer) / max(1, len(utilization_per_layer)),
        }

    def warmup_kmeans(self, samples, n_iters: int = 10) -> None:
        """Warmup each layer in sequence: layer 0 on samples, layer 1 on
        residual after layer 0, etc."""
        torch, _ = _load_torch()
        residual = samples.reshape(-1, self.cfg.code_dim).detach().clone()
        for layer in self.layers:
            layer.warmup_kmeans(residual, n_iters=n_iters)
            with torch.no_grad():
                # Compute assignments + residuals for the next layer.
                x_sq = (residual * residual).sum(dim=-1, keepdim=True)
                cb = layer.codebook.to(residual.dtype)
                e_sq = (cb * cb).sum(dim=-1)
                dist = x_sq + e_sq - 2.0 * residual @ cb.t()
                idx = dist.argmin(dim=-1)
                residual = residual - cb[idx]
