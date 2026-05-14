"""V1.5 quantization-aware IR training.

Same training stack as `train_v15_aux.py` (CE + KL + contrastive) but
applies a per-token fake-quantizer to the MLP-output soft prompt
before the cloud LM sees it. Straight-through estimator routes the
gradient so the MLP learns to produce values that round-trip cleanly
through the int8 / int4 grid.

This is direction (2) of the post-VQ pivot documented at the bottom
of `docs/v15-vqvae-and-cross-tokenizer-results.md`: avoids both VQ
failure modes (low codebook utilisation, residual-sum off-manifold)
by keeping the soft prompt continuous and just rounding it through a
regular grid.

Run::

    cd /path/to/gemma4-router
    HF_TOKEN=... PYTHONPATH=src python -m training.train_v15_qat \\
        --train-jsonl /tmp/v15_distilled_5000.jsonl \\
        --output-dir /tmp/v15-train-qat \\
        --max-steps 3000 --batch-size 2 --prompt-tokens 32 \\
        --qat-bits 4 \\
        --aux-ce-weight 1.0 --aux-kl-weight 0.05 --aux-contrastive-weight 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Caller is responsible for setting PYTHONPATH=src so router/ + training/ are importable.

import numpy as np


def step_loss_qat(edge, adapter, samples, device, weights, qat_bits):
    import torch
    import torch.nn.functional as F

    from router.quantization_aware import fake_quantize_per_token

    B = len(samples)
    edge_vecs = []
    for s in samples:
        ids = edge._tokenize(s["prompt"])
        attn = torch.ones_like(ids)
        out = edge.base(
            input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False
        )
        last = out.hidden_states[-1][:, -1, :]
        edge_vecs.append(edge.projection(last))
    edge_vec = torch.cat(edge_vecs, dim=0)

    soft_continuous = adapter.mlp(edge_vec).reshape(
        B, adapter.prompt_tokens, adapter.cloud_hidden
    )
    # Fake-quantize per (batch, slot). Same shape, gradient routes via STE.
    soft_q, scale = fake_quantize_per_token(soft_continuous, bits=qat_bits)

    ce_terms = []
    target_logits_per_sample = []
    target_ids_per_sample = []
    for i, s in enumerate(samples):
        target_ids = adapter.tokenizer(s["target_response"], return_tensors="pt").input_ids.to(device)
        target_embeds = adapter.cloud.get_input_embeddings()(target_ids)
        embeds = torch.cat([soft_q[i : i + 1], target_embeds], dim=1)
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        out = adapter.cloud(inputs_embeds=embeds, attention_mask=attn, labels=None)
        K = adapter.prompt_tokens
        logits = out.logits[:, K - 1 : -1, :]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        )
        ce_terms.append(ce)
        target_logits_per_sample.append(logits)
        target_ids_per_sample.append(target_ids)

    losses = {
        "ce": torch.stack(ce_terms).mean(),
        # Quantization noise reporting only — not added to total loss.
        "quant_err": (soft_q.detach() - soft_continuous.detach()).abs().mean(),
        "scale_mean": scale.mean().detach(),
    }

    if weights.get("kl", 0) > 0:
        kl_terms = []
        with torch.no_grad():
            for i, s in enumerate(samples):
                prompt_ids = edge._tokenize(s["prompt"])
                target_ids = adapter.tokenizer(s["target_response"], return_tensors="pt").input_ids.to(device)
                full_ids = torch.cat([prompt_ids, target_ids], dim=1)
                full_attn = torch.ones_like(full_ids)
                edge_out = edge.base(
                    input_ids=full_ids, attention_mask=full_attn, use_cache=False
                )
                T = target_ids.shape[1]
                edge_target_logits = edge_out.logits[:, -T - 1 : -1, :]
                edge_probs = F.softmax(edge_target_logits, dim=-1)
                cloud_log = F.log_softmax(target_logits_per_sample[i], dim=-1)
                kl_terms.append(F.kl_div(cloud_log, edge_probs, reduction="batchmean"))
        if kl_terms:
            losses["kl"] = torch.stack(kl_terms).mean()

    if weights.get("contrastive", 0) > 0 and B > 1:
        soft_mean = soft_q.mean(dim=1)
        D = min(soft_mean.shape[-1], edge_vec.shape[-1])
        a = F.normalize(edge_vec[:, :D], dim=-1)
        b = F.normalize(soft_mean[:, :D], dim=-1)
        logits = (a @ b.T) * 10.0
        labels = torch.arange(B, device=device)
        c1 = F.cross_entropy(logits, labels)
        c2 = F.cross_entropy(logits.T, labels)
        losses["contrastive"] = (c1 + c2) / 2

    total = (
        weights.get("ce", 1.0) * losses["ce"]
        + weights.get("kl", 0.0) * losses.get("kl", torch.tensor(0.0, device=device))
        + weights.get("contrastive", 0.0)
        * losses.get("contrastive", torch.tensor(0.0, device=device))
    )
    return total, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--edge-model", default="google/gemma-3-1b-it")
    ap.add_argument("--cloud-model", default="google/gemma-3-1b-it")
    ap.add_argument("--embedding-dim", type=int, default=1152)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qat-bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--aux-ce-weight", type=float, default=1.0)
    ap.add_argument("--aux-kl-weight", type=float, default=0.05)
    ap.add_argument("--aux-contrastive-weight", type=float, default=0.1)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raw_path = Path(args.train_jsonl)
    converted_path = raw_path.with_suffix(".converted.jsonl")
    n = 0
    with raw_path.open() as fin, converted_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tgt = d.get("target_response") or d.get("expected_answer") or ""
            if not tgt.strip():
                continue
            fout.write(json.dumps({
                "prompt": d["prompt"],
                "target_response": tgt,
                "task_type": d.get("task_type") or "other",
                "complexity": d.get("complexity") or d.get("expected_complexity") or "light",
            }) + "\n")
            n += 1
    print(f"[data] {n} samples -> {converted_path}")

    from training.train_v15 import TrainConfig, build_pipeline, load_jsonl

    cfg = TrainConfig(
        edge_model=args.edge_model,
        cloud_model=args.cloud_model,
        embedding_dim=args.embedding_dim,
        prompt_tokens=args.prompt_tokens,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        train_jsonl=str(converted_path),
        eval_jsonl=str(converted_path),
        mock_mode=False,
        device="cuda",
        hf_token=os.environ.get("HF_TOKEN"),
        seed=args.seed,
    )

    edge, adapter, device, _ = build_pipeline(cfg)
    edge.freeze_base()
    adapter.freeze_base()

    print(f"[qat] bits={args.qat_bits} prompt_tokens={args.prompt_tokens} cloud_hidden={adapter.cloud_hidden}")
    from router.quantization_aware import estimate_wire_bytes

    fp16_bytes = args.prompt_tokens * adapter.cloud_hidden * 2
    qat_bytes = estimate_wire_bytes(args.prompt_tokens, adapter.cloud_hidden, args.qat_bits)
    print(
        f"[qat] wire payload: fp16={fp16_bytes} B  qat-{args.qat_bits}bit={qat_bytes} B "
        f"({fp16_bytes / qat_bytes:.1f}x reduction)"
    )

    params = list(edge.trainable_parameters()) + list(adapter.trainable_parameters())
    optim = torch.optim.AdamW(params, lr=cfg.learning_rate)

    train_data = load_jsonl(cfg.train_jsonl)
    weights = {
        "ce": args.aux_ce_weight,
        "kl": args.aux_kl_weight,
        "contrastive": args.aux_contrastive_weight,
    }
    print(f"[loss] weights = {weights}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    t_start = time.perf_counter()
    while step < cfg.max_steps:
        idxs = np.random.choice(len(train_data), size=cfg.batch_size, replace=False).tolist()
        samples = [train_data[i] for i in idxs]
        total, parts = step_loss_qat(edge, adapter, samples, device, weights, args.qat_bits)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optim.step()
        optim.zero_grad()
        step += 1
        if step % cfg.log_every == 0 or step <= 2:
            line = " ".join(
                f"{k}={(v.item() if hasattr(v, 'item') else float(v)):.3f}"
                for k, v in parts.items()
            )
            print(f"step={step} total={total.item():.3f} {line}")
        if step % cfg.save_every == 0:
            torch.save({
                "step": step,
                "projection_state": edge.projection.state_dict(),
                "mlp_state": adapter.mlp.state_dict(),
                "qat_bits": args.qat_bits,
                "config": cfg.__dict__,
                "weights": weights,
            }, out_dir / f"ckpt-step{step}.pt")
            print(f"checkpoint -> {out_dir}/ckpt-step{step}.pt")

    torch.save({
        "step": step,
        "projection_state": edge.projection.state_dict(),
        "mlp_state": adapter.mlp.state_dict(),
        "qat_bits": args.qat_bits,
        "config": cfg.__dict__,
        "weights": weights,
    }, out_dir / "final.pt")
    print(f"[done] {(time.perf_counter() - t_start):.1f}s -> {out_dir}/final.pt")


if __name__ == "__main__":
    main()
