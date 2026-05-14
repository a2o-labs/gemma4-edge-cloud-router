"""V1.5 training with auxiliary losses (KL distillation + contrastive).

Drop-in replacement for training/train_v15.py's `step_loss` and `train`,
paper-companion training script. Adds:

  ce          : standard next-token cross-entropy on target tokens
  kl          : KL(cloud-given-soft-prompt || edge-given-prompt+target)
                forces the soft prompt to elicit the same target-token
                distribution that edge LM gets when it sees the raw prompt
  contrastive : symmetric InfoNCE between edge_vec[i] and soft_mean[j]
                across the batch — requires batch_size > 1, defends
                against mode collapse

Run:
    cd /path/to/gemma4-router
    HF_TOKEN=... PYTHONPATH=src python -m training.train_v15_aux \
        --train-jsonl /tmp/v15_distilled_500.jsonl \
        --output-dir /tmp/v15-train-l4-aux \
        --max-steps 1500 --batch-size 2 --prompt-tokens 8 \
        --aux-kl-weight 0.5 --aux-contrastive-weight 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Caller is responsible for setting PYTHONPATH=src so router/ + training/ are importable.

import numpy as np


def step_loss_aux(edge, adapter, samples, device, weights):
    """Multi-loss step. `samples` is a list of B sample dicts (batch).

    Returns (total_loss, losses_dict) where losses_dict is per-component for logging.
    """
    import torch
    import torch.nn.functional as F

    B = len(samples)
    # 1) Per-sample edge forward (we run them one at a time to avoid
    #    edge tokenizer padding edge cases; still backprop through projection).
    edge_vecs = []
    for s in samples:
        ids = edge._tokenize(s["prompt"])
        attn = torch.ones_like(ids)
        out = edge.base(
            input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False
        )
        last = out.hidden_states[-1][:, -1, :]
        edge_vecs.append(edge.projection(last))
    edge_vec = torch.cat(edge_vecs, dim=0)  # [B, D_edge]

    soft = adapter.mlp(edge_vec).reshape(B, adapter.prompt_tokens, adapter.cloud_hidden)

    # 2) Per-sample cloud forward & CE
    ce_terms = []
    target_logits_per_sample = []
    target_ids_per_sample = []
    for i, s in enumerate(samples):
        target = s["target_response"]
        target_ids = adapter.tokenizer(target, return_tensors="pt").input_ids.to(device)
        target_embeds = adapter.cloud.get_input_embeddings()(target_ids)
        embeds = torch.cat([soft[i : i + 1], target_embeds], dim=1)
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

    ce_loss = torch.stack(ce_terms).mean()
    losses = {"ce": ce_loss}

    # 3) KL distillation against edge LM's distribution
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
                # KL(edge || cloud_under_soft_prompt) — we want cloud to MATCH edge
                kl_terms.append(
                    F.kl_div(cloud_log, edge_probs, reduction="batchmean")
                )
        if kl_terms:
            losses["kl"] = torch.stack(kl_terms).mean()

    # 4) Contrastive (CLIP-style) between edge_vec and soft_mean across batch
    if weights.get("contrastive", 0) > 0 and B > 1:
        soft_mean = soft.mean(dim=1)  # [B, cloud_hidden]
        # Project soft_mean to edge_dim via slice (zero-cost, no extra params).
        D = min(soft_mean.shape[-1], edge_vec.shape[-1])
        a = F.normalize(edge_vec[:, :D], dim=-1)
        b = F.normalize(soft_mean[:, :D], dim=-1)
        logits = (a @ b.T) * 10.0  # temperature scaling
        labels = torch.arange(B, device=device)
        c1 = F.cross_entropy(logits, labels)
        c2 = F.cross_entropy(logits.T, labels)
        losses["contrastive"] = (c1 + c2) / 2

    total = sum(weights.get(k, 0) * v for k, v in losses.items())
    return total, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--edge-model", default="google/gemma-3-1b-it")
    ap.add_argument("--cloud-model", default="google/gemma-3-1b-it")
    ap.add_argument("--embedding-dim", type=int, default=1152)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    # Loss weights
    ap.add_argument("--aux-ce-weight", type=float, default=1.0)
    ap.add_argument("--aux-kl-weight", type=float, default=0.5)
    ap.add_argument("--aux-contrastive-weight", type=float, default=0.1)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Convert dataset (alpaca-cleaned -> training format) on the fly.
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
            comp = d.get("complexity") or d.get("expected_complexity") or "light"
            ttype = d.get("task_type") or "other"
            fout.write(
                json.dumps(
                    {"prompt": d["prompt"], "target_response": tgt, "task_type": ttype, "complexity": comp}
                )
                + "\n"
            )
            n += 1
    print(f"[data] converted {n} samples -> {converted_path}")

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
    losses_log: list[dict] = []
    t_start = time.perf_counter()
    while step < cfg.max_steps:
        # Build a batch by sampling B distinct rows.
        idxs = np.random.choice(len(train_data), size=cfg.batch_size, replace=False).tolist()
        samples = [train_data[i] for i in idxs]
        total, parts = step_loss_aux(edge, adapter, samples, device, weights)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optim.step()
        optim.zero_grad()
        step += 1
        if step % cfg.log_every == 0 or step <= 2:
            line = " ".join(f"{k}={v.item():.3f}" for k, v in parts.items())
            print(f"step={step} total={total.item():.3f} {line}")
        losses_log.append({k: v.item() for k, v in parts.items()})
        if step % cfg.save_every == 0:
            ckpt = {
                "step": step,
                "projection_state": edge.projection.state_dict(),
                "mlp_state": adapter.mlp.state_dict(),
                "config": cfg.__dict__,
                "weights": weights,
            }
            torch.save(ckpt, out_dir / f"ckpt-step{step}.pt")
            print(f"checkpoint -> {out_dir}/ckpt-step{step}.pt")

    final = {
        "step": step,
        "projection_state": edge.projection.state_dict(),
        "mlp_state": adapter.mlp.state_dict(),
        "config": cfg.__dict__,
        "weights": weights,
        "losses_per_step": losses_log,
    }
    torch.save(final, out_dir / "final.pt")
    elapsed = time.perf_counter() - t_start
    print(f"[done] {elapsed:.1f}s -> {out_dir}/final.pt")


if __name__ == "__main__":
    main()
