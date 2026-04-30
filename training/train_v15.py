"""V1.5 paired-data training loop.

Objective: train EdgeEncoder.projection + SoftPromptAdapter.mlp ONLY.
Both base LMs are frozen (BLIP-2 Q-Former pattern).

Inputs: JSONL of {"prompt": str, "target_response": str, "task_type": str,
                  "complexity": "light"|"heavy"} records.

Loss: standard LM cross-entropy on target_response tokens, computed on
the cloud LM's output logits with the soft-prompt prefix prepended.

Mock mode: V15_MOCK_MODE=true uses tiny-random-LlamaForCausalLM and a
6-sample synthetic dataset, runs 2 steps so CI verifies gradient flow
without GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrainConfig:
    edge_model: str = "google/gemma-4-26B-A4B-it"
    cloud_model: str = "google/gemma-4-31B-it"
    embedding_dim: int = 4096
    prompt_tokens: int = 8
    learning_rate: float = 5e-4
    batch_size: int = 4
    max_steps: int = 1000
    save_every: int = 200
    log_every: int = 10
    output_dir: str = "outputs/v15-train"
    train_jsonl: str = "data/v15/train.jsonl"
    eval_jsonl: str = "data/v15/eval.jsonl"
    mock_mode: bool = False
    device: str = "cuda"
    hf_token: str | None = None
    seed: int = 42
    grad_clip: float = 1.0
    warmup_steps: int = 50


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pipeline(cfg: TrainConfig):
    """Construct EdgeEncoder + SoftPromptAdapter.

    In mock mode swap both for tiny-random-Llama with embedding_dim=64,
    prompt_tokens=4 to keep things fast on CPU.
    """
    import torch

    from router.cloud_adapter import SoftPromptAdapter
    from router.edge_encoder import EdgeEncoder

    if cfg.mock_mode:
        edge_model = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        cloud_model = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        embedding_dim = 64
        prompt_tokens = 4
        device = "cpu"
        dtype = torch.float32
    else:
        edge_model = cfg.edge_model
        cloud_model = cfg.cloud_model
        embedding_dim = cfg.embedding_dim
        prompt_tokens = cfg.prompt_tokens
        device = cfg.device
        dtype = torch.float16

    edge = EdgeEncoder(
        model_name=edge_model,
        embedding_dim=embedding_dim,
        hf_token=cfg.hf_token,
        device=device,
        dtype=dtype,
        cache_size=0,
    )
    adapter = SoftPromptAdapter(
        edge_dim=embedding_dim,
        prompt_tokens=prompt_tokens,
        hf_token=cfg.hf_token,
        cloud_model_name=cloud_model,
        device=device,
        dtype=dtype,
    )
    return edge, adapter, device, dtype


def step_loss(edge, adapter, sample: dict, device):
    """Single-sample LM loss for adapter-only training.

    1. edge._tokenize(prompt) + edge.base(...) -> last hidden state of last token.
       Frozen base parameters mean no gradients accumulate on them; the
       gradient path to ``edge.projection`` flows through the projection's
       own trainable weights.
    2. adapter.mlp(edge_vec) -> soft-prompt embeds (trainable).
    3. tokenizer(target_response) -> target_ids -> embeds via cloud's
       (frozen) input embedding table.
    4. cloud(inputs_embeds=concat(soft, target_embeds)) -> logits.
    5. Cross-entropy over the target portion only (positions K-1..K+T-2
       predict positions K..K+T-1).
    """
    import torch
    import torch.nn.functional as F

    prompt = sample["prompt"]
    target = sample["target_response"]

    edge_input_ids = edge._tokenize(prompt)
    edge_attn = torch.ones_like(edge_input_ids)
    out = edge.base(
        input_ids=edge_input_ids,
        attention_mask=edge_attn,
        output_hidden_states=True,
        use_cache=False,
    )
    last_hidden_last_token = out.hidden_states[-1][:, -1, :]
    edge_vec = edge.projection(last_hidden_last_token)

    tok = adapter.tokenizer
    target_ids = tok(target, return_tensors="pt").input_ids.to(device)
    target_embeds = adapter.cloud.get_input_embeddings()(target_ids)

    soft = adapter.mlp(edge_vec).reshape(
        edge_vec.shape[0], adapter.prompt_tokens, adapter.cloud_hidden
    )

    embeds = torch.cat([soft, target_embeds], dim=1)
    attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)

    cloud_out = adapter.cloud(
        inputs_embeds=embeds,
        attention_mask=attn,
        labels=None,
    )
    logits = cloud_out.logits

    K = adapter.prompt_tokens
    target_logits = logits[:, K - 1 : -1, :]
    target_labels = target_ids
    loss = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_labels.reshape(-1),
    )
    return loss


def train(cfg: TrainConfig):
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    edge, adapter, device, _dtype = build_pipeline(cfg)
    edge.freeze_base()
    adapter.freeze_base()

    params = list(edge.trainable_parameters()) + list(adapter.trainable_parameters())
    optim = torch.optim.AdamW(params, lr=cfg.learning_rate)

    if cfg.mock_mode:
        train_data = [
            {"prompt": "What is 2+2?", "target_response": "4", "complexity": "light"},
            {"prompt": "Capital of France?", "target_response": "Paris", "complexity": "light"},
            {
                "prompt": "Explain entropy briefly",
                "target_response": "Disorder measure",
                "complexity": "heavy",
            },
        ] * 2
    else:
        train_data = load_jsonl(cfg.train_jsonl)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    losses: list[float] = []
    while step < cfg.max_steps and step < len(train_data):
        sample = train_data[step % len(train_data)]
        loss = step_loss(edge, adapter, sample, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        step += 1
        if step % cfg.log_every == 0 or step <= 2:
            print(f"step={step} loss={loss.item():.4f}")
        if step % cfg.save_every == 0:
            ckpt = {
                "step": step,
                "projection_state": edge.projection.state_dict(),
                "mlp_state": adapter.mlp.state_dict(),
                "config": cfg.__dict__,
            }
            torch.save(ckpt, out_dir / f"ckpt-step{step}.pt")
            print(f"checkpoint -> {out_dir}/ckpt-step{step}.pt")

    final = {
        "step": step,
        "projection_state": edge.projection.state_dict(),
        "mlp_state": adapter.mlp.state_dict(),
        "config": cfg.__dict__,
        "losses": losses,
    }
    torch.save(final, out_dir / "final.pt")
    print(f"final -> {out_dir}/final.pt")
    return final


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mock-mode",
        action="store_true",
        default=os.getenv("V15_MOCK_MODE", "false").lower() == "true",
    )
    p.add_argument(
        "--max-steps", type=int, default=int(os.getenv("V15_MAX_STEPS", "1000"))
    )
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-jsonl", default="data/v15/train.jsonl")
    p.add_argument("--output-dir", default="outputs/v15-train")
    p.add_argument(
        "--device", default="cpu" if os.getenv("V15_MOCK_MODE") else "cuda"
    )
    a = p.parse_args()
    cfg = TrainConfig(
        mock_mode=a.mock_mode,
        max_steps=a.max_steps,
        learning_rate=a.learning_rate,
        batch_size=a.batch_size,
        train_jsonl=a.train_jsonl,
        output_dir=a.output_dir,
        device=a.device,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print(f"=== V1.5 training (mock={cfg.mock_mode}) ===")
    train(cfg)
