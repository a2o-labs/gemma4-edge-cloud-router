"""V1.5 training entrypoint: edge encoder + cloud adapter joint training.

Skeleton only. No actual training execution. Hyperparameters are placeholders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

# accelerate / peft / transformers imports are kept inside main so this file
# imports cleanly without those heavy deps installed.


@dataclass
class TrainConfig:
    train_jsonl: str
    eval_jsonl: str | None = None
    output_dir: str = "outputs/v15"
    learning_rate: float = 2e-5
    batch_size: int = 4
    grad_accum: int = 8
    epochs: int = 1
    max_seq_len: int = 2048
    aux_json_loss_weight: float = 0.1
    seed: int = 7
    edge_model_id: str = "google/gemma-4-26b-a4b"
    cloud_model_id: str = "google/gemma-4-31b"
    bf16: bool = True
    log_steps: int = 50
    save_steps: int = 500


def iter_pairs(path: str) -> Iterator[Tuple[str, str, str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield (
                row["input_text"],
                row["target_text"],
                row.get("task_type", "other"),
                row.get("complexity", "light"),
            )


def build_models(cfg: TrainConfig):
    # TODO: from router.edge_encoder import EdgeEncoder, EdgeEncoderConfig
    # TODO: from router.cloud_adapter import CloudAdapter, CloudAdapterConfig
    # TODO: from peft import LoraConfig, get_peft_model
    # TODO: edge = EdgeEncoder(EdgeEncoderConfig(base_model_id=cfg.edge_model_id))
    # TODO: cloud = CloudAdapter(CloudAdapterConfig(base_model_id=cfg.cloud_model_id))
    return None, None


def loss_fn(cloud_out_logits, target_ids, json_validity_score, aux_weight: float):
    # TODO: lm = F.cross_entropy(cloud_out_logits.view(-1, V), target_ids.view(-1))
    # TODO: aux = 1.0 - json_validity_score  # 0 when valid JSON
    # TODO: return lm + aux_weight * aux
    return None


def train(cfg: TrainConfig) -> None:
    # TODO: from accelerate import Accelerator
    # TODO: accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum, mixed_precision="bf16" if cfg.bf16 else "no")
    edge, cloud = build_models(cfg)
    pairs = list(iter_pairs(cfg.train_jsonl))
    print(
        f"TODO real training: would load {len(pairs)} pairs, lr={cfg.learning_rate}, "
        f"bs={cfg.batch_size}x{cfg.grad_accum}, epochs={cfg.epochs}"
    )
    # TODO: hyperparameter tuning
    # TODO: optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, params), lr=cfg.learning_rate)
    # TODO: for epoch in range(cfg.epochs): for batch in dataloader: ...


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl")
    p.add_argument("--output-dir", default="outputs/v15")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--aux-json-loss-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    return TrainConfig(
        train_jsonl=a.train_jsonl,
        eval_jsonl=a.eval_jsonl,
        output_dir=a.output_dir,
        learning_rate=a.learning_rate,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        epochs=a.epochs,
        max_seq_len=a.max_seq_len,
        aux_json_loss_weight=a.aux_json_loss_weight,
        seed=a.seed,
    )


if __name__ == "__main__":
    train(parse_args())
