# V1.5 IR router — 1B/1B baseline results

This document summarises the 2026-05-08 / 2026-05-09 run that established
the V1.5 router's quality ceiling on the 1B-on-1B configuration. It is
the empirical companion to PRs #28 (bench harness), #29 (strip fix),
and the adaptive routing PR that ships this doc.

## Hardware and model

- **Edge** — `google/gemma-3-1b-it` fp16 on NVIDIA L4 24 GB
  (`l4-edge-2 @ <edge-host>`)
- **Cloud** — same model. Direction A pure-soft inference (`--no-schema-text`)
- **Soft prompt count K** — swept; final = 32
- **Adapter** — `nn.Sequential(Linear, GELU, Linear)`, bf16 trainable
- **Tokenizer** — gemma-3 (shared between edge and cloud)
- **Distillation teacher** — `gemma3:4b` Q4_K_M served by Ollama on the
  same L4 GPU

The full A100 27B config is intentionally out of scope here — it's the
next experiment.

## Stages

### Stage 0 — pre-strip-fix (the bug)

Cloud tokenizer was eating the entire schema JSON, including
`embedding_b64`. Wire-side input was 2256 tokens (~K=8 + 2200 of base64
noise). All 64 generated tokens decoded to `<pad>` because the cloud
context was saturated by the embedding string — the model had no
bandwidth left to produce text.

Captured in `bench/v15_xmachine_bench_2026-05-08.json` (PR #28).

### Stage 1 — strip fix (PR #29)

`SoftPromptAdapter.forward/forward_batch/forward_stream` now strip
`embedding_b64` from `json_text` before tokenizing. Inference moves from
2256 → 92 cloud-input tokens (K=8 + ~84 metadata).

Output stops being `<pad>` — model describes the schema instead, because
the schema metadata is now the only text signal it sees.

### Stage 2 — Direction A: pure soft prompt

Cloud server adds `--no-schema-text`: the cloud sees ONLY the K soft
tokens, no schema text. This matches `step_loss`'s training distribution
(`[K_soft, target_text]`).

After the schema strip, n_in_tokens drops further: 92 → **8** (just K=8).
Output is no longer schema description; it's coherent text generation.
But on the 62-sample untrained MLP, all 5 prompts collapse to the same
"ARPANET history" continuation — the soft prompt is random noise.

### Stage 3 — distill 500 samples + train

500 (prompt, target) pairs distilled from `gemma3:4b` (Ollama) on a random
Alpaca-cleaned subsample. Trained with the existing
`training/train_v15.py` step_loss (CE only, batch=1, bf16, 1500 steps).

| K  | mean of last-20 training losses |
|----|--------------------------------:|
| 4  | 1.14                            |
| 8  | 1.13                            |
| 16 | 1.10                            |
| 32 | **1.01** (winner)               |

K=32 is ~12 % better than K=8 on training fit. Inference: K=32 outputs
are diverse instruct-style text but content unrelated to prompt. Mode
collapse is gone; intent recognition isn't there yet.

### Stage 4 — auxiliary losses (BLIP-2 §3 inspired)

Added two auxiliary losses to `step_loss`:

- **KL distillation** — KL(edge_LM_prob_under_raw_prompt ∥ cloud_prob_under_soft_prompt)
  computed at each target position. Forces the soft prompt to elicit
  per-token distributions that match what the edge LM would produce given
  the actual prompt as text.
- **Contrastive (CLIP-style)** — InfoNCE between `edge_vec[i]` and
  `soft_mean[j]` across a batch of 2. Forces distinct prompts to produce
  distinguishable soft prompts. Counters mode collapse.

Result on 500-sample data, K=32, 1500 steps, KL weight 0.05, contrastive
0.1:

| metric              | K=32 baseline | K=32 + aux losses |
|---------------------|--------------:|------------------:|
| final ce loss       | 1.01          | **0.508** (-50%)  |
| contrastive         | n/a           | 0.014 (converged) |
| forward p50         | 1542 ms       | 1553 ms (=)       |
| n_in_tokens         | 32            | 32                |

Inference qualitatively shifts: outputs start showing **task-type
recognition**. "What is 2+2?" produces math operation text; "translate"
produces French translations; "summarize" produces tweet-style summaries.
Content is still hallucinated; format is improving.

### Stage 5 — full config (final)

Final run: K=32 + CE + KL@0.05 + contrastive@0.1, 5000 distilled
samples, batch=2, 3000 steps. ~1560 s on L4 sharing GPU with the
distillation Ollama process.

Bench output for 8 representative prompts:

| prompt                                         | output preview                                                          | judgment       |
|-----------------------------------------------|-------------------------------------------------------------------------|----------------|
| What is the capital of Japan?                 | "Capital of **Australia** is Canberra"                                  | format ✓ content ✗ |
| **What is the capital of France?**            | **"The capital of France is Paris."**                                   | **fully correct** |
| Who wrote Hamlet?                             | discusses personification                                                | off-topic |
| What is 2+2?                                  | "3 multiplied by 4 is 12"                                                | math op ✓ digits ✗ |
| Translate 'good morning' to Japanese          | "I'm going to the store" (French)                                       | translate task ✓ content ✗ |
| Summarize photosynthesis                      | defines a computer program                                               | off-topic |
| List the 7 wonders of the ancient world       | "3 most popular countries"                                               | list format ✓ |
| **Write a haiku about autumn rain**           | **"Blue water reflects the sun, A gentle current flows, Waves crash on the shore"** | **3-line haiku format ✓** |

### Headlines

- **First fully correct answer** ("Paris") and **first format-perfect
  structured response** (a haiku) on a non-trivial prompt.
- **Most prompts now show task-type recognition.**
- **Forward p50 = 795 ms** — early termination when the model emits
  `<end_of_turn>`, lower than the K=32 unconstrained 1542 ms.
- **n_in_tokens = 32 (constant)**, regardless of original prompt length —
  this is the architectural win that the V2.5 VQ-VAE codebook track
  intends to push to <50 bytes/request.

## Compression headline

| stage                        | n_in_tokens | output quality   |
|-----------------------------|------------:|------------------|
| pre-strip-fix (buggy)       | 2256        | 64×`<pad>`       |
| post-strip-fix (K=8)        | 92          | describes schema |
| pure-soft (K=8, 62 samples) | 8           | mode collapse    |
| pure-soft (K=32, 500 samples) | 32        | diverse, off-topic |
| **K=32 + aux + 5000 (final)** | **32**     | **task-aware, 1/8 correct, 1/8 format-perfect** |

Total wire-input compression vs the buggy state: **2256 → 32 = 70.5×.**

## What this confirms / what it doesn't

**Confirmed**

- The V1.5 IR architecture works in principle: with enough K, enough
  data, and the right loss mix, the soft prompt steers a frozen cloud LM
  to produce task-appropriate output.
- Direction A (no schema text in cloud input) is the right inference
  mode — it matches the training distribution and gives the smallest
  wire footprint.
- Strip fix is a hard requirement — without it the IR is strictly worse
  than V1.0 raw prompt (PR #29).

**Not confirmed**

- Whether V1.5 reaches production-grade quality. The 1B base lacks the
  factual knowledge to *answer* most prompts even given a perfect soft
  prompt. The next experiment is K=32 + aux + 5000 with
  `google/gemma-3-27b-it` 4-bit nf4 cloud on A100 — that probes whether
  the architecture's quality scales with the cloud's recall capacity.

## Reproduce

Artifacts are pinned at `~/v15-progress/2026-05-08/` on bench-client
(`final.pt` 178 MB, `v15_distilled_5000.jsonl`, training log,
five cross-machine bench JSONs). Re-run on a fresh L4:

```bash
# 1) distill 5000 from gemma3:4b
python /tmp/v15_distill_alpaca.py --model gemma3:4b --n-samples 5000 \
    --max-tokens 128 --max-prompt-chars 600 \
    --out /tmp/v15_distilled_5000.jsonl

# 2) final training (K=32 + aux losses)
python /tmp/train_v15_aux.py \
    --train-jsonl /tmp/v15_distilled_5000.jsonl \
    --output-dir /tmp/v15-train-l4-aux-k32-5k \
    --max-steps 3000 --batch-size 2 --prompt-tokens 32 \
    --aux-ce-weight 1.0 --aux-kl-weight 0.05 --aux-contrastive-weight 0.1

# 3) bench
python bench/v15_split_xmachine_bench.py \
    --edge-url http://<L4>:8002 --cloud-url http://<L4>:8003 \
    --max-new-tokens 64 --out /tmp/v15_xmachine_final.json
```

Server launches: `bench/l4_edge_server.py --checkpoint <ckpt>`
and `bench/a100_cloud_server.py --no-schema-text --no-quant
--prompt-tokens 32 --checkpoint <ckpt>`.

## Open questions for the 27B follow-up

1. Does aux loss still help when the cloud has stronger zero-shot
   recall, or does CE-only suffice?
2. Is K=32 still the sweet spot at 27B, or does the optimum drift?
3. With 27B, does `What is 2+2?` get to `4` and `Capital of Japan?` to
   `Tokyo`?
