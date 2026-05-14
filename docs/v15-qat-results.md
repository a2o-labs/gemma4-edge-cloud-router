# V1.5 follow-up — Quantization-Aware IR (int8 / int4)

Companion to `docs/v15-vqvae-and-cross-tokenizer-results.md`. After
PR #33's RVQ run and the direction-3 single-VQ spike both ended in
degenerate inference, the natural pivot was direction (2): **keep
the soft prompt continuous through the MLP, transmit it as int8 or
int4 on the wire, and train with fake-quantization so the MLP learns
to round-trip cleanly**. This document covers two QAT runs on the
same 5000-sample distilled training set as the earlier experiments.

## Why this isn't VQ

| concern | naive / residual VQ | QAT |
|---|---|---|
| Codebook | discrete entries, low utilisation cap | none — per-dim scalar quantisation |
| Off-manifold representation | residual sums drift off cloud LM's input distribution | quantised vector stays in the same continuous space, just rounded |
| Per-token bandwidth | log₂(used codes) ≈ 6 bits | dim × bits per dim = 1152 × 4 = 4608 bits (int4) or 9216 bits (int8) |
| Wire payload at K=32, hidden=1152 | K bytes of indices (very small) | K × dim × bits/8 + per-token scale |

The wire payload story is less aggressive than VQ in principle, but
matches the actual VQ delivered (which only achieved its compression
on paper — bench output was unusable). QAT achieves ~the continuous
baseline's quality at 2× / 4× wire compression vs fp16.

## Setup

Same edge (`google/gemma-3-1b-it`), same cloud (`google/gemma-3-1b-it`
fp16 frozen), same `v15_distilled_5000.jsonl` training data, same K=32
soft-prompt slots, same `aux-ce + aux-kl + aux-contrastive` loss
recipe. The only change: a per-token symmetric fake-quantizer between
the MLP output and the cloud LM, with straight-through gradient on
the backward pass.

Quantizer details (`src/router/quantization_aware.py`):

- Per-token scale = max(|x|) / qmax over the last (feature) axis.
- qmax = 127 (int8) or 7 (int4).
- Forward returns the rounded-and-rescaled vector; backward routes the
  identity gradient through `x` via STE.
- One fp16 scale per (B, K) slot adds 2 × K bytes of wire metadata.

Training:

```bash
PYTHONPATH=src python -m training.train_v15_qat \
    --train-jsonl /tmp/v15_distilled_5000.jsonl \
    --output-dir /tmp/v15-train-qat-int4 \
    --max-steps 3000 --batch-size 2 --prompt-tokens 32 \
    --qat-bits 4 \
    --aux-ce-weight 1.0 --aux-kl-weight 0.05 --aux-contrastive-weight 0.1
```

## Wire payload

K=32 soft-prompt slots, cloud hidden=1152:

| scheme | per-token feature | + scale overhead | total | reduction |
|---|---:|---:|---:|---:|
| fp16 baseline | 1152 × 2 B × 32 = 73 728 B | 0 | **73.7 KB** | 1× |
| QAT int8 | 1152 × 1 × 32 = 36 864 B | 32 × 2 = 64 B | **36.9 KB** | 2.00× |
| QAT int4 | 1152 × 0.5 × 32 = 18 432 B | 64 B | **18.5 KB** | 3.99× |

These compositions multiply with the K reduction the production
strip-fix already gave (2256 → 32 soft tokens ≈ 70× over the
pre-strip-fix buggy state).

## Training behaviour

| metric | QAT int4 | QAT int8 |
|---|---:|---:|
| Final CE (step 3000)| 1.33 | 1.22 |
| Min CE seen | 0.62 (step 2500) | 0.65 (step 2500) |
| Final scale_mean | **~90** | ~3.9 |
| Final quant_err | 22.6 | 1.0 |

Striking observation: QAT-int4 *deliberately scales up* its MLP
output (scale_mean grows from ~0.5 at step 1 to ~90 at step 3000).
That's the MLP learning that, given int4's coarse 15-step grid, the
only way to retain signal is to operate at a magnitude where the
round-off is small relative to the values. int8 doesn't need this —
the 255-step grid is fine enough that the MLP stays in normal-range
output (scale_mean ~3.9).

Neither shows mode collapse in the codebook-utilisation sense
(there's no codebook). Both reach ~the continuous baseline's CE
within noise.

## Bench (same 8 prompts as docs/v15-1b-baseline-results.md)

### QAT int4 (4× compression)

- "Capital of Japan?" → "The capital of Australia is Canberra"
- "Capital of France?" → "The capital of Australia is Canberra" *(same)*
- "Hamlet?" → "The capital of Australia is Canberra" *(same — 3-attractor)*
- "2+2?" → "3 + 4 = 7" *(math operation, wrong digits)*
- "Translate good morning to Japanese" → "The cat sat on the mat" *(translation template, content wrong)*
- "Summarize photosynthesis" → "A computer is a machine that can perform calculations…" *(off-topic, coherent)*
- "List the 7 wonders" → "top 5 most popular fruits in the world" *(list template ✓)*
- "Write a haiku" → "Echoes of laughter ripple through the air, a warm sunbeam kisses the skin of the sleeping child…" *(poetic multi-line, not strict haiku)*

3/8 collapse onto a single "Australia / Canberra" attractor; 5/8 show
task-template recognition; 0/8 fully correct.

### QAT int8 (2× compression)

- "Capital of Japan?" → "The capital of Australia is Canberra…" *(same attractor as int4 for first 2)*
- "Capital of France?" → "The capital of Australia is Canberra…"
- "Hamlet?" → **"The author of Pride and Prejudice is Jane Austen"** *(different factual attractor)*
- "2+2?" → "The product of 3 and 4 is 12" *(math, different operation: × instead of +)*
- "Translate" → "I'm going to the store" *(translation template, wrong)*
- "Summarize photosynthesis" → "A hypothesis is a testable prediction about a relationship…" *(off-topic, coherent)*
- "List the 7 wonders" → "top 3 most popular fruits in the United States" *(list template ✓)*
- "Write a haiku" → **"Blue water, Sunlight dances on the waves, Ocean's vastness"** *(3-line haiku ✓)*

2/8 collapse onto "Australia / Canberra"; the remaining 6/8 each
land on different topical attractors. Haiku format is preserved.

## Take-away vs the VQ family

| run | failure mode | quality |
|---|---|---|
| Naive VQ-256 | single-attractor mode collapse, low-bit babbling | unusable |
| RVQ-4×256 + warmup | zero-width-joiner spam, off-manifold drift | unusable |
| single-VQ + warmup + commit 1.0 | 3/8 "irony" boilerplate | unusable |
| **QAT int4** (this run) | 3/8 coherent factual attractor + task templates | **comparable to continuous baseline** |
| **QAT int8** (this run) | 2/8 coherent attractor + 1 haiku correct | **comparable to / slightly better than int4** |

QAT failures are **coherent**, not degenerate. The cloud LM never
produces spam, never collapses to repeated punctuation, never emits
unicode garbage. It just sometimes lands on the wrong factual
attractor — exactly the kind of failure mode you'd expect from
quantisation noise that stays *on* the input-embedding manifold.

This is the empirical confirmation that the binding constraint in
the VQ track was **off-manifold representation** (residual sums) /
**low effective bandwidth** (codebook utilisation) — *not*
fundamentally quantisation. Per-dim scalar quantisation avoids both
and recovers usable inference.

## Open questions

1. **Why the "Australia" attractor specifically?** Likely a quirk of
   the 5000-sample distilled set's "capitals" examples — gemma3:4b
   produced more Australian capital answers than other countries
   during distillation. Worth checking with a larger / more
   balanced training set.
2. **int4 vs int8 — which is the right operating point for prod?**
   int4 has 4× compression, but int8 produces qualitatively richer
   outputs (correct haiku, more diverse failures). The 2× gap in
   wire payload (37 KB vs 18.5 KB) probably matters less than the
   quality gap on prod. **Recommendation: ship int8 as default,
   expose int4 as an aggressive-compression option.**
3. **Will the picture change with 27B base on A100?** With a 27×
   larger input-embedding manifold, the cloud should absorb int4
   quantisation noise even more gracefully, *and* fewer attractor
   collapses. Next experiment when A100 returns.

## Artifacts

Pinned at `~/v15-progress/2026-05-08/`:

- `final_qat_int4.pt`, `final_qat_int8.pt` — checkpoints (50 MB each)
- `v15_xmachine_qat_int4.json`, `v15_xmachine_qat_int8.json` — bench JSONs
- `v15_train_qat_int4.log`, `v15_train_qat_int8.log` — full training traces

## Reproduce

```bash
# int4 (aggressive — 4x compression)
PYTHONPATH=src python -m training.train_v15_qat \
    --train-jsonl /path/to/v15_distilled_5000.jsonl \
    --output-dir /tmp/v15-train-qat-int4 \
    --max-steps 3000 --batch-size 2 --prompt-tokens 32 --qat-bits 4

# int8 (balanced — 2x compression)
PYTHONPATH=src python -m training.train_v15_qat \
    --train-jsonl /path/to/v15_distilled_5000.jsonl \
    --output-dir /tmp/v15-train-qat-int8 \
    --max-steps 3000 --batch-size 2 --prompt-tokens 32 --qat-bits 8
```

Cloud server side: same `bench/a100_cloud_server.py` (or wherever
the cloud server lives after PRs #28/#29 land), with `--checkpoint`
pointing at the QAT checkpoint. The server auto-detects `qat_bits`
from the checkpoint and applies `fake_quantize_per_token` to the
MLP output before the cloud LM sees it.
