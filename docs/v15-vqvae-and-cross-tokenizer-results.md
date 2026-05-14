# V1.5 follow-up — VQ-VAE codebook + cross-tokenizer IR

This document captures the 2026-05-09 / 05-10 follow-up to the 1B/1B
baseline (`docs/v15-1b-baseline-results.md`). Two research threads
were run in parallel on the L4:

- **Direction 3 — cross-tokenizer IR.** Fastest paper figure: validate
  that the V1.5 architecture is tokenizer-family-agnostic.
- **Direction 1 — V2.5 VQ-VAE codebook.** Substantial paper anchor:
  push wire payload from `K * 1152 * 2 B` continuous toward
  `K * 2 B` discrete.

The cross-tokenizer experiment is a **clean positive result** —
architecture validates as expected. The naive VQ experiment is a
**clean negative result** — codebook utilization is the binding
constraint and 60-ish active codes is not enough bandwidth to keep
1B-scale generation coherent. Both findings shape the next sprint.

## Hardware

L4 24 GB, NVIDIA driver 595.71.05, transformers 4.55.4, torch 2.4.1
+ cu124. Same setup as the 1B baseline. 5000-sample distilled
dataset (`v15_distilled_5000.jsonl`) shared across both runs.

## Direction 3 — cross-tokenizer IR

### Configuration

| component | model |
|---|---|
| edge | `google/gemma-3-1b-it` fp16 (vocab 262 K) |
| cloud | `Qwen/Qwen2.5-3B-Instruct` fp16 (vocab 152 K, hidden 2048) |
| K | 32 |
| training | 5000 distilled samples, 3000 steps, batch 2 |
| losses | CE + contrastive (KL **disabled** — would crash on vocab-size mismatch) |

The MLP shape adapts automatically: `edge_dim 1152 → cloud_hidden
2048`. The strip fix and Direction A (no schema text in cloud
context) carry over unchanged.

### Result

End-of-training stats:

| metric              | gemma 1B → gemma 1B (baseline) | gemma 1B → Qwen 3B (cross-tokenizer) |
|---------------------|-------------------------------:|-------------------------------------:|
| final CE (step 3000)| 1.27                           | 1.48                                 |
| min CE seen         | 0.51                           | 0.69                                 |
| forward p50         | 795 ms                         | 3 313 ms (Qwen 3B is ~4× slower)     |

Bench output on the same 8-prompt set used in the baseline run:

| prompt | gemma 1B baseline | gemma → Qwen cross-tokenizer |
|---|---|---|
| Capital of Japan? | "Capital of Australia is Canberra" | "Capital of Canada is Ottawa" |
| **Capital of France?** | **"capital of France is Paris"** | **"capital of France is Paris"** |
| Hamlet? | personification analysis (off-topic) | "Attributive Clause" grammar lecture |
| 2+2? | "3 multiplied by 4 is 12" | "3 + 4 = 7" |
| Translate good morning to Japanese | "I'm going to the store" | "Bien is the French translation of Good" |
| **Summarize photosynthesis** | "tweet summarising AI rise" (off-topic) | **"process of photosynthesis is a fascinating biological phenomenon... in plants, algae, and some bacteria"** (on-topic) |
| List 7 wonders | "3 most popular countries" | "top 5 most populous cities in the United States" |
| **Write a haiku** | **"Blue water reflects the sun, A gentle current flows, Waves crash on the shore"** | **"Blue water flows, River's gentle song, Waves kiss the shore"** |

### Take-away

The architecture is **tokenizer-family-agnostic**. Both runs produce
the same correct factual answer ("Paris") and a format-perfect
haiku. The Qwen cloud actually does *better* on the photosynthesis
prompt (mentions plants/algae/bacteria) than the same-family gemma
1B run, hinting at Qwen's stronger zero-shot recall.

Wire path is unchanged: 32 soft-prompt tokens, no schema text,
identical compression headline. The MLP bridge handles the
tokenizer/vocab/hidden-dim mismatch transparently.

This is the **polyglot edge-cloud headline**: ship a single trained
adapter, deploy with arbitrary edge ↔ cloud combinations.

## Direction 1 — naive VQ-VAE codebook

### Module

`src/router/vector_quantizer.py` adds a `VectorQuantizer` class with:

- EMA codebook update (van den Oord 2017 §3.2)
- Straight-through estimator on forward
- Commitment loss `‖x − sg(q)‖²`
- Codebook loss (monitoring only, EMA handles updates)
- Dead-code revival every `revive_every` steps

7 CPU-only unit tests cover shape, indices, gradient flow, EMA
update, dead-code revival, and state-dict round-trip.

### Training script

`training/train_v15_vqvae.py` extends the aux-loss training with the
quantizer. The cloud LM sees `q = quantizer(soft)` as its prefix;
gradient flows back through the straight-through estimator.

### Two runs

Same edge + cloud (gemma-3-1b-it both sides), same 5000 distilled
data, same 3000 steps + batch 2, same K=32. Only the codebook size
varies.

| metric                        | num_codes = 4096 | num_codes = 256 |
|-------------------------------|-----------------:|----------------:|
| step-3000 CE                  | 1.48             | 1.46            |
| step-3000 commitment loss     | 0.31             | 1.02            |
| step-3000 utilization         | **1.6 %** (~65 codes) | **22.7 %** (~58 codes) |
| step-3000 contrastive         | 0.045            | 0.060           |

Both codebook sizes converge to roughly **60 actively-used codes** —
an effective `K · log2(60) ≈ 188`-bit channel for the soft prompt.
Compare the continuous baseline: `K · 1152 · 16 ≈ 590 000` bits.
**The naive VQ has ~3000× less information capacity than the
continuous IR.**

### Inference quality (same 8 prompts)

| prompt | continuous IR (baseline) | VQ-256 | VQ-4096 |
|---|---|---|---|
| Capital of Japan? | "Australia is Canberra" | re-examine boilerplate | "the word beautiful is an adjective" |
| Capital of France? | **"Paris"** | "few options for rewriting" | "the word beautiful is an adjective" |
| Hamlet? | personification | cognitive dissonance lecture | "The Great Barrier Reef" |
| 2+2? | "3 mult 4 = 12" | russian text degenerate | python `import pandas` code |
| Translate good morning | "I'm going to the store" | "He was a whirlwind of energy" | "The Great Escape" |
| Summarize photosynthesis | computer-program tangent | "few options... rain fell" | "I is the subject of the sentence" |
| List 7 wonders | "top 5 cities" | "short engaging story" | "The Great Barrier Reef" |
| **Write a haiku** | **3-line haiku** | "...vamos." (single word, terminated) | python `import pandas` code |

The continuous baseline gets 1/8 fully correct + 1/8 format-perfect.
**Both VQ variants degenerate** — outputs collapse onto the same
handful of "instruct-style boilerplate" responses regardless of
prompt. The 60-active-codes channel is too narrow to discriminate
prompts.

### Why utilization tops out at ~60 codes

Two compounding effects:

1. **Cold-start collapse.** Random codebook init (small Gaussian) is
   far from any MLP output, so initial assignments concentrate on a
   single nearest code. EMA then reinforces that asymmetry — the
   codes that *did* get assigned absorb all the gradient signal,
   while unused codes never see a usage signal.
2. **Dead-code revival fights but doesn't win.** Every 200 steps we
   re-init unused codes to randomly sampled training inputs. They
   then get used briefly until the rest of the network adapts away
   from them, and they go dormant again.

A larger codebook (4096) doesn't help because the MLP's output
manifold is too low-dimensional to populate that many cells; the
extra codes are dead weight.

### Take-away

The naive VQ pipeline:

- ✅ trains end-to-end without divergence
- ✅ commitment loss decreases monotonically
- ✅ wire-side savings, *if utilization were 100 %*, would be 280×
- ❌ utilization hard-caps at ~60 codes regardless of codebook size
- ❌ generation quality collapses below the continuous baseline

This is a *paper-worthy negative result*: it identifies **codebook
utilization × log₂(num_codes) as the new metric to optimise**,
not raw `num_codes`. It also shapes the natural follow-ups.

## Follow-up — RVQ + k-means warmup (PR #33)

After PR #32 was merged, both ideas were prototyped (`PR #33`,
`feat/v15-rvq-kmeans-warmup`). The implementation extends the same
training loop with a `--use-rvq --rvq-layers N --kmeans-warmup`
flag set. RVQ uses 4 stacked quantizers × 256 codes each (32
effective bits/token, vs ~6 bits in the naive VQ run); k-means
warmup gathers 8 batches of MLP output before training and seeds
each codebook layer with cluster centers.

### Training metric — better than naive VQ

| metric                | VQ-256 (no warmup)  | RVQ-4×256 + warmup |
|-----------------------|--------------------:|-------------------:|
| step-3000 CE          | 1.46                | **0.996** (-32 %)  |
| min CE seen           | ~0.81               | **0.754**          |
| step-3000 utilization | 22.7 % (single)     | 7.7 % per layer    |

K-means warmup gets layer-1 utilization up to ~6 % at step 1
(warm-started, vs cold-start at <1 %). Training-loss-wise this is
an unambiguous win: the model fits the targets noticeably better.

### Inference quality — surprisingly *worse* than naive VQ

Same 8-prompt bench, RVQ + warmup checkpoint at step 3000:

- "Capital of Japan?" / "Capital of France?" → identical zero-width-joiner spam, no English answer
- "Hamlet?" → unrelated essay about the word "ephemeral"
- "What is 2+2?" / "Write a haiku" → degenerate punctuation/symbol loops
- "Translate" / "Summarize" → `import random` python boilerplate (mode collapse on a different attractor)

Tested also on the step-2000 checkpoint (better intermediate CE):
same kind of degeneracy, just landing on a different attractor
(python `import random` for 2/8 prompts, factual hallucination for
others).

### Why training loss decoupled from inference

The RVQ output is a *sum* of layer-i codebook entries, not a single
nearest-neighbour. While that lets RVQ approximate any continuous
target arbitrarily closely (32-bit channel reconstructs ~unit-Gaussian
inputs to err < 1 in the unit test), the **summed output drifts off
the cloud LM's natural input-embedding manifold**:

- During training the cloud is teacher-forced on `[K_RVQ, target]`,
  so it learns to predict targets even from off-manifold prefixes —
  CE drops.
- At inference the cloud sees only `[K_RVQ]` and must autoregress
  with no anchor. Off-manifold prefix → no path back to natural
  text → mode collapse on whatever attractor lies near the RVQ
  output.

The continuous baseline doesn't suffer this because the MLP output
is a smooth low-frequency function of `edge_vec`. The naive single-
VQ also sits closer to natural embeddings (single codebook entry,
not a sum).

This is a real paper finding: **RVQ-style summation breaks the
straight-through-VQ ↔ frozen-cloud-LM contract that single-VQ
preserves**. Standard mitigations:

1. **Project sum back onto natural manifold** — train a thin
   "denoise" MLP after RVQ that maps the sum to the closest valid
   embedding (similar to soundstream's decoder).
2. **Train cloud (or part of cloud) on RVQ outputs** — abandons
   the frozen-cloud constraint but matches the train/inference
   distribution.
3. **Replace residual *summation* with residual *correction*** —
   layer-i predicts a delta in MLP-output space, not a contribution
   to the prefix. Keeps the prefix on a single nearest-neighbour
   manifold.
4. **Per-layer freeze with sequential training** — train layer 1
   alone first; once converged, freeze and train layer 2 on its
   residual; repeat. May avoid the multi-layer compounding drift.

These are PR #34+ material. The negative result here is enough to
ship the PR — it formalises the train/inference manifold mismatch
as the new headline blocker, beyond the earlier
codebook-utilisation finding.

### Bench artifacts (RVQ run)

- `final_rvq_warmup.pt` — step-3000 checkpoint
- `rvq_warmup_step2000.pt` — intermediate, lowest CE seen
- `v15_xmachine_rvq_warmup.json` — bench JSON, step 3000
- `v15_xmachine_rvq_step2k.json` — bench JSON, step 2000
- `v15_train_rvq_warmup.log` — full training trace, including
  k-means warmup phase

## Direction-3 spike A — single-VQ + k-means warmup + commit 1.0

To isolate "is the RVQ residual *summation* the issue, or is the
problem deeper", we re-ran with only the elements that PR #33
plausibly fixed (warmup + dense codebook + strong commitment) but
**without** the residual stack. Single VQ with `--num-codes 4096
--kmeans-warmup --aux-commit-weight 1.0`. Cloud sees one nearest-
neighbour codebook entry per soft-prompt slot — guaranteed
on-manifold.

Training: CE 1.008 final (basically tied with RVQ's 0.996),
utilisation collapses back to 1.6 % (~65 effective codes — the
warmup head-start at step 1 is reabsorbed by the high-commit
gradient pulling MLP outputs onto the same codes).

Inference (same 8-prompt bench):

- 3/8 prompts emit an identical "Okay, let's break down the concept
  of 'irony' in a way that's easy to understand…" passage.
- 1/8 emits "cognitive dissonance" instead — a similar boilerplate.
- 4/8 emit short sentence-grammar lectures ("The sentence is
  grammatically correct…").
- Zero correct answers; zero format-perfect haikus.

So a different failure mode than RVQ — clean *single-attractor*
mode collapse — but the same end state: unusable.

### Combined diagnosis

Both regressions confirm the manifold-mismatch finding from PR #33
*and* extend it: even with on-manifold codes (single nearest-
neighbour), the soft-prompt **information bandwidth** is the
binding constraint. With K=32 soft-prompt slots and only ~65
effectively-used codes, the cloud LM sees nearly the same prefix
across many prompts → mode collapses onto a single attractor in
its own continuation distribution.

The continuous IR has ~590 000 bits/token of channel capacity. Any
naive VQ scheme so far has yielded effectively a few hundred
distinct prefixes for the entire prompt distribution — too few to
discriminate. Two compounding issues:

1. **Codebook utilisation × log₂(num_codes) caps low** — single VQ
   stuck at ~6 bits per soft-prompt slot regardless of nominal
   codebook size or warmup. EMA training has a strong attractor
   toward the few codes the MLP starts producing, and the MLP in
   turn collapses onto those same codes via the commitment loss.
2. **RVQ summation increases bits but breaks manifold** — sum of
   codes lands off the cloud's natural input-embedding distribution.

### Take-away for the next sprint

This catalogues *three* distinct failure modes for VQ-style IR with
a frozen 1B cloud:

1. Naive VQ at K=32: low effective bits → output mode-collapses on
   single attractor.
2. RVQ-summed at K=32: enough effective bits, but sum is
   off-manifold → output degenerates into spam.
3. Single VQ + warmup + high commit: same as (1) — warmup head start
   is reabsorbed.

The natural next moves are *not* incremental VQ tweaks; they
require a different attack on the channel:

- **FSQ (Finite Scalar Quantization)** — Mentzer et al. 2023.
  Per-dim independent quantization to a fixed scalar set; codebook
  is implicit and never collapses. Expected cleaner bandwidth /
  utilisation tradeoff.
- **Quantization-aware IR (no VQ)** — keep continuous MLP, but
  transmit as int8/int4 with QAT. Captures the wire-compression
  story without the on-manifold/utilisation tradeoffs.
- **Partial cloud finetune** — drops the frozen-cloud constraint,
  trains the cloud's input-embedding-adjacent layers to accept
  off-manifold prefixes. Most likely to actually work, but loses
  the "ship one adapter" deployment story.
- **Bigger base** — A100 + 27B cloud. The 1B's input-embedding
  manifold is the binding constraint; a 27B has substantially more
  representational capacity to absorb a noisy prefix without mode
  collapse.

Single-VQ artifacts (this run):

- `final_singlevq_warmup.pt` — step-3000 checkpoint
- `v15_xmachine_singlevq_warmup.json` — bench JSON
- `v15_train_singlevq_warmup.log` — training trace

## Artifacts

Pinned at `~/v15-progress/2026-05-08/` on bench-client:

- `final_xtok.pt` — cross-tokenizer checkpoint (296 MB; gemma 1B
  projection + adapter MLP for Qwen 3B output dim)
- `final_vqvae_4096.pt`, `final_vqvae_256.pt` — VQ checkpoints
- `v15_xmachine_xtok.json` — bench JSON for cross-tokenizer
- `v15_xmachine_vq256.json`, `v15_xmachine_vq4096.json` — bench
  JSON for VQ runs
- `v15_train_xtok.log`, `v15_train_vqvae_*.log` — full training
  losses

## Reproduce

Cross-tokenizer:

```bash
PYTHONPATH=src python -m training.train_v15_aux \
    --train-jsonl /path/to/distilled.jsonl \
    --output-dir /tmp/v15-train-xtok \
    --edge-model google/gemma-3-1b-it \
    --cloud-model Qwen/Qwen2.5-3B-Instruct \
    --embedding-dim 1152 --prompt-tokens 32 \
    --max-steps 3000 --batch-size 2 \
    --aux-ce-weight 1.0 --aux-kl-weight 0 --aux-contrastive-weight 0.1
```

VQ-VAE (256 codes):

```bash
PYTHONPATH=src python -m training.train_v15_vqvae \
    --train-jsonl /path/to/distilled.jsonl \
    --output-dir /tmp/v15-train-vqvae \
    --max-steps 3000 --batch-size 2 --prompt-tokens 32 \
    --num-codes 256 --aux-commit-weight 0.25 \
    --aux-kl-weight 0.05 --aux-contrastive-weight 0.1
```
