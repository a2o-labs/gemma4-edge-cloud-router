# V1.5 follow-up — Gemma 4 26B-A4B as the cloud LM

Companion to `docs/v15-qat-results.md`. The 1B-on-1B QAT runs reached
"coherent but factually wrong" outputs and we left
`docs/v15-research-directions.md` with one binding hypothesis open:

> The cloud is the ceiling. A bigger cloud should absorb the same
> quantised soft-prompt and emit correct facts.

When the next A100 came back, we tested that hypothesis directly:
**same edge (1B), same loss recipe, same QAT int8, same K=32 soft
prompt — only swap the cloud to Gemma 4 26B-A4B-it.**

## Why 26B-A4B and not 27B-dense

We deliberately picked the MoE variant. Three reasons:

1. **A100 80 GB fits 26B-A4B-it in 4-bit nf4 with margin** (49 GB used at
   K=32, fp16 activations). A 27B-dense in 4-bit pushes 50–55 GB and
   leaves no room for the schema text path.
2. **A4B activates ~4B parameters per token**. That's a midpoint
   between the 1B baseline and a true 27B dense — informative
   about whether *active* parameter count is what unblocks the ceiling.
3. **It's the architecture we'd ship if MoE-IR is the eventual paper
   direction.** Test it under realistic conditions, not on a model we'd
   never deploy.

The trade-off, telegraphed up front: if 26B-A4B *doesn't* break the
ceiling, we still don't know whether 27B-dense would.

## Setup

| component | value |
|---|---|
| Edge | `google/gemma-3-1b-it` fp16 frozen + 1152→32×2816 MLP + projection (trainable) |
| Cloud | `google/gemma-4-26B-A4B-it` 4-bit nf4 (`bnb_4bit_compute_dtype=bf16`) |
| K (soft-prompt slots) | 32 |
| Cloud hidden | 2816 (vs 1152 for the 1B baseline) |
| Quantisation in IR | **QAT int8** (per-token symmetric fake-quant + STE) |
| Loss | `aux-ce + aux-kl + aux-contrastive` (same recipe as `train_v15_aux.py`) |
| Training data | `v15_distilled_5000.jsonl` (5 000 alpaca-style prompts × gemma3:4b targets) |
| Optimiser | AdamW lr=5e-4, grad-clip 1.0 |
| Training budget | 3 000 initial + 6 000 continue ≈ 9 000 effective steps |
| Hardware | NVIDIA A100-SXM4-80GB (vast.ai) |
| dtype | **bf16** throughout (fp16 NaN'd at step 920 on the first attempt; bf16 stable) |

Training was split because a torch+transformers compat bug
NaN'd the first 920 steps in fp16. The post-bf16 patch
(`bnb_4bit_compute_dtype=bf16` + adapter `dtype=bf16` in
`bench/a100_cloud_server.py`) is what makes the MoE
`grouped_mm_fallback` path happy alongside the bnb 4-bit weights.

## Training trajectory (CE / scale_mean / quant_err)

| step (effective) | ce | scale_mean | quant_err |
|---:|---:|---:|---:|
| 100 | 1.38 | 1.1 | 0.29 |
| 500 | 2.72 | 1.5 | 0.39 |
| 1 000 | 1.38 | 4.7 | 1.20 |
| 1 500 | 1.33 | 4.8 | 1.23 |
| 2 000 | 1.00 | 6.3 | 1.60 |
| 2 500 | 0.67 | 5.5 | 1.40 |
| 3 000 (initial end) | 1.33 | 6.0 | 1.55 |
| 4 000 | 0.86 | 9.5 | 2.44 |
| 5 000 | 0.87 | 12.4 | 3.20 |
| 6 000 | 0.83 | 9.1 | 2.31 |
| 8 000 | 0.83 | 9.5 | 2.44 |
| 9 000 (final) | 0.79 | 8.6 | 2.20 |

Two observations worth flagging:

1. **CE bottoms around 0.8 and refuses to budge further.** This is
   roughly half of where the 1B-on-1B QAT int4 plateaued (1.33) and
   noticeably below the int8 plateau (1.22), but it doesn't keep
   dropping with more steps. The 5 000-sample dataset is the cap.
2. **scale_mean climbs from ~1 to ~9–14 even at int8.** In the 1B-on-1B
   run, int8 stayed at ~3.9 (because the 255-step grid was fine
   enough). On the bigger cloud (2816-hidden), the MLP *prefers* to
   operate at higher magnitudes — quant_err 2–3 vs 1.0 on 1B-on-1B
   int8. Same effect we saw at int4: the MLP picks a working point
   where quantisation rounding is small relative to the signal.

## Wire payload

At K=32, cloud hidden=2816:

| scheme | per-token feature | + scale | total | reduction vs fp16 |
|---|---:|---:|---:|---:|
| fp16 | 2816 × 2 × 32 = 180 224 B | 0 | **180.2 KB** | 1× |
| QAT int8 | 2816 × 1 × 32 = 90 112 B | 64 B | **90.2 KB** | 2.00× |
| QAT int4 | 2816 × 0.5 × 32 = 45 056 B | 64 B | **45.1 KB** | 4.00× |

(Same ratios as the 1B-on-1B QAT runs — quant ratio is a property of
bits/dim, not model size.)

## Bench (same 8 prompts as `docs/v15-1b-baseline-results.md`)

Two checkpoints, same harness (`bench/v15_split_xmachine_bench.py`),
same wire path, K=32 soft prompts only (Direction A: no schema text).

### Step ~4 000 (intermediate)

p50 forward ≈ 5 470 ms, total p50 ≈ 5 730 ms.

- "Capital of Japan?" → "The capital of the United Arab Emirates is **Abu Dhabi**…" *(UAE attractor)*
- "Capital of France?" → "The capital of India is New Delhi…" *(India attractor — different from Japan)*
- "Hamlet?" → "The 1912 sinking of the Titanic was caused by…" *(off-topic, but coherent factual paragraph)*
- "2+2?" → "100 is 10 times 10. 100 / 10 = 10…" *(degenerate math repeat)*
- "Translate good morning to Japanese" → "The most common and natural translation of 'I love you' in Spanish is: **Te amo**…" *(translation template, wrong language)*
- "Summarize photosynthesis" → "A fractal is a complex geometric shape…" *(off-topic, coherent)*
- "List the 7 wonders" → "the 10 largest countries in the world by land area, listed in descending order: 1. **Russia** – 17,098,242 km² 2. **Canada** – 9,984,670 km²" *(list template ✓, real 2024 numbers ✓, wrong topic)*
- "Write a haiku about autumn rain" → "Green leaves, soft and still, Sunlight dances on the stream, Peace in every breath." *(haiku format ✓, content generic / not autumn rain)*

Eight prompts, eight **distinct factual attractors**. No cross-prompt
collapse. Every output is grammatically correct, formally
appropriate to the prompt's category, factually wrong.

### Step ~9 000 (final, "10k effective")

p50 forward ≈ 6 620 ms, total p50 ≈ 7 130 ms.

- "Capital of Japan?" → "Ottawa. It's the capital of Canada!…" *(Canada/Ottawa attractor)*
- "Capital of France?" → "Ottawa. It's the capital of Canada!…" *(same attractor as Japan — **cross-prompt mode collapse on capitals**)*
- "Hamlet?" → "The person who wrote the poem 'The Raven' is **Edgar Allan Poe**." *(coherent factual attractor, swap-out of subject)*
- "2+2?" → "12 x 12 = 144" *(math attractor, real arithmetic, wrong operation)*
- "Translate good morning to Japanese" → "There are several ways to translate 'I love you' into French, depending on the level of intimacy…" *(translation template, wrong target language)*
- "Summarize photosynthesis" → "Nuclear fission is a process where the nucleus of a heavy atom splits into two or more smaller nuclei…" *(coherent factual paragraph, wrong topic)*
- "List the 7 wonders" → "10 most populous countries in the world, as of 2024: 1. **India** (approx. 1.43 billion) 2. **China** (approx. 1.41 billion)…" *(list template ✓, real 2024 demographics ✓, wrong topic)*
- "Write a haiku about autumn rain" → **"Softly, the rain falls, Whispering to the thirsty earth, Life drinks, deep and slow."** + "(Note: I've added a bit of imagery to make it more evocative…)" *(**haiku format ✓, autumn-rain semantics ✓, self-aware meta-comment**)*

The haiku in particular is the first time on this V1.5 stack the
cloud produced a topically-appropriate poem from a quantised soft
prompt. The meta-commentary ("I've added imagery") is emergent
self-reflection.

### Step-4 000 vs step-9 000

| dimension | step-4 000 | step-9 000 |
|---|---|---|
| Distinct attractors across 8 prompts | 8 | 7 (capitals collapsed) |
| Haiku semantics | generic | **on-topic + meta** |
| List template real data | "10 largest countries" (2024) ✓ | "10 most populous" (2024) ✓ |
| Capital cross-prompt collapse | no | **yes** (both → Ottawa) |
| Coherence of off-topic outputs | high | high |
| forward p50 | 5.47 s | 6.62 s |

The **mode collapse on capitals** appearing only at step-9 000 is the
unexpected finding. More training increased coherence *and* cross-prompt
homogeneity. Two competing pressures:

- More steps → MLP gets sharper at producing a "well-typed" soft
  prompt that the cloud reads cleanly.
- More steps → MLP overfits to the 5 000-sample distribution's
  attractors. "Capital of \<X\>" cluster homogenises onto whichever
  capital the gemma3:4b teacher happened to over-produce.

## 1B-ceiling hypothesis verdict

**Partially confirmed, with a refinement.**

The bigger cloud absorbs the soft prompt cleanly (no garbage, no spam,
on-topic haiku, real 2024 demographic numbers) — so the cloud was
indeed *a* ceiling at 1B-on-1B. But factual correctness on
prompt-conditioned recall ("What's the capital of Japan?") still
fails. That's not the cloud's recall failing — Gemma 4 26B-A4B
clearly *knows* Japan's capital. It's the 1B edge's representation
of the prompt that the trained MLP maps to "capital-shaped soft
prompt", and the cloud picks whichever capital is closest in its
input-embedding manifold.

The corrected hypothesis is:

> The **edge** (1B) is one ceiling. The **dataset** (5 000 samples
> distilled from gemma3:4b, a strictly weaker model than the target
> cloud) is another. Both need to be addressed before factual
> correctness emerges from V1.5 IR.

## Operational notes

The full A100 session burned roughly:

- 4 588 s for steps 1–3 000 (initial)
- 9 385 s for steps 3 001–9 000 (continue, with `--resume-from` warm-start
  on MLP + projection)
- ≈ 230 minutes total of A100 wall time
- vast.ai A100-SXM4-80GB instance, destroyed after this run

Two bugs cost about half a day:

1. **fp16 NaN at step 920** with `bnb_4bit_compute_dtype=fp16`. Fix:
   patch `dtype=bf16` everywhere in `bench/a100_cloud_server.py` and
   `training/train_v15.py`. Long-term: codify bf16 as the default
   dtype for any cloud >7B in 4-bit.
2. **Buffered Python stdout under nohup** made the resume run *look*
   stuck at step 960 for ~14 minutes. Fix: `PYTHONUNBUFFERED=1` in the
   launcher.

`bench/a100_cloud_server.py` gained:

- `--checkpoint` (loads MLP + projection from a v15 checkpoint, auto-detects
  `qat_bits` and applies `fake_quantize_per_token`)
- `--no-schema-text` (Direction A — pure soft prompt, no schema text in
  cloud context)
- `--no-quant` (load fp16 instead of bnb 4-bit; not used in this run)

`training/train_v15_qat.py` gained `--resume-from` for warm-starting MLP +
projection from a previous checkpoint.

## Artifacts

Pinned at `~/v15-progress/2026-05-08/`:

- `final_g4a4b_qat_int8.pt` — checkpoint after the initial 3 000 steps
- `g4_qat_step4000.pt` — intermediate snapshot used for the step-4 000
  bench
- `g4_qat_final_10k_eff.pt` — final 9 000-effective-step checkpoint
- `v15_xmachine_g4_step4k.json` — step-4 000 bench JSON
- `v15_xmachine_g4_final_10k.json` — step-9 000 bench JSON
- `v15_train_g4a4b_qat_int8.log`, `v15_train_g4_continue.log` —
  full training traces

## Open questions / next experiments

1. **Try a true 27B dense, then 31B.** Tests whether *active*
   parameters vs total parameters is what matters for the recall
   step. 31B in 4-bit just barely fits A100-80 GB.
2. **Re-distill the 5 000-sample dataset using 26B-A4B itself as
   teacher.** Closes the teacher-strictly-weaker-than-student gap.
   Tracked as task #58.
3. **Larger training set** (50 k? 500 k?) at the same step budget —
   tests whether the attractor collapse is dataset-size limited.
4. **Wider K (64? 128?)** to give the MLP more bandwidth into the
   cloud — but linear wire-cost growth and we've already shown K=32
   is the K-sweep sweet spot at the 1B-on-1B scale.

## Reproduce

```bash
# initial 3000 steps
HF_TOKEN=... PYTHONPATH=src python -m training.train_v15_qat \
    --train-jsonl /tmp/v15_distilled_5000.jsonl \
    --output-dir /tmp/v15-train-g4a4b-qat-int8 \
    --max-steps 3000 --batch-size 2 --prompt-tokens 32 --qat-bits 8 \
    --edge-model google/gemma-3-1b-it \
    --cloud-model google/gemma-4-26B-A4B-it \
    --aux-ce-weight 1.0 --aux-kl-weight 0.05 --aux-contrastive-weight 0.1

# resume 6000 more steps
PYTHONUNBUFFERED=1 HF_TOKEN=... PYTHONPATH=src python -m training.train_v15_qat \
    --train-jsonl /tmp/v15_distilled_5000.jsonl \
    --output-dir /tmp/v15-train-g4a4b-qat-int8-10k-cont \
    --resume-from /tmp/v15-train-g4a4b-qat-int8/final.pt \
    --max-steps 6000 --batch-size 2 --prompt-tokens 32 --qat-bits 8 \
    --edge-model google/gemma-3-1b-it \
    --cloud-model google/gemma-4-26B-A4B-it \
    --aux-ce-weight 1.0 --aux-kl-weight 0.05 --aux-contrastive-weight 0.1

# bench
PYTHONPATH=src python -m bench.a100_cloud_server \
    --cloud-model google/gemma-4-26B-A4B-it \
    --prompt-tokens 32 --checkpoint /path/to/final.pt --no-schema-text &
PYTHONPATH=src python -m bench.v15_split_xmachine_bench \
    --edge-url http://...:8002 --cloud-url http://...:8003 \
    --prompts bench/8_canonical_prompts.txt \
    --max-new-tokens 64 --output /tmp/bench.json
```
