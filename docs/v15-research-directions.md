# V1.5 IR router — paper-worthy research directions

Concrete expansions of the 7 paper-worthy threads from the
2026-05-09 brainstorm. Each direction sketches: experimental setup,
success criteria, engineering effort, key risks, and prior work to
cite. Ordered by what builds on the current 1B/1B Direction A
baseline.

## 1 · V2.5 VQ-VAE codebook (highest paper ROI)

**Hypothesis** — the K continuous soft-prompt vectors can be quantized
to K discrete codes from a learned codebook with negligible quality
loss, while shrinking wire payload from `K × 1152 × 2 B = 18 KB` to
`K × 2 B = 16 B` (with K=8). That's a ~1000× wire-side reduction on
top of the 70× we already have over the buggy baseline.

**Setup**

- Add a `VectorQuantizer` module after `SoftPromptAdapter.mlp`:
  codebook of 4096 entries × `cloud_hidden`. Use exponential-moving-
  average updates per van den Oord 2017.
- Joint training: existing CE/KL/contrastive losses + VQ commitment
  loss `‖sg(z_e) − e_k‖²` + codebook loss `‖z_e − sg(e_k)‖²`.
- Wire format change: schema_v15.embedding_b64 becomes
  `code_indices_b64` carrying `K` int16s. Cloud reconstructs via
  codebook lookup before the MLP→soft-prompt projection.
- Mitigate codebook collapse: dead-code revival (re-init unused codes
  to high-loss inputs every N steps), Gumbel-softmax ablation as a
  fallback.

**Success criteria**

- ≤2 % loss on token-CE vs continuous-IR baseline
- Codebook utilization ≥ 60 % (most codes used)
- Per-code interpretability: cluster prompts by code, label clusters
  ("translate", "math", "factual recall", ...). The interpretability
  story is the *headline figure*.

**Effort** — 1 week spike, 2 weeks for clean implementation + paper-
ready ablations.

**Prior work** — VQ-VAE (van den Oord et al. 2017), VQ-VAE-2 (Razavi
et al. 2019), Coconut (Hao et al. 2024) for latent-space reasoning,
DALL·E 1's discrete tokens.

**Why first** — clean self-contained extension of the current arch.
Direct paper hook ("compressed soft prompts via VQ"). Engineering
risk low; we already have all the moving parts.

## 2 · Latent reasoning chain (Coconut-style)

**Hypothesis** — multi-step reasoning (CoT) can be done in IR latent
space, with the cloud LM iterating on edge-produced "thought
embeddings" instead of token-level chain-of-thought.

**Setup**

- Edge produces a *sequence* of soft-prompt blocks `step_1, step_2,
  step_3`, each conditioned on the prompt + prior steps.
- Cloud executes step-by-step: at step `t`, take `step_t` as soft
  prompt + already-generated text, produce intermediate output,
  encode that output back into `step_{t+1}` (or feed forward).
- Training: per-step CE on intermediate targets, plus a final-answer
  CE. Curriculum: start with 1 step, scale to 4–8.
- Eval on GSM8K (math reasoning) and StrategyQA (multi-hop).

**Success criteria**

- Match or beat token-CoT on GSM8K with 1/3 the wire bandwidth
- Show that the latent steps are non-degenerate (each one moves the
  prediction)
- Edge can produce step plans without cloud roundtrip (one cloud
  pass for whole chain)

**Effort** — 2–3 weeks. Needs an instrumented reasoning dataset.

**Prior work** — Coconut (Hao et al. 2024, arXiv:2412.06769),
Quiet-STaR (Zelikman et al. 2024), Looped Transformers (Giannou et
al. 2023).

**Risks** — Coconut training is fiddly (continuous thought tokens fed
back as inputs requires per-layer attention masks). Get the simple
1-step latent CoT working first.

## 3 · Cross-tokenizer IR (almost free with current arch)

**Hypothesis** — V1.5's MLP bridge already projects edge-hidden-state
into cloud-hidden-space; nothing in the architecture cares about the
underlying tokenizers. Empirical question: does soft-prompt fidelity
hold when edge tokenizer ≠ cloud tokenizer?

**Setup**

- Configurations to test:
  - edge `gemma-3-1b-it`  +  cloud `Qwen2.5-3B-Instruct` (different
    tokenizer family)
  - edge `Llama-3.2-1B`  +  cloud `gemma-3-12b-it` (cross-family)
  - edge `Phi-3.5-mini`  +  cloud `gemma-3-4b-it`
- Train MLP per pair on shared distilled dataset.
- Bench against same-family baseline.

**Success criteria**

- Cross-tokenizer trained MLP gets to within 10 % of same-family CE
  loss
- Generation quality (judge score) within 15 % of same-family
- Demonstrates polyglot edge-cloud deployment as a first-class
  capability

**Effort** — 1 week. The arch already supports this; only training
and benching needed.

**Prior work** — mBERT cross-lingual transfer, NLLB tokenizer-
agnostic embeddings, BLIP-2's tokenizer-blind Q-Former.

**Why early** — fastest paper figure. One smoke run could yield a
top-of-funnel headline ("V1.5 enables polyglot edge").

## 4 · Multimodal IR (vision via text-only cloud)

**Hypothesis** — edge VLM perceives image; soft prompt carries
visual content; text-only cloud LM produces the answer.

**Setup**

- Edge: a VLM (Gemma 3 4B is multimodal, also SmolVLM-2B, Qwen2-VL-2B,
  LLaVA-1.6-7B at the limit).
- Take last-token hidden state from the VLM after image+text fusion.
- Project via MLP to soft prompt (may need K=32–128 to carry visual
  info; vision encoders typically use 256+ patch tokens).
- Cloud: a text-only Gemma/Qwen with no vision component.
- Eval: VQAv2, OK-VQA, MMBench (subset).

**Success criteria**

- Beat random baseline (~25 % accuracy) by a wide margin on VQAv2
- Within 30 % of native VLM cloud (which would be ~70 % vs. 50 % is
  acceptable proof-of-concept)
- Cost win: text-only cloud is 5–10× cheaper to serve than VLM cloud

**Effort** — 2–3 weeks. Largest of the seven; needs vision data
plumbing and likely K bumping.

**Prior work** — BLIP-2 (Q-Former 32 query tokens for image), LLaVA
adapter, MiniGPT-4.

## 5 · Speculative decoding via IR

**Hypothesis** — edge produces draft tokens *and* IR; cloud verifies
both in a single forward pass. On accept, no extra generation; on
reject, cloud generates from rejection point with the IR still
available as context.

**Setup**

- Edge: existing 1B LM. Pass prompt → IR + greedy draft of next N
  tokens (N=8–16).
- Wire payload: IR + draft token IDs.
- Cloud: forward pass on `[K_soft, draft_text]`, compute next-token
  log-probs at each draft position, compare to draft token ID. Accept
  while probs > τ; reject at first failure and continue with cloud
  generation from there.
- Two flavors:
  - Token-level (Leviathan-style): standard speculative decoding with
    edge LM as draft model.
  - IR-level: the IR itself is the speculative signal; cloud computes
    "would my output match the IR's intent" at each position.

**Success criteria**

- TTFT cut by ≥40 % vs full-cloud generation
- Mean accepted draft length ≥4 tokens
- Throughput unchanged or improved

**Effort** — 2–3 weeks. Hard part is aligning edge-draft tokenization
with cloud's vocabulary; requires same-family pairs to start.

**Prior work** — Speculative Decoding (Leviathan et al. 2023,
arXiv:2211.17192), Medusa, EAGLE-2, Cascade Speculative Drafting.

## 6 · Quantization-aware IR

**Hypothesis** — the IR can be transmitted as int8 or int4 with
minimal quality loss if MLP is quantization-aware-trained.

**Setup**

- Stage 1 (post-training quant smoke, 2 days): quantize the existing
  fp16 MLP output to int8/int4 at inference time only. Bench. Measure
  quality drop.
- Stage 2 (QAT, 1 week): add fake-quantize ops to the MLP forward
  pass during training. Train through quantization noise.
- Wire format: int4 IR via packed-pair encoding → 1 byte per 2 dims.

**Success criteria**

- int8: ≤1 % CE loss, 2× wire reduction
- int4: ≤3 % CE loss, 4× wire reduction
- Combined with VQ codebook (direction 1): the IR fits in a few dozen
  bytes per request; this is the headline that justifies "edge sends
  a tweet's worth of bytes per inference"

**Effort** — 1 week PTQ smoke. 2–3 weeks for clean QAT + wire
format change.

**Prior work** — QAT (Krishnamoorthi 2018), bitsandbytes-int4
(Dettmers et al. 2023), Q-LoRA (Dettmers et al. 2023).

## 7 · Adversarial robustness

**Hypothesis** — soft prompts are a *new* attack surface for
edge-cloud LLM deployments. A malicious or compromised edge device
can craft IR that bypasses cloud-side safety alignment, leaks data,
or jailbreaks the cloud.

**Threat model**

- **Adversarial-edge**: attacker controls or compromises an edge
  device; trains an MLP to output IR that elicits unsafe outputs from
  a known-frozen cloud.
- **MitM IR-tamper**: attacker intercepts IR in transit and modifies
  it (rules out by signing, but worth considering).
- **Backdoor MLP**: trojan in the released MLP weights triggers on
  specific prompt patterns.

**Attacks to study**

- Universal adversarial soft prompt: a fixed K-token block that
  jailbreaks regardless of prompt content (analog to Wallace 2019
  universal triggers).
- Goal-directed: train MLP that, given any prompt, produces IR that
  outputs `<some target string>` (data exfiltration / brand damage).
- Steganographic: IR encodes hidden message that cloud LM decodes.

**Defenses**

- Cloud-side IR classifier — flags suspicious soft prompts before
  feeding to LM.
- Sandbox alignment — cloud trained with adversarial IR augmentation
  during RLHF.
- Trust anchor — only accept IR cryptographically signed by approved
  edge SKUs.
- Wire-side noise — cloud adds small Gaussian noise to IR; benign IR
  tolerates, adversarial IR loses precision.

**Success criteria**

- Empirical attack-success rate against an unmodified frozen cloud
  with safety RLHF (e.g., Llama-3.1-8B-Instruct).
- Defense efficacy: attack-success-rate reduction from each defense.
- Paper: first systematic threat model of soft-prompt-based attacks
  on edge-cloud LLMs.

**Effort** — 3–4 weeks for paper-quality study. Includes coordination
on responsible disclosure if any attack is novel.

**Prior work** — Universal Adversarial Triggers (Wallace et al.
2019), GCG attacks (Zou et al. 2023), Prompt injection (Greshake et
al. 2023), AutoPrompt (Shin et al. 2020).

**Risks** — ethical / safety. Need to coordinate disclosure with
upstream cloud LM providers if a new class of attack is found.

## Recommended sequencing

| order | direction | why now | weeks |
|---|---|---|---|
| 1 | (3) cross-tokenizer | almost free, fastest paper figure | 1 |
| 2 | (1) VQ-VAE codebook | highest paper ROI, builds on V1.5 | 2 |
| 3 | (6) quantized IR | engineering win, multiplies on (1) | 1–3 |
| 4 | (2) latent reasoning | hot research area, harder | 2–3 |
| 5 | (5) speculative decoding | production value, needs same-family | 2–3 |
| 6 | (4) multimodal IR | high impact, biggest engineering | 2–3 |
| 7 | (7) adversarial security | important, last because depends on a stable system | 3–4 |

(3) and (1) are the two-track recommendation: (3) yields a quick
result that demonstrates V1.5 is architecture-agnostic; (1) is the
substantial follow-up paper hook. (6) compounds (1) for a clean
"compress to <100 bytes" headline.

The other four are full follow-up papers in their own right.
