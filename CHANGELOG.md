# Changelog

All notable changes to gemma4-router are documented here.

## Unreleased

### V1.5 bench + production fixes

- `bench/v15_split_xmachine_bench.py` + `bench/a100_cloud_server.py` +
  `bench/l4_edge_server.py` — cross-machine L4↔A100 V1.5 split bench
  harness, with production-sizing sweeps (length / concurrency / batch).
  First cross-machine end-to-end validation of the schema_v15 wire
  format on real GPUs. (#28)
- `_strip_schema_embedding()` in `src/router/cloud_adapter.py` — drops
  `embedding_b64` from `forward` / `forward_batch` / `forward_stream`
  before the cloud tokenizer sees it. 24.5× input compression vs the
  pre-fix buggy state; unblocks real text output instead of all-`<pad>`.
  +unit tests. (#29)

### V1.5 adaptive routing

- `POST /route/auto` — adaptive V1.0/V1.5 dispatch. Counts the prompt's
  cloud-tokenizer tokens and routes to V1.5 only when prompt > threshold
  (default 92, the empirical V1.0/V1.5 crossover). Gated by
  `V15_ADAPTIVE_ROUTING_ENABLED=true`. New `src/router/adaptive_router.py`
  + 9 unit tests.
- `docs/v15-1b-baseline-results.md` — write-up of the 2026-05-08 / 05-09
  1B/1B baseline run: K sweep (K=4/8/16/32), auxiliary losses
  (CE + KL distillation + contrastive), 5000-sample alpaca→gemma3:4b
  distillation, and the breakthrough on the K=32 + aux + 5k checkpoint
  (first factually-correct answer, first format-perfect haiku). (#31)

### V1.5 follow-up — VQ-VAE codebook + cross-tokenizer

- `src/router/vector_quantizer.py` — `VectorQuantizer` module with
  EMA-updated codebook, straight-through estimator, dead-code revival.
  4096 / 256 codebooks trained on the 1B/1B baseline; both converge
  to ~60 actively-used codes (clean negative result documented in
  `docs/v15-vqvae-and-cross-tokenizer-results.md`). 7 unit tests in
  `tests/test_vector_quantizer.py`.
- `training/train_v15_aux.py` — paper-companion training script with
  CE + KL distillation + contrastive auxiliary losses (BLIP-2 §3
  inspired). Halved final CE on the 1B baseline (1.01 → 0.508).
- `training/train_v15_vqvae.py` — same loop with the vector
  quantizer wired in.
- `docs/v15-vqvae-and-cross-tokenizer-results.md` — write-up of the
  cross-tokenizer experiment (gemma-3-1b-it → Qwen2.5-3B-Instruct,
  validated, factual answer + haiku format) and the VQ-VAE codebook
  failure mode (~60 effectively-used codes regardless of codebook
  size, generation degenerates below continuous baseline). Lists
  Residual VQ + k-means warmup as the natural follow-up.
- `ResidualVectorQuantizer` + `RVQConfig` in
  `src/router/vector_quantizer.py`: stack of N quantizers operating
  on layer-i residuals, effective bandwidth = sum of layer-bits.
- `VectorQuantizer.warmup_kmeans()`: k-means init from a sample of
  representative MLP outputs to defeat cold-start codebook collapse.
- `training/train_v15_vqvae.py` gains `--use-rvq`, `--rvq-layers`,
  `--kmeans-warmup`, `--kmeans-warmup-batches` flags.
- 5 new RVQ + warmup unit tests in `tests/test_vector_quantizer.py`
  (12 total, all CPU-only).
- Empirical finding documented at the bottom of
  `docs/v15-vqvae-and-cross-tokenizer-results.md`: RVQ + warmup
  drops final CE from 1.46 → 0.996 (32 % better than naive VQ),
  but bench inference *degrades* — the residual-sum representation
  drifts off the cloud LM's natural input-embedding manifold.
  Train/inference manifold mismatch is the new headline blocker
  for VQ-style IR; PR #34+ to explore on-manifold projection,
  partial-cloud finetune, and residual-correction (vs summation)
  variants.
- `src/router/quantization_aware.py` — quantization-aware IR:
  symmetric per-token fake quantization at int8 / int4 with
  straight-through estimator. Different attack angle than VQ-VAE
  (per-dim scalar quantization, codebook-free, no off-manifold
  summation). Helpers for both training-time fake-quantize and
  wire-side real-quantize/dequantize. 8 unit tests.
- `training/train_v15_qat.py` — paper-companion training with
  fake-quantize applied to MLP soft prompt before the cloud LM.
  `--qat-bits` flag picks 4 or 8. Wire payload at K=32, hidden=1152
  drops from 73 KB fp16 → 36 KB int8 → 18 KB int4.
- `training/train_v15_qat.py` `--resume-from` flag — warm-start MLP
  + projection from a previous checkpoint. Used to split the
  Gemma 4 26B-A4B 10k-step run across an A100 session boundary.
- `docs/v15-gemma4-26b-a4b-results.md` — 1B-ceiling hypothesis test:
  same edge + same K=32 QAT int8 + ~9 000 effective steps, only swap
  cloud to Gemma 4 26B-A4B-it (4-bit nf4 bf16). First on-topic
  haiku ("Softly, the rain falls…") with self-aware meta-commentary,
  real 2024 demographics from "list" prompts, but cross-prompt
  capital-attractor collapse appeared at step 9 000 that wasn't
  there at step 4 000. Verdict: 1B edge + 5 000-sample distilled set
  are the next-binding ceilings; cloud quality is no longer the
  blocker.

### Research (compiler-IR retrieval, ACL 2027 target)

- `docs/paper-compiler-ir-retrieval.md` — paper outline. Thesis:
  compiler-IR embeddings retrieve semantically-equivalent-but-
  syntactically-different code neighbours that AST and token
  embeddings miss, measured extrinsically by how much they
  accelerate an LLM-driven evolutionary code search.
  Decisions logged: ACL 2027, a2o-labs affiliation, Gemma 4 31B-it
  for Phase 2 extrinsic eval, L4 for Phase 0–1. (#36)
- `research/p0-compiler-ir-retrieval/` — Phase 0 closeout:
  `lit-review.md` (13 papers across 3 literatures; gap defensible),
  `humaneval-coverage-spike.md` + `humaneval_repr_spike.py` +
  `results/compile_coverage.json` (100 % HumanEval coverage on
  tokens / AST / Python bytecode; sizes inside encoder budgets),
  `extended-abstract.md` (2-page draft, recommends green-light
  Phase 1). (#40)

### Infrastructure

- **L4 edge endpoint migrated** from `l4-edge-1 @ <redacted-tailnet-ip>`
  (free-trial expired, instance terminated) to `l4-edge-2 @
  <edge-host>`. Stack is now Ollama 0.22.1 (was llama.cpp + k3s);
  model is `gemma3:27b` Q4_K_M (was `gemma-4-26B-A4B-it` Q4_K_M GGUF).
  Full migration write-up in `bench/README.md`.

## v0.2.0 — 2026-05-01

V1.5 production release: real PyTorch model code, full pipeline plumbed
through FastAPI, training loop + eval harness + bench data on real GPU,
plus production-grade extras (Prometheus, OTel, middleware, CLI, CI).

### V1.5 core

- `CompactSchemaV15` wire schema with `embedding_b64` (#1, #2)
- `EdgeEncoder` — frozen Gemma 4 base + trainable `Linear` projection,
  `nn.Module` real impl using `transformers.AutoModelForCausalLM` (#3)
- `SoftPromptAdapter` — trainable MLP → K=8 soft-prompt tokens prepended
  to a frozen cloud LM, BLIP-2 Q-Former pattern (#3)
- `V15Pipeline` — orchestrator with mock + real modes (#4)
- `/route/v15` FastAPI endpoint behind `V15_ENABLED` flag (#4)
- `/route/v15/stream` Server-Sent Events streaming variant (#15)
- `transformers>=4.45` BatchEncoding-compat fix in `_tokenize` (#8)

### Production extras

- `bitsandbytes` 4-bit support for both encoder + adapter (#9)
- LRU cache on `EdgeEncoder.encode` (sha256-keyed, 256 default,
  cache_size=0 disables for training) (#16)
- `EdgeEncoder.encode_chunked` for long-context inputs (#18)
- `EdgeEncoder.encode_batch` + `SoftPromptAdapter.forward_batch` for
  high-QPS GPU usage (#23)
- `EmbeddingClassifier` — cosine-NN classifier over an exemplar bank
  (standalone, plug-in into pipeline) (#17, #24)
- `CloudRouter` — task_type → cloud_model policy with
  `complexity_override` (standalone) (#19)
- `CombinedRouteGuard` — raw ASGI middleware: prompt-size + sliding
  window rate limit, wired into the FastAPI app by default (#20, #22)
- `gemma4-router-cli` console script: `encode` / `pipeline-v15` /
  `health` (#14)

### Observability

- OpenTelemetry tracing across V1 and V1.5 paths with no-op fallback
  (#12)
- `/metrics/prometheus` text exposition format (#21)

### Training + eval

- Real training loop in `training/train_v15.py`: AdamW over
  `EdgeEncoder.projection` + `SoftPromptAdapter.mlp` only (#10)
- Eval harness in `eval/eval_v15.py` with LLM-as-judge,
  token-reduction-ratio (tiktoken proxy), Brier score (#5)
- 62-sample synthetic eval dataset at `eval/data/eval_v15.jsonl` (#11)
- L4 smoke-test script `bench/v15_l4_smoke.py` + real-run JSON summary
  on `gemma-3-1b-it` (#6, #7)

### Tooling

- GitHub Actions CI: ruff + pytest on every PR + master push (#13)

## v0.1.0 — 2026-04-24

Initial V1 skeleton.

- Light/heavy classifier via edge LLM
- `CompactTask` v1.0 schema with mask-map for PII redaction
- Cloud forwarder to OpenAI-compatible endpoints (litellm)
- FastAPI `/route`, `/health`, `/metrics`
- pytest + ruff baseline

(`docs/v15-architecture.md` covers the V2.5 VQ-VAE codebook upgrade
gate that follows V1.5.)
