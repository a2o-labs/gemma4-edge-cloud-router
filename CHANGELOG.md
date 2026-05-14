# Changelog

All notable changes to gemma4-router are documented here.

## Unreleased

### V1.5

- `POST /route/auto` — adaptive V1.0/V1.5 dispatch. Counts the prompt's
  cloud-tokenizer tokens and routes to V1.5 only when prompt > threshold
  (default 92, the empirical V1.0/V1.5 crossover). Gated by
  `V15_ADAPTIVE_ROUTING_ENABLED=true`. New `src/router/adaptive_router.py`
  + 9 unit tests.
- `docs/v15-1b-baseline-results.md` — write-up of the 2026-05-08 / 05-09
  1B/1B baseline run: K sweep (K=4/8/16/32), auxiliary losses
  (CE + KL distillation + contrastive), 5000-sample alpaca→gemma3:4b
  distillation, and the breakthrough on the K=32 + aux + 5k checkpoint
  (first factually-correct answer, first format-perfect haiku).

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
