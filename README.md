# gemma4-router

[![tests](https://github.com/a2o-labs/gemma4-edge-cloud-router/actions/workflows/test.yml/badge.svg)](https://github.com/a2o-labs/gemma4-edge-cloud-router/actions/workflows/test.yml)

Edge-cloud semantic router for the Gemma 4 family. Two coexisting wire formats:

- **V1.0** — light/heavy classifier dispatches to a local edge LLM or a
  remote OpenAI-compatible endpoint via a compact JSON schema.
- **V1.5** — adds a dense embedding bridge: the edge model emits a
  4096-d vector packed into the schema, and a trainable soft-prompt MLP
  prepends K=8 token embeddings to a frozen cloud LM at inference.
  BLIP-2 Q-Former pattern; only `~50M` trainable params.

V1.5 lives behind a feature flag (`V15_ENABLED`); V1 is the default and is
unaffected.

## Quickstart

```bash
uv venv
uv pip install -e '.[dev]'
cp .env.example .env   # edit
cp config.example.yaml config.yaml

uvicorn router.api:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints (V1):

- `POST /route` — main entry, body `{ "prompt": "...", "session_id": "..." }`
- `GET  /health` — liveness
- `GET  /metrics` — JSON counters
- `GET  /metrics/prometheus` — Prometheus text exposition

Endpoints (V1.5, opt-in via `V15_ENABLED=true`):

- `POST /route/v15` — encode + soft-prompt + cloud generate, blocking
- `POST /route/v15/stream` — same but Server-Sent Events
  (frame types: `schema`, `token`, `done`)

## Feature summary

| Capability                              | Module                                     | Status |
|----------------------------------------|--------------------------------------------|--------|
| V1 classifier + JSON forwarder         | `src/router/{classifier,encoder,forwarder}.py` | ready |
| V1.5 schema (`embedding_b64`)          | `src/router/schema_v15.py`                 | ready |
| V1.5 EdgeEncoder (real PyTorch)        | `src/router/edge_encoder.py`               | ready |
| V1.5 SoftPromptAdapter                 | `src/router/cloud_adapter.py`              | ready |
| V1.5 pipeline (mock + real)            | `src/router/v15_pipeline.py`               | ready |
| Streaming SSE response                 | `/route/v15/stream`                        | ready |
| Embedding-similarity classifier        | `src/router/embedding_classifier.py`       | optional plug-in |
| Multi-cloud router (task→model)        | `src/router/cloud_router.py`               | standalone (not yet wired into pipeline) |
| Input validation + rate-limit middleware | `src/router/middleware.py`               | wired by default (`MIDDLEWARE_ENABLED=true`) |
| OpenTelemetry tracing                  | `src/router/telemetry.py`                  | opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT` |
| Prometheus metrics                     | `/metrics/prometheus`                      | always on |
| LRU cache on `EdgeEncoder.encode`      | `_EncodeCache` (256 default, off in training) | ready |
| Chunked + batch encoding               | `EdgeEncoder.encode_chunked` / `encode_batch` | ready |
| bnb 4-bit support (Gemma 4 26B-A4B fit) | `quantization_config` kwarg               | optional via `[quantization]` extra |
| CLI tool                               | `python -m router.cli` / `gemma4-router-cli` | ready |
| Training loop (frozen-base + adapter)  | `training/train_v15.py`                    | ready |
| Eval harness (LLM-as-judge + brier)    | `eval/eval_v15.py`                         | 62-sample dataset |
| GitHub Actions CI                      | `.github/workflows/test.yml`               | runs on every PR |

## Test

```bash
pytest tests/ -m "not slow"      # ~100 tests, CPU only, no HF downloads
pytest tests/ -m slow            # adds SmolLM2-135M chat-template regression
```

## Docker

```bash
cd docker && docker compose up --build
```

Brings up the router plus an edge LLM sidecar slot (default: llama.cpp server,
swap for Ollama/vLLM/MLX by editing `docker/compose.yaml`).

## Layout

V1 (untouched since v0.1):

- `src/router/api.py` — FastAPI app + `/route`, `/route/v15`, `/route/v15/stream`,
  `/health`, `/metrics`, `/metrics/prometheus`
- `src/router/classifier.py` — light/heavy classifier via edge LLM
- `src/router/edge_executor.py` — local answer path
- `src/router/encoder.py` + `schema.py` — CompactTask v1.0 + mask_map
- `src/router/cloud_forwarder.py` — forwards to litellm
- `src/router/decoder.py` — unpack cloud response + unmask
- `src/router/telemetry.py` — counters + OTel + Prometheus exposition

V1.5 (added in v0.2):

- `src/router/schema_v15.py` — `CompactSchemaV15`
- `src/router/edge_encoder.py` — frozen base + trainable projection +
  LRU cache + batch + chunked encoding
- `src/router/cloud_adapter.py` — soft-prompt MLP + frozen cloud +
  blocking + streaming + batch forward
- `src/router/v15_pipeline.py` — orchestrator with optional classifier
- `src/router/embedding_classifier.py` — cosine-NN over exemplar bank
- `src/router/cloud_router.py` — task_type → cloud model policy
- `src/router/middleware.py` — `CombinedRouteGuard` (raw ASGI)
- `src/router/cli.py` — one-shot inference CLI
- `training/train_v15.py` — AdamW over `projection` + `mlp` only
- `eval/eval_v15.py` + `eval/data/eval_v15.jsonl` — judge + 62 samples
- `bench/v15_l4_smoke.py` — real-weight L4 smoke (run logs in `bench/README.md`)
- `deploy/k8s/` — edge inference pod, cloud vLLM, adapter wrapper
- `docs/v15-architecture.md` — design rationale

## Status

V1.5 functional end-to-end:

- Real-weight L4 smoke run on `google/gemma-3-1b-it` exercises the
  encode → wire → cloud generate path on a real CUDA device. JSON
  summary in `bench/README.md`.
- Training loop verified on `tiny-random-LlamaForCausalLM` (CPU, 2
  steps); frozen-base invariant covered by tests.
- Eval harness ships a 62-sample synthetic dataset; LLM-as-judge mode
  hooks into a litellm-compatible endpoint.

External dependencies still pending integration:

- A100 80GB cloud target for `SoftPromptAdapter` — rented (vast.ai).
- Paired training data — an external data-harvest pipeline, not in this repo.
- Liqo NamespaceOffloading on the `ai-infra` namespace — cluster-side follow-up, not configured here.

See `bench/README.md` and `docs/v15-architecture.md` for hardware specs and
the V2.5 codebook upgrade gate.

## V1.5 — hybrid compact-JSON + soft-prompt embedding bridge

V1.5 keeps the V1 JSON path and adds a dense embedding bridge:

- Edge: Gemma4-26B-A4B Q4 (frozen) + linear projection head emits a
  4096-d float16 vector alongside the V1 JSON, packed into the schema as
  `embedding_b64`.
- Cloud: a small MLP turns that vector into K=8 soft-prompt tokens that
  are prepended to the JSON before forwarding through the frozen Gemma4-31B
  (Q4 GGUF, served via vLLM 0.19).
- Schema source of truth: `src/router/schema_v15.py` (CompactSchemaV15).
- Training and eval skeletons live under `training/train_v15.py` and
  `eval/eval_v15.py`; both are structural only — no training is executed
  in this repo.
- Deployment manifests for the edge inference pod, cloud vLLM, and the
  adapter wrapper (with an ExternalName service in `ai-infra` ns) live
  under `deploy/k8s/`.
- Full design rationale: `docs/v15-architecture.md`.

V1.5 modules sit alongside V1.0; the existing V1 router code is untouched.
Setting `embedding_b64=null` makes a V1.5 payload behaviourally identical to
V1.0, so the cloud side can ship the adapter and roll the bridge in
gradually.

## Training & inference

The V1.5 edge encoder and cloud adapter are real PyTorch + HuggingFace
transformers modules in `src/router/edge_encoder.py` and
`src/router/cloud_adapter.py`. Both lazy-import torch/transformers, so
plain `import` of the package stays light for users who only need the
schema or routing layer.

- **Tests**: `tests/test_v15_pipeline.py` exercises edge encode → wire
  round-trip → cloud adapter generate using
  `trl-internal-testing/tiny-random-LlamaForCausalLM` so CI runs on CPU
  with no GPU and no real Gemma 4 weights.
- **Production edge**: Gemma 4 26B-A4B-it (frozen) on an L4 24GB. The
  trainable `nn.Linear` projection head trains on the same L4 since the
  base is frozen.
- **Production cloud**: Gemma 4 31B-it (frozen) on an A100 80GB
  (vast.ai, rented). Only the soft-prompt MLP is
  trainable.
- **Install training deps**: `uv pip install -e '.[training]'` pulls
  torch, transformers, accelerate, peft.
- **Deployment manifests**: see `deploy/k8s/` for the edge inference
  pod, cloud vLLM, and adapter wrapper. Manifests are unchanged by this
  PR — the model code drop-in is source-compatible.

### 4-bit quantization for L4 (Gemma 4 26B-A4B fit)

Gemma 4 26B-A4B-it at fp16 is ~52GB, which exceeds an L4's 24GB VRAM.
Use bitsandbytes 4-bit quantization (NF4) to fit ~13GB:

```bash
uv pip install -e '.[quantization]'
```

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

enc = EdgeEncoder(
    model_name="google/gemma-4-26B-A4B-it",
    embedding_dim=4096,
    quantization_config=bnb_config,
)
```

Same `quantization_config` kwarg is also accepted by `SoftPromptAdapter`
for the cloud-side base. Trainable layers (`projection` / `mlp`) stay
fp16 on `cuda`; only the frozen base is 4-bit. This composes cleanly
with the V1.5 trainer.

### Calling the V1.5 endpoint

The V1 `/route` endpoint stays unchanged. V1.5 lives under `/route/v15` and
is gated behind the `V15_ENABLED` flag so it is a no-op when disabled.

```bash
# Production: real Gemma 4 26B-A4B-it + Gemma 4 31B-it on L4/A100
export V15_ENABLED=true
export V15_HF_TOKEN=...
export V15_DEVICE=cuda

# Local dev / CI: mock mode (no GPU, no HF weights)
export V15_ENABLED=true
export V15_MOCK_MODE=true
export V15_DEVICE=cpu

curl -X POST http://localhost:8080/route/v15 \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "How does photosynthesis work?"}'
```

Response shape: `{ task_id, path: "v1.5", answer, schema_version: "1.5",
embedding_dim, complexity, latency_ms }`. When `V15_ENABLED=false` (default)
or the pipeline failed to load, `/route/v15` returns `503`.

### Observability — OpenTelemetry tracing

Install the optional extra and set `OTEL_EXPORTER_OTLP_ENDPOINT`:

```bash
pip install '.[observability]'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.observability:4317
export OTEL_SERVICE_NAME=gemma4-router
```

The router emits these spans:

| Span name        | Surface           | Useful attrs                          |
|------------------|-------------------|---------------------------------------|
| `route.classify` | `/route` only     | `force`, `complexity`, `confidence`   |
| `route.dispatch` | `/route` only     | `path` ∈ {light, heavy}               |
| `v15.run`        | `/route/v15`      | `prompt_chars`, `complexity`, `answer_chars` |
| `v15.encode`     | inside `v15.run`  | `embedding_dim`, `complexity`         |
| `v15.forward`    | inside `v15.run`  | `max_new_tokens`, `response_chars`    |

FastAPI HTTP-layer spans (request method, route, status) are auto-emitted
by `opentelemetry-instrumentation-fastapi`.

If the optional extra is not installed or `OTEL_EXPORTER_OTLP_ENDPOINT`
is unset, all instrumentation degrades to no-ops with zero overhead.

### Evaluating V1 vs V1.5

The eval harness in `eval/eval_v15.py` runs a small canned dataset through
both V1 and V1.5 paths and reports:

- `v1_success_rate` / `v15_success_rate` via LLM-as-judge
- `delta` (V1.5 - V1; positive = V1.5 better)
- `token_reduction_ratio` (compact JSON schema vs full prompt, measured with
  `tiktoken cl100k_base` as a proxy for the Gemma tokenizer)
- `classifier_brier_score` (light/heavy correctness)

Mock mode (no GPU, no real judge — used in CI):

    python -m eval.eval_v15 --dataset canned --mode mock --judge mock

Real mode (router running, judge LLM accessible):

    python -m eval.eval_v15 --dataset eval/data/eval_v15.jsonl --mode http \
        --router-base http://localhost:8080 \
        --judge llm \
        --judge-base http://localhost:4000/v1 \
        --judge-model claude-haiku-4-5

The bundled canned dataset lives at `eval/data/eval_v15.jsonl` (10 samples
covering qa / code / summarize / translate / reason / chat).

A locust load test stub for `/route/v15` lives at `eval/locust_v15.py` —
manual run: `pip install locust && locust -f eval/locust_v15.py --host
http://localhost:8080`. Not exercised in CI.

### Training V1.5 adapters

The frozen Gemma 4 base models stay fixed; only `EdgeEncoder.projection`
(`Linear(base_hidden, embedding_dim)`) and `SoftPromptAdapter.mlp`
(`Linear(edge_dim, edge_dim*2) -> GELU -> Linear(edge_dim*2,
prompt_tokens*cloud_hidden)`) are trainable (~50M params total at the
4096/8 V1.5 defaults). BLIP-2 Q-Former pattern.

Mock mode (CI / local dev, no GPU, `tiny-random-LlamaForCausalLM`):

    V15_MOCK_MODE=true python -m training.train_v15 --max-steps 100 --mock-mode

Real mode (L4 GPU, paired data):

    HF_TOKEN=... python -m training.train_v15 \
        --train-jsonl data/v15/train.jsonl \
        --max-steps 5000 --batch-size 4 --learning-rate 5e-4 \
        --device cuda

Checkpoints land in `outputs/v15-train/ckpt-stepN.pt` with the
`projection` and `mlp` `state_dict`s plus the `TrainConfig` used. The
final state is also written to `outputs/v15-train/final.pt`. The
`outputs/` directory is already gitignored.

The trainer is intentionally minimal: AdamW, single-sample SGD steps
(no grad-accum yet), no `accelerate`/`peft` wrappers. If/when
distributed or LoRA-style adapters are needed, wrap with `accelerate`
and add `peft.LoraConfig` against `edge.projection` / `adapter.mlp`
(both already in the optional `training` extra of `pyproject.toml`).
