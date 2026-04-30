# gemma4-router

V1 skeleton for the Gemma4 Edge-Cloud Semantic Router.

Flow: local Gemma4 classifies an inbound request as `light` or `heavy`. Light tasks
are answered on-device by the edge LLM. Heavy tasks are encoded into a compact
schema (v1.0) and forwarded to a remote OpenAI-compatible endpoint (litellm).

## Setup

```bash
uv venv
uv pip install -e '.[dev]'
cp .env.example .env   # then edit
cp config.example.yaml config.yaml
```

## Run

```bash
uvicorn router.api:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints:

- `POST /route` — main entry, body `{ "prompt": "...", "session_id": "..." }`
- `GET  /health` — liveness
- `GET  /metrics` — basic counters (requests, split_ratio, p50_latency)

## Test

```bash
pytest
```

## Docker

```bash
cd docker && docker compose up --build
```

Brings up the router plus an edge LLM sidecar slot (default: llama.cpp server,
swap for Ollama/vLLM/MLX by editing `docker/compose.yaml`).

## Layout

- `src/router/api.py` — FastAPI routes
- `src/router/classifier.py` — light/heavy classifier via edge LLM
- `src/router/edge_executor.py` — local answer path
- `src/router/encoder.py` + `schema.py` — CompactTask v1.0 + mask_map
- `src/router/cloud_forwarder.py` — forwards to litellm
- `src/router/decoder.py` — unpack cloud response + unmask
- `src/router/telemetry.py` — OTel stub
- `bench/judge_ab.py` — offline LLM-as-judge A/B harness (stub)
- `docs/compact-schema-v1.md` — schema field spec

## Status

V1 skeleton. Classifier, encoder, forwarder wired end-to-end but no trained
Gemma4 adapter yet. LLM-as-judge bench is a stub. See
`/tmp/router-research-report.md` for the framework evaluation that motivated
the build decision.

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
  (vast.ai per the SRE deployment plan). Only the soft-prompt MLP is
  trainable.
- **Install training deps**: `uv pip install -e '.[training]'` pulls
  torch, transformers, accelerate, peft.
- **Deployment manifests**: see `deploy/k8s/` for the edge inference
  pod, cloud vLLM, and adapter wrapper. Manifests are unchanged by this
  PR — the model code drop-in is source-compatible.

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
