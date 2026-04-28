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
