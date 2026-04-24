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
