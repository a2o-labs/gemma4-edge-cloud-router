# bench/

> **Redaction note (2026-09-19, before open-sourcing).** Hostnames and tailnet addresses in this
> directory — in this README and in the three `*_2026-05-08.json` records — were replaced by
> `l4-edge-1` / `l4-edge-2` / `<edge-host>` / `<cloud-host>` / `bench client`. `<edge-host>` was an
> NVIDIA L4 on a private tailnet and `<cloud-host>` a rented A100; nothing else in the records changed.

Real-weight benchmarks and smoke tests for the V1.5 router pipeline.

## Status

This bench script is checked in WITHOUT a real-run JSON summary. Two
blockers prevented the smoke from running on L4 in the original PR:

1. Tailscale SSH `check` ACL required browser re-auth on every attempt.
2. `google/gemma-4-4b-it` does not exist on Hugging Face Hub (404). The
   script was substituted to `google/gemma-3-4b-it` as the closest-existing
   real instruct model that fits L4 24GB at fp16 (~8GB).

Once the SSH auth is unblocked, run:

    HF_TOKEN=... python bench/v15_l4_smoke.py 2>&1 | tee /tmp/v15-l4-smoke.log

…and append the resulting JSON summary to this README in a follow-up PR.

## v15_l4_smoke.py

End-to-end smoke test that exercises the production V1.5 code path
(`EdgeEncoder` -> `CompactSchemaV15` wire -> `SoftPromptAdapter`) against
real CUDA-resident Gemma-family weights.

### What it verifies

- `EdgeEncoder` loads a real instruct LM and produces an
  `(embedding_dim,)` float16 numpy vector
- `CompactSchemaV15` round-trips losslessly via base64
  (`encode().embedding_b64` -> `decode_embedding_from_schema`)
- `SoftPromptAdapter` accepts the decoded edge vector + the schema JSON
  and emits a non-empty decoded string via `inputs_embeds`
- `encode` and `forward` latency captured with p50/p95 stats
- CUDA memory headroom logged before and after each load

### Hardware target

NVIDIA L4 24GB (GCP `g2-standard-*` or equivalent). Default model is
`google/gemma-3-4b-it` (~8GB fp16) so both edge and cloud sides fit
comfortably with room for KV cache.

The original spec referenced `google/gemma-4-4b-it`; that model ID does
not exist on the Hugging Face Hub (verified 2026-04, HTTP 404). Gemma 4
currently ships only as the 26B-A4B mixture (~52GB fp16, doesn't fit on
L4 without 4-bit quantization). `google/gemma-3-4b-it` is the closest
shipping Google instruct model that satisfies the spec's intent of a
real, production-grade ~4B model that fits L4 in fp16. Override with
`--edge-model` / `--cloud-model` once a true Gemma 4 4B SKU is
published.

### Run

```bash
# in a fresh venv on the L4 host
pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45 accelerate>=0.34 numpy pydantic
git clone https://github.com/a2o-labs/gemma4-edge-cloud-router.git
cd gemma4-edge-cloud-router && pip install -e .

HF_TOKEN=$(cat path/to/hf-token) python bench/v15_l4_smoke.py
```

### Expected output

```
=== SUMMARY ===
{
  "edge_model": "google/gemma-3-4b-it",
  "cloud_model": "google/gemma-3-4b-it",
  "embedding_dim": 2048,
  "prompt_tokens": 8,
  "edge_load_ms": <measured>,
  "cloud_load_ms": <measured>,
  "encode": {"n": 5, "p50": ..., "p95": ..., "min": ..., "max": ...},
  "forward": {"n": 5, "p50": ..., "p95": ..., "min": ..., "max": ...},
  "cuda_mem_used_gb": ~16,
  "cuda_device": "NVIDIA L4",
  "cuda_total_gb": 23.6
}
```

### Known limits

- **No quantization path.** Both `EdgeEncoder.__init__` and
  `SoftPromptAdapter.__init__` accept only `dtype` (fp16/fp32); they do
  not currently expose a `load_in_4bit` / `quantization_config` kwarg.
  Loading 26B-A4B-it fp16 will OOM on L4; the smoke deliberately stays on
  4B fp16 until that flag is added.
- **No cloud-side A100 yet.** `vast-a100-80g` is offline in the current
  tailnet; once it returns, the cloud-side `SoftPromptAdapter` should be
  re-pinned to the A100 to match the production split.

### Future work

- Add `load_in_4bit` / `bnb_4bit_*` plumbing through both `EdgeEncoder`
  and `SoftPromptAdapter` so 26B-A4B fits on L4 (4-bit ~ 13GB).
- Re-run with cloud side on A100 80GB once `vast-a100-80g` rejoins the
  tailnet.
- Add a longer-prompt regression set (currently 5 short prompts, ~1-3
  tokens each post-tokenization).

## 2026-04-30 smoke run

First successful real-weight L4 run.

### Hardware

- Host: `l4-edge-1` (asia-northeast1-b, GCE g2 family)
- GPU: NVIDIA L4 24GB (driver 550.90.07, CUDA 12.4)
- Disk pre-run: 12 GB free; post-run unchanged (HF cache ~2 GB for the 1B model + tokenizer)

### Model

`google/gemma-3-1b-it` (~2 GB fp16) — substituted for the spec's
`google/gemma-3-4b-it` because the L4 disk had only ~7-13 GB free at run
time and the venv (torch cu124 + transformers) already consumed ~5 GB.
Code path verified end-to-end on real CUDA weights; the same script with
`--edge-model google/gemma-3-4b-it --embedding-dim 2048` works once disk
frees up.

### Results

```
bench/v15_l4_smoke.py \
    --edge-model google/gemma-3-1b-it \
    --cloud-model google/gemma-3-1b-it \
    --embedding-dim 1024 \
    --prompt-tokens 4
```

```json
{
  "edge_model": "google/gemma-3-1b-it",
  "cloud_model": "google/gemma-3-1b-it",
  "embedding_dim": 1024,
  "prompt_tokens": 4,
  "edge_load_ms": 8617.2,
  "cloud_load_ms": 5781.7,
  "encode": {
    "n": 5,
    "p50": 58.7,
    "p95": 553.2,
    "min": 52.8,
    "max": 553.2
  },
  "forward": {
    "n": 5,
    "p50": 4042.0,
    "p95": 4525.4,
    "min": 3973.1,
    "max": 4525.4
  },
  "cuda_mem_used_gb": 4.14,
  "cuda_device": "NVIDIA L4",
  "cuda_total_gb": 23.6
}
```

Notes on the numbers:

- `edge_load_ms` reflects a warm HF cache (second attempt). Cold pull
  from HF was an additional ~30s for the 2 GB safetensors.
- `encode` p95 is dominated by the first call (553 ms) — CUDA kernel
  warmup. Subsequent encodes settle at 53-64 ms.
- `forward` is dominated by `cloud.generate(..., max_new_tokens=64,
  do_sample=False)`. Roughly 16 tokens/s on L4 fp16 — consistent with
  Gemma 3 1B at this batch size (1) and prompt length (4 soft + ~50
  text-embedding tokens from the schema JSON).
- `cuda_mem_used_gb` = both 1B models + adapter MLP + activations,
  comfortably inside L4's 24 GB.

### Two pre-existing issues uncovered, both flagged as follow-ups

1. **`EdgeEncoder._tokenize` BatchEncoding-vs-Tensor compat.** With
   `transformers >= 5.0`, `tokenizer.apply_chat_template(...,
   return_tensors="pt")` returns a `BatchEncoding`, not a Tensor, which
   then trips `torch.ones_like(input_ids)` in `_last_hidden`. The smoke
   run patched this on L4 only with a 2-line guard before
   `return ids.to(self.device)`:

   ```python
   if hasattr(ids, "input_ids"):
       ids = ids.input_ids
   ```

   This patch is **not** in the PR — it belongs in a separate small
   router-fix PR against `src/router/edge_encoder.py`.

2. **`pyproject.toml` doesn't pin transformers.** `transformers>=4.45`
   resolves to 5.7.0 today, which (a) introduced the BatchEncoding
   change above and (b) is the only version that supports the
   `gemma3_text` architecture (added in 4.50+). Rolling back to 4.45.2
   fixes (1) but breaks Gemma 3 loading entirely. Either pin the
   floor higher (`transformers>=4.50`) or fix (1) — preferably both.

### Caveats

- Both edge and cloud are the same 1 B model on the same GPU.
  Production split (edge: 4B-26B on L4, cloud: 31B on A100 80GB) isn't
  exercised yet — A100 (`vast-a100-80g`) still offline.
- 1 B `hidden_size` is small (1152); `embedding_dim` was lowered to
  1024 to keep the projection sane. The same script with
  `--embedding-dim 4096` works on larger models.
- bitsandbytes 4-bit not yet supported by `EdgeEncoder` /
  `SoftPromptAdapter` `__init__` — still a follow-up before 26B-A4B
  fits.

## 2026-05-02 endpoint migration — Ollama replaces llama.cpp+k3s

The original L4 (`l4-edge-1` @ `<redacted-tailnet-ip>`) was on a free-trial
GCE create that expired; the instance terminated. A replacement
`g2-standard-4` + L4 was provisioned and runs **Ollama 0.22.1**
(simpler than the previous llama.cpp + k3s + flannel-bypass stack).

### New edge endpoint

| field | value |
|---|---|
| tailnet hostname | `l4-edge-2` |
| tailnet IP | `<edge-host>` |
| stack | Ollama 0.22.1 |
| OpenAI-compat URL | `http://<edge-host>:11434/v1` |
| model | `gemma3:27b` (Q4_K_M, 17 GB on disk) |
| provisioning | GCE `g2-standard-4` Spot, `auto_restart=false` |
| disk | 64 GB boot (resized from 10 GB default) |

### Why Ollama instead of llama.cpp+k3s

Fresh Debian 12 + 64 GB disk + a single L4: time-to-first-inference is
the priority, not orchestration. Ollama's installer auto-handles the
NVIDIA driver, exposes a OpenAI-compatible HTTP API on `:11434`, and
ships pre-quantised models from its registry. The previous llama.cpp +
k3s + hostNetwork DNS-bypass plumbing was solving problems that don't
exist on a fresh single-node VM.

### Smoke

```
$ time curl -sS http://<edge-host>:11434/api/generate \
    -d '{"model":"gemma3:27b","prompt":"Hello, what is 2+2? answer in 5 words.","stream":false}'
{"model":"gemma3:27b","response":"The answer is simply four.",
 "total_duration":131763819227,"prompt_eval_duration":132126411,
 "eval_count":8,"eval_duration":518070744,...}
```

131 s cold-load (17 GB → GPU); warm path ~520 ms / 8 generated tokens
on the L4 Q4_K_M (~ 15 tok/s). Peak GPU memory: 18 / 23 GiB.

### Caveats vs the previous endpoint

- **Different model family**: `gemma3:27b` (text-only Causal LM) instead
  of `gemma-4-26B-A4B-it` (multimodal, MoE). Prompt formatting may need
  re-tuning for callers ported from the old endpoint.
- **No model surgery via Ollama**: the V1.5 `SoftPromptAdapter` still
  needs direct `embed_tokens` access; that path requires a custom
  transformers loader (Ollama only exposes chat completions). Track 2
  of the deployment plan documented in `docs/v15-architecture.md`.
- **Spot preemption**: `auto_restart=false` means the instance stays
  off after preemption. Ops need a manual GCP-console restart; OS
  state (Ollama install, GPU driver, model weights) survives on the
  64 GB boot disk.

### What's gone

- `<redacted-tailnet-ip>` no longer routable (instance terminated).
- The k3s / svclb / nvidia-device-plugin manifests under `deploy/k8s/`
  describe the old llama.cpp deployment — left in tree as historical
  reference. If/when k3s is needed again on a fresh L4, those
  manifests still apply but disk-pressure-eviction safeguards from
  PR #7's debrief should be added (cache cleanup CronJob).

## v15_split_xmachine_bench.py — true L4 ↔ A100 cross-machine split

End-to-end split bench that exercises the production V1.5 wire path
across two physical hosts on the tailnet:

- **Edge (L4 24GB, `l4-edge-2 @ <edge-host>`)** runs a
  FastAPI server (`v15-edge/edge_server.py`) wrapping `EdgeEncoder`
  with `google/gemma-3-1b-it` fp16; exposes `POST /v15/encode {prompt}`
  → `CompactSchemaV15` JSON with `embedding_b64`.
- **Cloud (A100 SXM4 80GB, vast.ai @ `<cloud-host>`)** runs a FastAPI
  server (`/opt/v15-cloud-adapter/repo/bench/a100_cloud_server.py`)
  wrapping `SoftPromptAdapter` with `google/gemma-3-27b-it` 4-bit
  nf4; exposes `POST /v15/forward {schema_json, max_new_tokens}` →
  decoded response + token counts.
- The bench script (`bench/v15_split_xmachine_bench.py`) lives on the
  tailnet client (here the bench client host) and chains
  `L4 /v15/encode` → `A100 /v15/forward` per prompt, decomposing
  latency into compute (server-reported) vs network (RTT − compute).

### 2026-05-08 first cross-machine run

Hardware:

- Edge: NVIDIA L4 24GB on GCE `g2-standard-4` Spot in `asia-northeast1-b`.
  Driver 595.71.05, kernel `6.1.0-45-cloud-amd64` (rebuilt nvidia dkms
  for the new kernel before this run).
- Cloud: NVIDIA A100-SXM4-80GB on vast.ai (Japan host).
- Both hosts on the same tailnet; bench client on a separate tailnet host in
  Tokyo. Each prompt = 1 L4 RPC + 1 A100 RPC.

Models:

- Edge `google/gemma-3-1b-it` fp16, embedding_dim=1152, no quant
  (~2 GB GPU; sits alongside Ollama's `gemma3:27b` Q4_K_M on the same
  L4, peak ~17 GB used + 2 GB edge = well under 23 GB headroom).
- Cloud `google/gemma-3-27b-it` 4-bit nf4 (`bnb_4bit_compute_dtype=fp16`),
  hidden_size=5376, K=8 soft-prompt tokens, MLP randomly initialised
  (no checkpoint loaded — adapter weights are not trained yet).

5-prompt cold-cache run (prompts unique to this run, no LRU hits):

```
$ python3 bench/v15_split_xmachine_bench.py --max-new-tokens 64 \
    --prompts "What is the capital of Japan?" "List the planets" \
              "Write a poem about the ocean" "Solve: 17 * 23" \
              "Define the word 'serendipity'"
```

```json
{
  "edge":  {"model":"google/gemma-3-1b-it",  "embedding_dim":1152, "cuda_device":"NVIDIA L4",            "cuda_mem_used_gb":2.06},
  "cloud": {"model":"google/gemma-3-27b-it", "hidden":5376, "quantization":"bnb_4bit_nf4", "prompt_tokens":8, "cuda_device":"NVIDIA A100-SXM4-80GB", "cuda_mem_used_gb":18.35},
  "max_new_tokens": 64,
  "n_prompts": 5,
  "encode_compute_ms":  {"p50": 57.1,   "p95": 81.7,   "min": 55.6,   "max": 81.7},
  "encode_rtt_ms":      {"p50": 70.7,   "p95": 95.2,   "min": 68.9,   "max": 95.2},
  "encode_network_ms":  {"p50": 13.6,   "p95": 13.9,   "min": 13.3,   "max": 13.9},
  "forward_compute_ms": {"p50": 6704.5, "p95": 6932.1, "min": 6674.4, "max": 6932.1},
  "forward_rtt_ms":     {"p50": 6807.0, "p95": 7053.4, "min": 6768.0, "max": 7053.4},
  "forward_network_ms": {"p50": 102.5,  "p95": 121.3,  "min":  93.6,  "max": 121.3},
  "total_ms":           {"p50": 6877.5, "p95": 7122.3, "min": 6856.2, "max": 7122.3},
  "n_in_tokens_per_request": [2256, 2256, 2240, 2294, 2256],
  "n_out_tokens_per_request": [64, 64, 64, 64, 64]
}
```

LRU-hit run (same 5 prompts as `bench/v15_a100_4bit_bench.py`, repeated
after first cold pass): encode_compute drops to ~0.1 ms p50, encode_rtt
collapses to ~13.5 ms (pure HTTP RTT) — the EdgeEncoder LRU cache is
sha256-keyed by prompt and intercepts before the GPU.

### Decomposition

| stage              | p50      | what it includes                                         |
|--------------------|----------|----------------------------------------------------------|
| encode network     | 13.6 ms  | tailnet RTT + JSON serdes (`L4 ↔ bench client`)           |
| encode compute     | 57.1 ms  | tokenize + forward + projection on L4 (warm GPU)         |
| forward network    | 102.5 ms | tailnet RTT + 3.3 KB schema_json upload (`A100 ↔ bench client`) |
| forward compute    | 6704 ms  | 64-token greedy generate on 27B 4-bit nf4, ~2250 in tok |
| **end-to-end p50** | **6878 ms** | one full split round-trip                            |

The 27B 4-bit forward dominates (~98% of total). Network adds ~115 ms
per prompt over the local-only A100 bench (`bench/v15_a100_4bit_bench.py`):
prior local run was forward p50 ~7055 ms vs cross-machine ~6807 ms RTT
— variance within run-to-run noise, so the tailnet adds essentially no
overhead on top of the 27B generate cost.

### Caveat — empty decoded output (expected, not a pipeline bug)

All 5 forward responses decoded to 64 `<pad>` tokens (output_chars=0).
This is expected: the `SoftPromptAdapter.mlp` is randomly initialised
(no `training/train_v15.py` checkpoint in this bench), so the K=8 soft
prompt embeddings live in a region of embedding space that doesn't map
to any meaningful token distribution; with `do_sample=False` the LM
head argmax falls onto the lowest-entropy `<pad>` token. The bench
measures **wire path latency**, which is unaffected. Output quality is
gated on a trained adapter checkpoint — a separate workstream.

### Reproduce

L4 (edge) — assumes Debian 12 + L4 + working nvidia driver:

```bash
ssh <user>@<edge-host>
mkdir -p ~/v15-edge/router && cd ~/v15-edge
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install "transformers==4.55.4" accelerate fastapi uvicorn numpy pydantic
# scp src/router/{__init__.py,edge_encoder.py,schema_v15.py,schema.py} from gemma4-router
# scp bench/edge_server.py (this repo: see /tmp/l4_edge_server.py)
HF_TOKEN=hf_... nohup .venv/bin/python edge_server.py \
    --edge-model google/gemma-3-1b-it --embedding-dim 1152 --port 8002 \
    > /tmp/l4_edge_server.log 2>&1 &
```

A100 (cloud) — vast.ai instance with `/opt/v15-cloud-adapter/repo` and
a working `.venv` (torch 2.4.1+cu124, transformers 4.55.4, bnb 0.49.2,
fastapi, uvicorn) plus `/dev/shm/hf-cache` containing `gemma-3-27b-it`:

```bash
ssh <user>@<cloud-host>
cd /opt/v15-cloud-adapter/repo
export HF_HOME=/dev/shm/hf-cache HF_HUB_CACHE=/dev/shm/hf-cache/hub \
       PYTHONPATH=/opt/v15-cloud-adapter/repo/src
nohup .venv/bin/python bench/a100_cloud_server.py \
    --cloud-model google/gemma-3-27b-it --edge-dim 1152 --prompt-tokens 8 \
    --port 8001 > /tmp/a100_server.log 2>&1 &
```

Bench client (any host on the tailnet):

```bash
python3 bench/v15_split_xmachine_bench.py --max-new-tokens 64
```

### 2026-05-08 production-sizing sweeps

After the headline cross-machine run we kept the same edge+cloud
servers up and ran three orthogonal sweeps before tearing down the
A100. All sweeps share the same hardware (L4 edge gemma-3-1b-it fp16
on `<edge-host>`, A100 cloud gemma-3-27b-it bnb-4bit-nf4 on
`<cloud-host>`) and the K=8 untrained soft-prompt MLP. We only report
**latency / throughput** numbers — output quality is invariant under
these knobs (every generation still decodes to `<pad>` until the
adapter is trained).

#### A. `max_new_tokens` scaling (cross-machine)

5 prompts × 4 token-length points, raw at
`bench/v15_len_sweep_2026-05-08.json`.

| max_new_tokens | forward_compute p50 | forward_compute p95 | total p50  | ms / generated token |
|---------------:|--------------------:|--------------------:|-----------:|---------------------:|
|             32 |          3 676 ms   |          3 925 ms   |  3 783 ms  |               115 ms |
|             64 |          6 694 ms   |          6 708 ms   |  6 801 ms  |               105 ms |
|            128 |         12 724 ms   |         12 754 ms   | 12 830 ms  |                99 ms |
|            256 |         24 785 ms   |         24 904 ms   | 24 901 ms  |                97 ms |

Linear in generated-token count, asymptote ~96 ms/token at the long
end (steady-state generate). The ~10 ms/token excess at short lengths
is per-call setup (`generate` loop init, soft-prompt projection,
attention-mask build) amortising across more tokens as length grows.
Use these numbers for cost/latency planning per request length.

Reproduce:

```bash
for n in 32 64 128 256; do
  python3 bench/v15_split_xmachine_bench.py --max-new-tokens $n \
    --out /tmp/v15_xmachine_n$n.json \
    --prompts "What is the capital of Japan?" "List the planets" \
              "Write a poem about the ocean" "Solve: 17 * 23" \
              "Define the word 'serendipity'"
done
```

#### B. Concurrency under serial-forward semantics (cross-machine)

8 cross-machine end-to-end requests fired with N concurrent workers
against the same A100 server (one `cloud.generate(...)` call per
request, default FastAPI sync-in-threadpool semantics). Raw at
`bench/v15_xmachine_concurrency_2026-05-08.json`.

| concurrency | wall-clock | RPS    | total p50 | total p95 | per-request slowdown vs C=1 |
|------------:|-----------:|-------:|----------:|----------:|----------------------------:|
|           1 |     50.4 s | 0.159  | 6 858 ms  | 7 118 ms  |                  1.00× (ref) |
|           2 |     50.4 s | 0.159  |12 590 ms  |13 301 ms  |                       1.84× |
|           4 |    161.8 s | 0.049  |82 046 ms  |83 907 ms  |                      11.97× |

C=2 doubles per-request latency cleanly (the GPU serializes), so
end-to-end RPS is unchanged versus C=1. **C=4 collapses** — every
request takes ~12× the C=1 latency and overall RPS drops 3×. The
single-process FastAPI (sync handlers in anyio threadpool) lets four
generate calls coexist on the GPU; their KV-caches and dequant work
fight for memory bandwidth and the 4-bit dequant path appears to
serialize particularly poorly. **Conclusion: at A100-80GB-4bit, the
V1.5 cloud server should be capped at concurrency = 1–2 per
process.** Higher fan-out needs explicit batching (next sweep) or
horizontal replicas, not opportunistic concurrency.

Reproduce:

```bash
python3 bench/v15_concurrency_sweep.py --max-new-tokens 64 \
    --concurrency-levels 1 2 4 --requests-per-level 8
```

#### C. Static batch throughput (`forward_batch`, cross-machine)

Pre-encode 8 schemas on L4, then call `/v15/forward_batch` (added to
`bench/a100_cloud_server.py` for this sweep) with B prompts in a
single `cloud.generate(inputs_embeds=...)` of shape `(B, 8 + ~2256)`.
3 trials per B, best-of-3. Raw at
`bench/v15_batch_throughput_2026-05-08.json`.

|   batch | best forward_compute | tokens/sec | per-user latency | speedup vs serial B=1 |
|--------:|---------------------:|-----------:|-----------------:|----------------------:|
|       1 |             6 724 ms |        9.5 |          6 724 ms|                  1.00× |
|       2 |             9 410 ms |       13.6 |          4 705 ms|                  1.43× |
|       4 |            11 404 ms |       22.4 |          2 851 ms|                  2.36× |
|       8 |            15 682 ms |       32.6 |          1 960 ms|                  3.43× |

Static batching wins decisively over opportunistic concurrency: B=8
finishes 8 requests in 15.7 s (~32.6 tok/s aggregate) versus C=4
finishing 8 requests in 161.8 s. Per-user latency drops from 6.7 s to
2.0 s as B grows, and the speedup-vs-serial curve is sub-linear (3.43×
at B=8) — A100 4-bit nf4 27B is compute-bound, not bandwidth-bound, at
these K=8 / ~2256-input-token shapes.

**Production guidance**: a real V1.5 deployment should batch at the
adapter layer (collect requests in a small queue and fire B=4–8
batches) rather than letting concurrent HTTP requests race on the
GPU.

Reproduce (server must expose `/v15/forward_batch`):

```bash
python3 bench/v15_batch_client.py --batch-sizes 1 2 4 8 \
    --max-new-tokens 64 --trials 3
```

## judge_ab.py

Pre-existing A/B judge harness from earlier work; unrelated to V1.5.
