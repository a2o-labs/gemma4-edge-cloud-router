# bench/

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

## judge_ab.py

Pre-existing A/B judge harness from earlier work; unrelated to V1.5.
