# V1.5 Architecture — Hybrid Compact-JSON + Soft-Prompt Embedding Bridge

## Goal

V1.0 (already shipped as the V1 skeleton in this repo) routes a prompt through
a local Gemma4-26B-A4B classifier; light requests are answered on-device, and
heavy requests are encoded as compact JSON (CompactTask v1.0) and forwarded to
a remote OpenAI-compatible endpoint (litellm fronting a larger Gemma4 31B).

V1.5 keeps that structure, but enriches the heavy path with a **dense
embedding bridge**: the edge model's last-hidden-state is projected through a
small linear head, base64-packed into the schema as `embedding_b64`, and on
the cloud side a tiny soft-prompt MLP turns that 4096-d vector into K=8
"virtual" prompt tokens that are prepended to the JSON forward through the
frozen Gemma4-31B.

The result: the cloud model receives both the explicit semantic JSON
(human-readable, debuggable, redactable) **and** an opaque latent that
preserves whatever signal the edge model could not flatten into discrete
tags. Compression budget for the JSON stays tight; the bridge absorbs
overflow.

## Why V1.5 (not V1 or V2)

V1: pure JSON. Simple, audit-friendly, but lossy on nuanced prompts ("the bug
that I described two paragraphs ago").

V2: end-to-end soft prompt only. Maximum compression, near zero
debuggability, no privacy redaction surface, and a single model upgrade
breaks everything.

V1.5: keep the JSON for control, add the embedding for fidelity. Both signals
are ablatable: setting `embedding_b64=null` falls back to V1 behaviour;
setting `action_graph=null` lets the cloud lean entirely on the embedding.
This is what the fleet (TL data, SRE infra, redacted training) converged on
in `cc.fleet.coord.v15-final` (NATS topic).

## Component diagram

```
            +-----------------------------------------------------+
            |                       EDGE                          |
            |                                                     |
   prompt ->|  Gemma4-26B-A4B (Q4, frozen)                        |
            |     |                                               |
            |     +-> classifier  -> {task_type, complexity}      |
            |     |                                               |
            |     +-> last hidden -> Linear[4096->4096] -> vec    |
            |                                                     |
            |  CompactSchemaV15(json_text, vec) --b64--> network  |
            +------------------------+----------------------------+
                                     |
                                     v
            +------------------------+----------------------------+
            |                      CLOUD                          |
            |                                                     |
            |  vec --MLP[4096, 8192, 8x4096]--> soft prompt (K,D) |
            |                                       |             |
            |  json_text --tokenizer--> token embs  |             |
            |                                       v             |
            |     [soft_prompt | token_embs] -> Gemma4-31B (Q4)   |
            |                                       |             |
            |                                       v             |
            |                                  generated text     |
            +-----------------------------------------------------+
```

K=8 was chosen as a compromise between expressivity and prefix-cache hit
rate; if profiling shows latency overhead from the embedding-only prefix, K
can be dialled to 4 without retraining the MLP head shape (just zero-pad).

## Schema V1.5 spec

The pydantic source of truth is `src/router/schema_v15.py`. Summary:

| field             | type                                | notes                                    |
|-------------------|-------------------------------------|------------------------------------------|
| version           | "1.5" (literal)                     | bumped from 1.0                          |
| task_id           | str                                 | UUIDv4                                   |
| task_type         | enum (7 vals)                       | unchanged from V1                        |
| complexity        | "light" \| "heavy"                  | unchanged                                |
| semantic_tags     | list[str]                           | unchanged                                |
| action_graph      | dict\|None                          | optional, V1 carryover                   |
| symbol_packet     | dict\|None                          | optional, V1 carryover                   |
| privacy           | dict\|None                          | optional, V1 carryover                   |
| max_tokens_hint   | int                                 | default 512                              |
| **embedding_b64** | str\|None                           | base64(float16, 4096) — NEW              |
| **embedding_dim** | int                                 | 4096 — NEW                               |
| **embedding_dtype** | "float16" \| "float32"            | NEW                                      |
| **confidence**    | float                               | edge classifier confidence — NEW         |

Backwards compatibility: a V1.5 schema with `embedding_b64=null` and
`version` patched back to "1.0" is byte-identical to V1.0 — the cloud
adapter must therefore treat a null embedding as "skip the soft prompt" and
fall through to V1.0 behaviour.

## Training pipeline

Ground truth: paired examples `(input_text, target_text, task_type,
complexity)`, harvested by TL from production logs (rotated for privacy,
with PII red-team filter applied — see SRE compliance note).

```
   train.jsonl
       |
       v
  +-------------------+   freeze base
  |  EdgeEncoder      |   train Linear head
  |  (Gemma4-26B-A4B) |
  +---------+---------+
            | vec
            v
  +-------------------+   freeze base
  |  CloudAdapter     |   train MLP only
  |  (Gemma4-31B)     |
  +---------+---------+
            | logits
            v
       +----+-------+
       | LM loss    |  +  aux JSON-validity loss
       +------------+     (parse v1.5 schema, 0 if valid)
            |
            v
        AdamW step
```

Trainable param budget: Linear[4096 x 4096] (16 M) + MLP[4096 x 8192 +
8192 x 32768] (~302 M). Total ~318 M trainable on top of two frozen Gemma4
bases. Fits a single L4 24GB with bf16 + grad-checkpointing per the SRE
sizing memo.

Hyperparameters in `training/train_v15.py` are placeholders. Real sweep
will be redacted's responsibility once TL hands over the harvested pairs.

## Evaluation

`eval/eval_v15.py` exposes three primary metrics:

1. **task_success_rate** — LLM-as-judge A/B vs full-text Gemma4 31B
   baseline. Re-uses `bench/judge_ab.py` from V1.
2. **token_reduction_ratio** — `1 - (V1.5 prompt tokens / baseline tokens)`.
   Negative would mean the JSON+embedding overhead is eating into savings;
   target >= 0.6 on the heavy distribution.
3. **brier_score** — calibration of the light/heavy classifier; lower is
   better, target <= 0.15.

Plus a locust-compatible load test stub that hits `/route` end-to-end for
p50/p95/p99 latency under concurrency.

## V2.5 upgrade gate

V1.5 is intentionally a stepping stone. V2.5 retires the JSON entirely and
relies on the embedding bridge alone, with optional discrete tags only
when redaction is required. The gate from V1.5 -> V2.5 is:

- task_success_rate parity (>= baseline) on a 5k held-out eval set
- token_reduction_ratio >= 0.75 on heavy
- privacy red-team passes a 100-prompt PII leak suite
- p95 latency under load <= V1.0 + 10 percent

Until those clear, V2.5 stays on a feature branch.

## Repository layout (V1.5 additions only)

```
src/router/schema_v15.py           pydantic CompactSchemaV15
src/router/edge_encoder.py         frozen Gemma4-26B-A4B + linear head stub
src/router/cloud_adapter.py        soft-prompt MLP + frozen Gemma4-31B stub
training/train_v15.py              accelerate / peft training skeleton
eval/eval_v15.py                   success / token / Brier / locust skeleton
deploy/k8s/edge-inference.yaml     edge serving deployment + svc
deploy/k8s/cloud-vllm.yaml         vLLM 0.19 + Gemma4-31B Q4 + PVC
deploy/k8s/cloud-adapter-wrapper.yaml  adapter front, ExternalName in ai-infra
docs/v15-architecture.md           this file
```

V1.0 modules in `src/router/{api,classifier,encoder,decoder,
cloud_forwarder,...}.py` are untouched.

## Open questions deferred to fleet

- Embedding privacy: does the 4096-d vector leak more than the JSON?
  SRE has flagged this for the next compliance review; until cleared,
  embedding emission is gated by a config flag (default off in production).
- MLP K: 8 vs 4 vs 16 — to be settled empirically by redacted post-training.
- vLLM soft-prompt support: vLLM 0.19 needs `--enable-prefix-caching` plus
  a custom prompt-embedding adapter; SRE will validate on the GPU node.
