# CompactSchemaV15 — Wire Schema (V1.5)

Frozen field spec for the V1.5 hybrid envelope the edge router ships to the
cloud LLM. V1.5 = V1.0 (compact JSON) + parallel **embedding bridge**.

> Companion test contract: `tests/test_schema_v15.py` (9 invariants).
> Companion implementation: `src/router/schema_v15.py`.
> Architecture rationale: [`v15-architecture.md`](./v15-architecture.md).

Bump `version` before mutating any field name or type.

## Top-level fields

| field              | type                  | required | notes                                                                  |
|--------------------|-----------------------|----------|------------------------------------------------------------------------|
| `version`          | `"1.5"` literal       | yes      | Pinned. Schema-bump must be intentional.                               |
| `task_id`          | string                | yes      | UUID4-hex or other unique id, generated at encode time                 |
| `task_type`        | enum                  | yes      | `summarize` \| `qa` \| `code` \| `translate` \| `search` \| `reason` \| `other` |
| `complexity`       | enum                  | yes      | `light` \| `heavy` — classifier output                                 |
| `semantic_tags`    | `list[str]`           | no       | Free-form tags (e.g. `["finance", "urgent"]`); default `[]`            |
| `action_graph`     | `dict[str,Any]?`      | no       | Optional opaque dict — passthrough of node/edge plan                   |
| `symbol_packet`    | `dict[str,Any]?`      | no       | Optional opaque dict — entities, numbers, mask references              |
| `privacy`          | `dict[str,Any]?`      | no       | Optional redaction metadata                                            |
| `max_tokens_hint`  | int                   | no       | Default 512                                                            |
| **`embedding_b64`**| `str?`                | no       | base64-encoded `embedding_dim`-vector (see embedding section)          |
| **`embedding_dim`**| int                   | no       | Default 4096; matches projection-head output size                      |
| **`embedding_dtype`**| enum                | no       | `float16` (default) \| `float32`                                       |
| `confidence`       | float                 | no       | Classifier confidence, 0.0–1.0; default 0.0                            |

The bolded fields are **new in V1.5**. All other fields are V1.0-compatible
(see [`compact-schema-v1.md`](./compact-schema-v1.md) for V1.0 spec).

## Embedding bridge

The edge model emits a dense vector — typically the last-hidden-state at the
`[INST]`/`EOS` position projected through a frozen-base + trained Linear head
to a fixed `embedding_dim`. This vector carries semantic nuance the JSON
schema cannot easily flatten (e.g. tone, contextual references "the bug from
two paragraphs ago").

### Wire encoding

```python
import base64
import numpy as np

# encode (edge side)
vec: np.ndarray  # shape=(embedding_dim,), dtype=float16
embedding_b64 = base64.b64encode(vec.tobytes()).decode("ascii")
embedding_dim = vec.shape[0]
embedding_dtype = "float16"  # or "float32"

# decode (cloud adapter side)
raw = base64.b64decode(embedding_b64.encode("ascii"))
np_dtype = {"float16": np.float16, "float32": np.float32}[embedding_dtype]
vec = np.frombuffer(raw, dtype=np_dtype)
assert vec.shape == (embedding_dim,)
```

### Dtype contract

- `float16` is the default for production: 2 bytes per element × 4096 = 8 KiB
  payload (≈ 11 KiB after base64).
- `float32` is permitted for training-time / dev mode (preserves gradient
  precision when the projection head is being trained). 4 bytes per element ×
  4096 = 16 KiB payload (≈ 22 KiB after base64).
- `float64` and other dtypes are rejected at parse time.

### Dim contract

- Default `embedding_dim = 4096` matches Gemma 4 base model hidden size.
- Non-default dims (e.g. `2048` for low-rank projection experiments) are
  permitted as long as the projection head and cloud soft-prompt MLP agree on
  the value. The schema validates the wire format only; head/MLP are
  responsible for rejecting mismatched dims at runtime.

## V1-compat fallback path

If the edge model fails to produce an embedding (model loading error,
out-of-memory, or training-time pre-init), the envelope is still valid with
`embedding_b64 = None`. The cloud adapter MUST handle this case by falling
back to JSON-only inference (V1 path). This guarantees graceful degradation:
a bad adapter checkpoint never bricks the cloud path.

## Example envelope (V1.5 with embedding)

```json
{
  "version": "1.5",
  "task_id": "f4c7a8e2",
  "task_type": "qa",
  "complexity": "heavy",
  "semantic_tags": ["finance", "urgent"],
  "action_graph": {
    "nodes": [
      {"id": "n1", "op": "retrieve", "args": {"query": "@symbol_0"}},
      {"id": "n2", "op": "summarize", "args": {"length": "short"}}
    ],
    "edges": [{"from": "n1", "to": "n2"}]
  },
  "symbol_packet": {
    "entities": [{"type": "company", "value": "[ENTITY_01]"}]
  },
  "privacy": {
    "mask_applied": true,
    "mask_version": "v1",
    "redaction_level": "medium"
  },
  "max_tokens_hint": 512,
  "embedding_b64": "Lh4qBQAA...<8KB base64>...",
  "embedding_dim": 4096,
  "embedding_dtype": "float16",
  "confidence": 0.91
}
```

## Example envelope (V1.5 V1-compat fallback, no embedding)

```json
{
  "version": "1.5",
  "task_id": "f4c7a8e2",
  "task_type": "qa",
  "complexity": "heavy",
  "action_graph": {"nodes": [...], "edges": [...]},
  "embedding_b64": null,
  "confidence": 0.0
}
```

The cloud adapter sees `embedding_b64=None` and routes straight to JSON-only
path (equivalent to V1.0 behaviour).

## Cloud adapter input assembly

```
cloud_31b_input = [
  ...soft_prompt_MLP(embedding_b64) → K=8 prompt tokens
  ...JSON-rendered envelope text
  ...optionally: original masked instruction text
]
```

Order matters: prompt tokens go FIRST so the 31B attention sees them at
position 0 (most influence on early-layer routing). JSON text follows so the
model can cross-reference structured fields against the latent.

## Schema bump policy

A new V1.6/V2.0 must be a breaking change (new field added → version bump).
Adding `description`-level annotations or test cases is non-breaking and may
land on V1.5.

## Cross-reference

- Pydantic model: `src/router/schema_v15.py` (`CompactSchemaV15`)
- Round-trip tests: `tests/test_schema_v15.py` (9 invariants)
- Edge encoder: `src/router/edge_encoder.py` (writes embedding_b64)
- Cloud adapter: `src/router/cloud_adapter.py` (reads embedding_b64 + soft-prompt MLP)
- Architecture rationale: [`v15-architecture.md`](./v15-architecture.md)
- V1.0 spec (subset compat): [`compact-schema-v1.md`](./compact-schema-v1.md)
