# CompactTask v1.0 — Wire Schema

Frozen field spec for the envelope the edge router ships to the cloud LLM.
Bump `schema_version` before mutating.

## Top-level fields

| field            | type                  | required | notes                                                       |
|------------------|-----------------------|----------|-------------------------------------------------------------|
| `schema_version` | `"1.0"` literal       | yes      | Must equal `"1.0"` for this revision                        |
| `task_id`        | string (hex, 32)      | yes      | UUID4-hex, generated at encode time                         |
| `task_type`      | enum                  | yes      | `qa` \| `code` \| `chat` \| `reasoning` \| `other`          |
| `instruction`    | string                | yes      | Masked user request — may contain `@symbol_N` placeholders  |
| `symbols`        | array<SymbolPacket>   | no       | Typed references explaining each placeholder                |
| `context_refs`   | array<string>         | no       | Opaque ids (memory, file hash) the cloud may resolve        |
| `privacy`        | Privacy object        | yes      | Redaction metadata                                          |
| `meta`           | object                | no       | Free-form bag for classifier confidence, trace ids, etc.    |

Extra fields are **rejected** at parse time (`extra="forbid"`).

## SymbolPacket

| field    | type    | notes                                               |
|----------|---------|-----------------------------------------------------|
| `symbol` | string  | Opaque id, e.g. `@email_0`                          |
| `kind`   | enum    | `pii` \| `file` \| `url` \| `code` \| `other`       |
| `hint`   | string? | Non-sensitive hint (e.g. `"email-like"`)            |

## Privacy

| field             | type    | notes                                                 |
|-------------------|---------|-------------------------------------------------------|
| `mask_applied`    | bool    | true if any symbols were inserted                     |
| `mask_version`    | string  | Mask algorithm version; currently `v1`                |
| `redaction_level` | enum    | `none` \| `low` \| `medium` \| `high`                 |

## mask_map (edge-only)

- Kept server-side in the edge router; **never** serialised to the cloud.
- Keyed by `session_id`; TTL default 3600s (config: `MASK_MAP_TTL_SECONDS`).
- GC is lazy — runs on each `get_or_create` / `symbols_for` call.
- Unmasking is a textual replace of `symbol -> original` against the cloud
  response, so symbol names must not collide with natural language fragments
  (prefix `@` + kind + index guarantees this in practice).

## Example

```json
{
  "schema_version": "1.0",
  "task_id": "f4c7...",
  "task_type": "qa",
  "instruction": "draft a reply to @email_0 asking about @url_0",
  "symbols": [
    {"symbol": "@email_0", "kind": "pii",  "hint": "email-like"},
    {"symbol": "@url_0",   "kind": "url",  "hint": "https host"}
  ],
  "context_refs": ["mem://thread/abc123"],
  "privacy": {"mask_applied": true, "mask_version": "v1", "redaction_level": "medium"},
  "meta": {"classifier_confidence": 0.88}
}
```
