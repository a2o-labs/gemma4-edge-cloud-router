# V1.5 evaluation dataset

`eval_v15.jsonl` — 62 JSONL samples covering qa / code / summarize /
translate / reason / chat task types, balanced light vs heavy complexity.

## Origin

All samples are **synthetic** — composed by the project team. No
copyrighted text, no PII, no real user logs. The set is designed to
exercise both the complexity classifier (light vs heavy split) and the
V1 vs V1.5 quality delta via LLM-as-judge.

## Schema

| Field                  | Type                                                                  |
|------------------------|-----------------------------------------------------------------------|
| id                     | str (e.g. `qa-011`)                                                   |
| prompt                 | str                                                                   |
| expected_answer        | str (short reference for LLM-as-judge — not a full canonical answer)  |
| expected_complexity    | `"light"` \| `"heavy"`                                                |
| task_type              | `"qa"` \| `"code"` \| `"summarize"` \| `"translate"` \| `"reason"` \| `"chat"` |

## Distribution

- Total: **62** samples
- Light/heavy split: **32 / 30** (≈52% / 48%)
- Per task type:

  | task_type | total | light | heavy |
  |-----------|-------|-------|-------|
  | qa        | 14    | 9     | 5     |
  | code      | 12    | 3     | 9     |
  | summarize | 9     | 4     | 5     |
  | translate | 9     | 7     | 2     |
  | reason    | 10    | 2     | 8     |
  | chat      | 8     | 7     | 1     |

The first 10 IDs (`qa-001..002`, `code-001..002`, `summarize-001`,
`translate-001`, `reason-001..002`, `chat-001..002`) are the original
plumbing samples and are kept byte-identical — append-only growth.

## Adding samples

Append-only. New IDs continue the per-task-type numbering (`qa-015`,
`code-013`, …). After appending, run the existing eval to confirm no
regression:

    .venv/bin/python -m pytest tests/test_eval_v15.py -v
    .venv/bin/python -m eval.eval_v15 --dataset canned --mode mock --judge mock

Light heuristic for `expected_complexity`:

- `light` — prompts < ~200 chars, single-step, factual recall, short
  greetings or single-sentence translations.
- `heavy` — prompts ≥ 200 chars, code generation, multi-step
  reasoning, or any task expecting ≥ ~100 words of output.

Keep the JSONL under 100 KB so it stays cheap to load in CI; the
current file is ~21 KB.
