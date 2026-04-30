"""Embedding-similarity classifier.

Takes a V1.5 edge embedding (numpy float16/32 1D array) and returns a
predicted ``task_type`` + ``complexity`` by nearest-neighbor cosine
similarity against a small in-memory exemplar bank.

The exemplar bank is loaded from ``eval/data/eval_v15.jsonl`` (or any
JSONL with ``prompt`` / ``task_type`` / ``expected_complexity`` fields)
at initialization: each sample's ``prompt`` is run through a provided
encoder to populate the bank. Tests can also build a bank directly
via ``add()`` / ``add_many()``.

This is much faster than LLM-as-judge: <5ms per query for 60 exemplars
on a single numpy dot product, vs ~500ms for an LLM round-trip.

Standalone module — not wired into ``V15Pipeline`` yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

TaskType = Literal[
    "qa", "code", "summarize", "translate", "reason", "chat", "other"
]
Complexity = Literal["light", "heavy"]


@dataclass
class Exemplar:
    """One labeled exemplar in the bank. ``vec`` is L2-normalized."""

    prompt: str
    vec: np.ndarray
    task_type: TaskType
    complexity: Complexity


@dataclass
class ClassifyResult:
    task_type: TaskType
    complexity: Complexity
    confidence: float
    nearest_id: str


class EmbeddingClassifier:
    """Cosine-similarity nearest-neighbor classifier.

    The exemplar bank is a list of ``(vec, task_type, complexity, id)``.
    Vectors are L2-normalized at insert time so cosine similarity
    reduces to a single dot product against a stacked matrix.
    """

    def __init__(self) -> None:
        self._exemplars: list[
            tuple[np.ndarray, TaskType, Complexity, str]
        ] = []
        self._matrix: np.ndarray | None = None

    def add(
        self,
        vec: np.ndarray,
        task_type: TaskType,
        complexity: Complexity,
        id_: str,
    ) -> None:
        v = vec.astype(np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        self._exemplars.append((v, task_type, complexity, id_))
        self._matrix = None

    def add_many(
        self,
        items: Iterable[tuple[np.ndarray, TaskType, Complexity, str]],
    ) -> None:
        for it in items:
            self.add(*it)

    def _rebuild_matrix(self) -> None:
        if not self._exemplars:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        D = self._exemplars[0][0].shape[0]
        self._matrix = np.stack([e[0] for e in self._exemplars]).astype(np.float32)
        if self._matrix.shape[1] != D:
            raise ValueError(
                f"inconsistent exemplar dim: matrix shape {self._matrix.shape}"
            )

    def classify(self, query_vec: np.ndarray) -> ClassifyResult:
        if not self._exemplars:
            raise RuntimeError("no exemplars loaded; call add() first")
        if self._matrix is None:
            self._rebuild_matrix()

        q = query_vec.astype(np.float32)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n

        sims = self._matrix @ q
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        _, task_type, complexity, id_ = self._exemplars[best_idx]
        conf = max(0.0, best_sim)
        return ClassifyResult(
            task_type=task_type,
            complexity=complexity,
            confidence=conf,
            nearest_id=id_,
        )

    def __len__(self) -> int:
        return len(self._exemplars)

    @classmethod
    def from_jsonl_with_encoder(
        cls,
        jsonl_path: str,
        encoder,
    ) -> "EmbeddingClassifier":
        """Build a classifier by encoding every prompt in a JSONL file.

        ``encoder`` is any object with an ``encode(text, return_schema=...)
        -> (schema, vec)`` method. Records missing ``prompt`` are
        skipped; missing ``task_type`` defaults to ``"other"`` and
        missing ``expected_complexity`` defaults to ``"light"``.
        Progress is logged every 10 entries.
        """
        import json
        import logging

        log = logging.getLogger("embedding-classifier")

        clf = cls()
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = rec.get("prompt")
                tt = rec.get("task_type", "other")
                cx = rec.get("expected_complexity", "light")
                id_ = rec.get("id", f"row-{i}")
                if not prompt:
                    continue
                _, vec = encoder.encode(prompt, return_schema=False)
                clf.add(vec, tt, cx, id_)
                if (i + 1) % 10 == 0:
                    log.info("loaded %d exemplars", i + 1)
        return clf
