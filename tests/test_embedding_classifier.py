"""Cosine-similarity nearest-neighbor classifier — unit tests.

Standalone module: no torch / transformers / encoder required. Pure
numpy. Tests cover hit / empty / negative-cosine / mixed-magnitude /
inconsistent-dim cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from router.embedding_classifier import EmbeddingClassifier


def test_classify_returns_nearest_exemplar() -> None:
    clf = EmbeddingClassifier()
    clf.add(np.array([1.0, 0.0, 0.0]), "qa", "light", "ex-1")
    clf.add(np.array([0.0, 1.0, 0.0]), "code", "heavy", "ex-2")

    r = clf.classify(np.array([0.95, 0.05, 0.0]))
    assert r.task_type == "qa"
    assert r.complexity == "light"
    assert r.nearest_id == "ex-1"
    assert r.confidence > 0.9


def test_empty_classifier_raises() -> None:
    clf = EmbeddingClassifier()
    with pytest.raises(RuntimeError):
        clf.classify(np.array([1.0, 0.0]))


def test_negative_cosine_clipped_to_zero_confidence() -> None:
    clf = EmbeddingClassifier()
    clf.add(np.array([1.0, 0.0]), "qa", "light", "ex-1")
    r = clf.classify(np.array([-1.0, 0.0]))
    assert r.confidence == 0.0
    assert r.nearest_id == "ex-1"


def test_add_many_and_len() -> None:
    clf = EmbeddingClassifier()
    clf.add_many(
        [
            (np.array([1.0, 0.0]), "qa", "light", "a"),
            (np.array([0.0, 1.0]), "code", "heavy", "b"),
            (np.array([0.7, 0.7]), "summarize", "light", "c"),
        ]
    )
    assert len(clf) == 3


def test_consistent_dim_required() -> None:
    """Mixing dims should raise on classify after lazy matrix rebuild."""
    clf = EmbeddingClassifier()
    clf.add(np.array([1.0, 0.0]), "qa", "light", "a")
    clf.add(np.array([1.0, 0.0, 0.0]), "code", "heavy", "b")
    with pytest.raises(Exception):
        clf.classify(np.array([1.0, 0.0]))


def test_normalization_handled_at_add_time() -> None:
    """Vectors are normalized at add; classify accepts a raw query."""
    clf = EmbeddingClassifier()
    clf.add(np.array([100.0, 0.0]), "qa", "light", "a")
    clf.add(np.array([0.0, 0.001]), "code", "heavy", "b")

    r = clf.classify(np.array([1.0, 0.0]))
    assert r.task_type == "qa"
    assert r.confidence > 0.99
