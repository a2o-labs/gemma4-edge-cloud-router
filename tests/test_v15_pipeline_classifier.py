"""V15Pipeline + EmbeddingClassifier wiring tests (mock-mode)."""

from __future__ import annotations

import os

os.environ["V15_ENABLED"] = "true"
os.environ["V15_MOCK_MODE"] = "true"
os.environ["V15_DEVICE"] = "cpu"

import numpy as np

import router.config as config_module

config_module._settings = None

from router.config import get_settings  # noqa: E402
from router.embedding_classifier import EmbeddingClassifier  # noqa: E402
from router.v15_pipeline import V15Pipeline  # noqa: E402


def test_pipeline_without_classifier_uses_encoder_default():
    p = V15Pipeline(get_settings().v15)
    p.load()
    schema, _ = p.run("Hello")
    # Mock encoder returns task_type="other"; complexity is length-driven.
    assert schema.task_type == "other"


def test_pipeline_with_classifier_overrides_when_confident():
    """A non-zero exemplar against a zero query gives confidence 0.0,
    which is below the default 0.5 threshold — encoder default wins."""
    clf = EmbeddingClassifier()
    clf.add(np.ones(4096), "code", "heavy", "ex-1")

    p = V15Pipeline(get_settings().v15, classifier=clf)
    p.load()
    schema, _ = p.run("Hello")
    assert schema.task_type == "other"


def test_pipeline_classifier_override_with_lowered_threshold():
    """Lower the threshold to 0.0 — even a zero-confidence match overrides."""
    settings = get_settings().v15
    settings_with_low_thresh = settings.model_copy(
        update={"classifier_min_confidence": 0.0}
    )

    clf = EmbeddingClassifier()
    clf.add(np.ones(4096), "code", "heavy", "ex-1")

    p = V15Pipeline(settings_with_low_thresh, classifier=clf)
    p.load()
    schema, _ = p.run("Hello")
    assert schema.task_type == "code"
    assert schema.complexity == "heavy"


def test_pipeline_classifier_failure_falls_back_silently():
    """Classifier exceptions must not block the pipeline."""

    class _BrokenClassifier:
        def classify(self, vec):
            raise RuntimeError("simulated classifier failure")

    p = V15Pipeline(get_settings().v15, classifier=_BrokenClassifier())
    p.load()
    schema, answer = p.run("Hello")
    assert schema.task_type == "other"
    assert "[mock-v15-response" in answer
