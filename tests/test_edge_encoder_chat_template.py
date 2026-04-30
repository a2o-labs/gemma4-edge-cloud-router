"""Regression for transformers 5.x BatchEncoding return from apply_chat_template.

Uses a tokenizer that *has* a chat_template, then exercises the full
``EdgeEncoder.encode()`` path. The existing ``tests/test_v15_pipeline.py``
uses ``trl-internal-testing/tiny-random-LlamaForCausalLM`` which has no
chat_template, so it exercises only the fallback branch — which is why
PR #3 CI was green while the L4 real-weight smoke broke on
``google/gemma-3-1b-it``.

We use ``HuggingFaceTB/SmolLM2-135M-Instruct`` as the smallest cheap
public model that ships a Jinja chat_template (~135M params,
~135MB float16, no gating). Marked ``@pytest.mark.slow`` and gated
behind ``PYTEST_SKIP_NETWORK`` so default CI without HF network access
can opt out.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

SKIP_NET = pytest.mark.skipif(
    os.environ.get("PYTEST_SKIP_NETWORK", "false").lower() == "true",
    reason="network tests skipped (PYTEST_SKIP_NETWORK=true)",
)


@SKIP_NET
@pytest.mark.slow
def test_edge_encoder_with_chat_template_tokenizer() -> None:
    """Smoke-level: load a tiny real model that has chat_template, encode.

    Regression target: on transformers 5.x,
    ``apply_chat_template(return_tensors="pt")`` returns a BatchEncoding,
    not a Tensor. ``_tokenize`` must extract ``input_ids`` so that
    ``_last_hidden``'s ``torch.ones_like(input_ids)`` and the model
    forward call see a real Tensor.
    """
    from router.edge_encoder import EdgeEncoder

    enc = EdgeEncoder(
        model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
        embedding_dim=128,
        device="cpu",
        dtype=torch.float32,
    )
    assert getattr(enc.tokenizer, "chat_template", None), (
        "test setup invariant: SmolLM2-135M-Instruct should ship a chat_template; "
        "if HF dropped it the regression is no longer covered"
    )

    schema, vec = enc.encode("Hello world")
    assert schema is not None
    assert schema.version == "1.5"
    assert vec.shape == (128,)
    assert schema.embedding_b64 is not None and len(schema.embedding_b64) > 0


@pytest.mark.slow
def test_edge_encoder_tokenize_handles_batchencoding(monkeypatch) -> None:
    """Unit-level: monkeypatch the tokenizer to fake the transformers 5.x
    BatchEncoding return, exercise ``_tokenize`` only.

    This runs without network so default CI also catches the regression
    even with PYTEST_SKIP_NETWORK=true.
    """
    from router.edge_encoder import EdgeEncoder

    class FakeBatchEncoding(dict):
        """transformers BatchEncoding is a dict subclass with .input_ids."""

        def __init__(self, input_ids):
            super().__init__(input_ids=input_ids)
            self.input_ids = input_ids

        def to(self, device):  # pragma: no cover — not exercised
            return self

    enc = EdgeEncoder.__new__(EdgeEncoder)
    enc.device = "cpu"

    fake_ids = torch.tensor([[1, 2, 3, 4]])

    class FakeTokenizer:
        chat_template = "{{ messages[0].content }}"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs.get("return_tensors") == "pt"
            assert kwargs.get("return_dict") is True
            return FakeBatchEncoding(fake_ids)

        def __call__(self, text, return_tensors=None):  # pragma: no cover
            raise AssertionError("fallback branch should not be hit")

    enc.tokenizer = FakeTokenizer()

    out = enc._tokenize("hi")
    assert torch.is_tensor(out), f"_tokenize returned non-Tensor: {type(out)}"
    assert torch.equal(out, fake_ids)
