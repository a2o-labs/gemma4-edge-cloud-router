import os

os.environ["V15_MOCK_MODE"] = "true"
os.environ["V15_DEVICE"] = "cpu"

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_train_v15_mock_mode_runs_2_steps(tmp_path):
    from training.train_v15 import TrainConfig, train

    cfg = TrainConfig(
        mock_mode=True,
        max_steps=2,
        log_every=1,
        save_every=2,
        output_dir=str(tmp_path / "outputs"),
        device="cpu",
    )
    final = train(cfg)
    assert final["step"] == 2
    assert "projection_state" in final
    assert "mlp_state" in final
    assert len(final["losses"]) == 2
    assert all(loss > 0 for loss in final["losses"])
    assert (tmp_path / "outputs" / "ckpt-step2.pt").exists()
    assert (tmp_path / "outputs" / "final.pt").exists()


def test_train_v15_only_adapter_params_get_gradients():
    """Verify frozen-base invariant during training: only projection +
    mlp params have non-zero grads after a step."""
    from training.train_v15 import TrainConfig, build_pipeline, step_loss

    cfg = TrainConfig(mock_mode=True, max_steps=1, device="cpu")
    edge, adapter, device, _dtype = build_pipeline(cfg)
    edge.freeze_base()
    adapter.freeze_base()

    sample = {"prompt": "test", "target_response": "ok"}
    loss = step_loss(edge, adapter, sample, device)
    loss.backward()

    for p in edge.base.parameters():
        assert p.grad is None
    for p in adapter.cloud.parameters():
        assert p.grad is None

    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in edge.projection.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in adapter.mlp.parameters()
    )


def test_train_config_defaults():
    from training.train_v15 import TrainConfig

    cfg = TrainConfig()
    assert cfg.embedding_dim == 4096
    assert cfg.prompt_tokens == 8
    assert cfg.batch_size == 4
