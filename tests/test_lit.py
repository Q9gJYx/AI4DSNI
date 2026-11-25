import lightning as L
import pytest
import torch

from dsni.data import DSNIDataModule
from dsni.lit import LitCNN, LitTransformer, export_checkpoint, fit, masked_loss, predict_proba
from dsni.metrics import read_result
from dsni.schema import IGNORE_INDEX, TASKS

TINY = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "n_media": 4}


def trainer():
    return L.Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )


def test_masked_loss_skips_missing_labels():
    logits = {
        "media": torch.randn(4, 4),
        "temperature": torch.randn(4, 3),
        "ph": torch.randn(4, 3),
        "oxygen": torch.randn(4, 4),
    }
    labels = torch.full((4, 4), IGNORE_INDEX)
    assert masked_loss(logits, labels) == 0
    labels[0, 0] = 1
    expected = torch.nn.functional.cross_entropy(logits["media"][:1], torch.tensor([1]))
    assert masked_loss(logits, labels) == pytest.approx(expected.item())


@pytest.mark.parametrize(
    "model",
    [
        LitCNN(d_model=16, channels=(8, 16), kernels=(5, 3), n_media=4),
        LitTransformer(gate="none", **TINY),
        LitTransformer(gate="bio", **TINY),
        LitTransformer(gate="bio", gate_features=("position", "gc"), **TINY),
        LitTransformer(gate="learned", **TINY),
    ],
    ids=["cnn", "flat", "gated", "ablated", "learned"],
)
def test_every_arm_trains_one_step(model, toy_path):
    data = DSNIDataModule(toy_path, max_len=128, batch_size=16)
    run = trainer()
    run.fit(model, data)
    assert torch.isfinite(run.callback_metrics["train/loss"])
    assert set(model(torch.ones(2, 128, dtype=torch.long))) == set(TASKS)


def test_model_rejects_a_mismatched_label_space(toy_path):
    data = DSNIDataModule(toy_path, max_len=128, batch_size=16)
    with pytest.raises(ValueError, match="do not match data"):
        trainer().fit(LitCNN(d_model=16, n_media=42), data)


def test_fit_writes_the_result_and_skips_a_finished_run(toy_path, tmp_path, capsys):
    run_dir = tmp_path / "v0" / "transformer_gated" / "s3"
    options = {
        "model": "LitTransformer",
        "model.gate": "bio",
        "model.gate_features": ["region", "gc"],
        "model.d_model": 16,
        "model.n_layers": 1,
        "model.n_heads": 2,
        "model.d_ff": 32,
        "model.n_media": 4,
        "data.path": toy_path,
        "data.max_len": 96,
        "data.crop": "center",
        "data.batch_size": 16,
        "trainer.max_epochs": 1,
        "trainer.accelerator": "cpu",
        "trainer.enable_progress_bar": False,
        "seed_everything": 3,
    }
    result = read_result(fit(run_dir, **options))
    assert result["meta"]["arm"] == "transformer_gated" and result["meta"]["seed"] == 3
    assert result["meta"]["config"]["data"]["seed"] == 3  # linked from seed_everything
    assert tuple(result["meta"]["config"]["model"]["gate_features"]) == ("region", "gc")
    assert set(result["metrics"]) == {"val", "test"}
    assert len(result["preds"]["test"]["strain_id"]) == len(result["preds"]["test"]["labels"])
    log = (run_dir / "logs" / "metrics.csv").read_text()
    assert "val/joint_acc" in log and "test/joint_acc" in log
    capsys.readouterr()
    assert fit(run_dir, **options) == run_dir / "result.pt"
    assert "skip" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="different config"):
        fit(run_dir, **{**options, "model.d_ff": 64})
    with pytest.raises(RuntimeError, match="different config"):
        fit(run_dir, **{**options, "trainer.max_epochs": 2})
    assert fit(run_dir, **{**options, "trainer.enable_progress_bar": True}).exists()  # display only


def test_exported_checkpoint_reloads_and_predicts_the_same(toy_path, tmp_path):
    run_dir = tmp_path / "v0" / "transformer_gated" / "s0"
    options = {
        "model": "LitTransformer",
        "model.gate": "bio",
        "model.d_model": 16,
        "model.n_layers": 1,
        "model.n_heads": 2,
        "model.d_ff": 32,
        "model.n_media": 4,
        "data.path": toy_path,
        "data.max_len": 96,
        "data.batch_size": 16,
        "trainer.max_epochs": 1,
        "trainer.accelerator": "cpu",
        "trainer.enable_progress_bar": False,
    }
    result = read_result(fit(run_dir, **options))
    shipped = export_checkpoint(
        run_dir / "logs" / "checkpoints" / "best.ckpt", tmp_path / "w.ckpt", seed=0
    )
    assert torch.load(shipped, weights_only=False)["dsni"] == {"seed": 0}
    assert "optimizer_states" not in torch.load(shipped, weights_only=False)
    model = LitTransformer.load_from_checkpoint(shipped, map_location="cpu")
    data = DSNIDataModule(toy_path, max_len=96, batch_size=16)
    data.setup()
    proba = predict_proba(model, data.loader("test"))
    assert set(proba) == set(TASKS) and len(proba["media"]) == len(data.sets["test"])
    assert proba["media"] == pytest.approx(result["preds"]["test"]["proba"]["media"], abs=1e-5)
