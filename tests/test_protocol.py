from pathlib import Path

import pytest

from dsni import protocol
from dsni.features import GATE_FEATURES
from dsni.metrics import read_result


def test_protocol_covers_the_experiments_of_the_paper():
    arms = [*protocol.NEURAL, *protocol.BASELINE]
    assert len(arms) == len(set(arms)) == 22
    assert {f"ablate_no_{f}" for f in GATE_FEATURES} <= set(arms)
    dropped = protocol.NEURAL["ablate_no_region"]["model.gate_features"]
    assert dropped == ["position", "gc", "entropy"]
    assert protocol.NEURAL["fewshot_gated_0.05"]["data.train_fraction"] == 0.05
    assert protocol.NEURAL["crop440_gated"]["data.crop"] == "center"


def test_run_executes_the_selected_arms_and_aggregates(toy_path, tmp_path):
    overrides = {
        "baseline": {"n_estimators": 5},
        "neural": {
            "data.max_len": 96,
            "data.batch_size": 16,
            "model.d_model": 16,
            "model.n_media": 4,
            "trainer.max_epochs": 1,
            "trainer.accelerator": "cpu",
            "trainer.enable_progress_bar": False,
        },
        "LitTransformer": {"model.n_layers": 1, "model.n_heads": 2, "model.d_ff": 32},
    }
    arms = ["random_kmer_rf", "crop440_gated", "fewshot_flat_0.50"]
    results = protocol.run(tmp_path, [0, 1], arms, Path(toy_path), overrides)
    assert set(results["arm"]) == set(arms) and set(results["seed"]) == {0, 1}
    assert (tmp_path / "leaderboard.md").exists()
    for arm in arms:  # every result records the table it was trained on
        data = read_result(tmp_path / arm / "s0" / "result.pt")["meta"]["data"]
        assert data["source"] == "toy.parquet" and set(data["rows"]) == {"train", "val", "test"}


def test_run_refuses_to_fall_back_to_the_sample(tmp_path):
    with pytest.raises(FileNotFoundError, match="Build the full table"):
        protocol.run(tmp_path / "out", path=tmp_path / "missing.parquet")
