import numpy as np
import pytest

from dsni import baselines
from dsni.metrics import read_result
from dsni.splits import random_split


def test_kmer_features_are_the_1344_normalised_counts():
    features = baselines.kmer_features().fit_transform(["ACGTACGTAC", "GGGGGGNNGG"])
    assert features.shape == (2, 1344)
    assert np.allclose(np.sqrt(features.multiply(features).sum(axis=1)), 1.0)
    assert features[0, baselines.KMERS.index("ACG")] > 0
    assert features[1, baselines.KMERS.index("ACG")] == 0


def test_run_writes_the_shared_artefact_once(toy_path, tmp_path, capsys):
    run_dir = tmp_path / "v0" / "kmer_rf" / "s0"
    result = read_result(baselines.run(run_dir, toy_path, n_estimators=10))
    assert result["meta"]["arm"] == "kmer_rf"
    assert result["preds"]["test"]["proba"]["media"].shape[1] == 4
    assert np.isfinite(result["metrics"]["test"]["oxygen_acc"])
    capsys.readouterr()
    assert baselines.run(run_dir, toy_path, n_estimators=10) == run_dir / "result.pt"
    assert "skip" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="different config"):
        baselines.run(run_dir, toy_path, n_estimators=20)


def test_random_split_leaves_closer_training_neighbours(toy_table):
    rng = np.random.default_rng(0)
    genera = toy_table["genus"].unique()
    per_genus = {g: "".join(rng.choice(list("ACGT"), size=300)) for g in genera}
    table = toy_table.assign(sequence=toy_table["genus"].map(per_genus))  # one per genus
    genus = baselines.nearest_train_similarity(table)
    leaky = baselines.nearest_train_similarity(table, random_split(table, seed=0))
    assert genus.shape == ((table["split"] == "test").sum(),)
    assert np.median(leaky) > np.median(genus)
