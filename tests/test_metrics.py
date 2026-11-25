import numpy as np
import pytest

from dsni import metrics
from dsni.data import encode_labels
from dsni.schema import IGNORE_INDEX, TASKS

N_CLASSES = {"media": 5, "temperature": 3, "ph": 3, "oxygen": 4}


def one_hot(labels):
    proba = {}
    for i, task in enumerate(TASKS):
        y = np.where(labels[:, i] == IGNORE_INDEX, 0, labels[:, i])
        proba[task] = np.eye(N_CLASSES[task])[y] * 0.9 + 0.1 / N_CLASSES[task]
    return proba


def random_labels(n=60, seed=0, missing=0.3):
    rng = np.random.default_rng(seed)
    labels = np.stack([rng.integers(N_CLASSES[t], size=n) for t in TASKS], axis=1)
    labels[rng.random(labels.shape) < missing] = IGNORE_INDEX
    labels[:5] = 1  # some complete rows
    return labels


def test_perfect_predictions_score_one_and_missing_labels_are_masked():
    labels = random_labels()
    scores = metrics.evaluate(one_hot(labels), labels)
    for task in TASKS:
        assert scores[f"{task}_acc"] == 1.0 and scores[f"{task}_f1"] == 1.0
    assert scores["media_top3"] == 1.0 and scores["joint_acc"] == 1.0
    assert scores["joint_recall@5"] == 1.0
    assert scores["n_complete"] == (labels != IGNORE_INDEX).all(axis=1).sum()


def test_joint_accuracy_needs_all_four_correct():
    labels = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    proba = one_hot(labels)
    proba["oxygen"] = proba["oxygen"][::-1].copy()  # wrong on both rows
    scores = metrics.evaluate(proba, labels)
    assert scores["oxygen_acc"] == 0.0 and scores["media_acc"] == 1.0 and scores["joint_acc"] == 0.0


def test_joint_rank_counts_better_combinations():
    labels = np.array([[0, 0, 0, 0], [2, 1, 0, IGNORE_INDEX]])
    proba = {t: np.full((2, N_CLASSES[t]), 1.0 / N_CLASSES[t]) for t in TASKS}
    proba["media"] = np.array([[0.1, 0.6, 0.3, 0.0, 0.0]] * 2)  # true medium ranks third
    rank = metrics.joint_rank(proba, labels)
    assert rank.shape == (1,)  # the incomplete row is not ranked
    # 72 combinations hold a better medium; the 36 with the true medium tie and share ranks.
    assert rank[0] == 2 * 36 + (36 - 1) / 2


def test_ties_never_count_as_hits():
    labels = np.array([[0, 0, 0, 0], [1, 2, 1, 3]])
    uniform = {t: np.full((2, N_CLASSES[t]), 1.0 / N_CLASSES[t]) for t in TASKS}
    scores = metrics.evaluate(uniform, labels)
    assert scores["joint_recall@5"] == 0.0 and scores["joint_recall@25"] == 0.0


def test_no_complete_rows_gives_nan_joint_metrics():
    labels = np.array([[0, IGNORE_INDEX, 0, 0]])
    scores = metrics.evaluate(one_hot(labels), labels)
    assert np.isnan(scores["joint_acc"]) and scores["n_complete"] == 0
    assert "temperature_acc" not in scores


def test_majority_baseline_is_the_training_prior():
    labels = np.array([[0, 1, IGNORE_INDEX, 3], [0, 1, 2, 3], [1, 0, 2, IGNORE_INDEX]])
    priors = metrics.majority_baseline(labels, N_CLASSES)
    assert priors["media"][:2].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert priors["ph"][2] == 1.0 and priors["oxygen"][3] == 1.0


def test_result_round_trip_and_aggregate(toy_table, tmp_path):
    tables = {split: toy_table[toy_table["split"] == split] for split in ("val", "test")}
    labels = {split: encode_labels(part).numpy() for split, part in tables.items()}
    n_media = len(toy_table["media"].cat.categories)
    proba = {
        split: {t: np.eye(n_media if t == "media" else N_CLASSES[t])[np.maximum(y[:, i], 0)]
                for i, t in enumerate(TASKS)}
        for split, y in labels.items()
    }  # fmt: skip
    for seed in (0, 1):
        path = metrics.write_result(tmp_path / "arm_a" / f"s{seed}", seed, {"k": 1}, tables, proba)
    result = metrics.read_result(path)
    meta = result["meta"]
    assert (meta["arm"], meta["seed"], meta["config"]) == ("arm_a", 1, {"k": 1})
    assert meta["data"]["rows"] == {split: len(part) for split, part in tables.items()}
    assert result["metrics"]["test"]["joint_acc"] == 1.0
    assert np.array_equal(result["preds"]["test"]["labels"], labels["test"])
    assert result["preds"]["test"]["strain_id"] == tables["test"]["strain_id"].tolist()
    long = metrics.aggregate(tmp_path)
    assert list(long.columns) == ["arm", "seed", "split", "metric", "value"]
    assert long["seed"].nunique() == 2 and set(long["split"]) == {"val", "test"}
    assert (tmp_path / "results.parquet").exists()
    assert "arm_a" in (tmp_path / "leaderboard.md").read_text()


def test_bootstrap_brackets_the_point_estimate():
    labels = random_labels(n=200, seed=2)
    rng = np.random.default_rng(0)
    proba = {t: rng.dirichlet(np.ones(N_CLASSES[t]), size=200) for t in TASKS}
    table = metrics.bootstrap(proba, labels, groups=np.arange(200) // 10, n_boot=40)
    row = table.loc["oxygen_acc"]
    assert row["low"] <= row["value"] <= row["high"] and row["high"] - row["low"] > 0


def test_a_result_is_reused_only_for_the_same_seed_recipe_and_table(toy_table, tmp_path, capsys):
    tables = {split: toy_table[toy_table["split"] == split] for split in ("val", "test")}
    for part in tables.values():
        part.attrs["fingerprint"] = "table-a"
    proba = {s: {t: np.eye(9)[np.zeros(len(p), int)] for t in TASKS} for s, p in tables.items()}
    run_dir = tmp_path / "arm" / "s0"
    assert metrics.reusable(run_dir, 0, {"k": 1}, tables) is None  # nothing there yet
    path = metrics.write_result(run_dir, 0, {"k": 1, "checkpoint": "x.ckpt"}, tables, proba)
    assert metrics.reusable(run_dir, 0, {"k": 1}, tables) == path  # extra stored keys ignored
    assert "skip" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="different config"):
        metrics.reusable(run_dir, 0, {"k": 2}, tables)
    with pytest.raises(RuntimeError, match="different seed"):
        metrics.reusable(run_dir, 1, {"k": 1}, tables)
    for part in tables.values():
        part.attrs["fingerprint"] = "table-b"
    with pytest.raises(RuntimeError, match="different table"):
        metrics.reusable(run_dir, 0, {"k": 1}, tables)
    metrics.write_result(tmp_path / "other" / "s0", 0, {"k": 1}, tables, proba)
    with pytest.raises(RuntimeError, match="2 different tables"):
        metrics.aggregate(tmp_path)


def test_paired_bootstrap_of_a_difference():
    labels = random_labels(n=300, seed=4, missing=0.0)
    good, bad = one_hot(labels), one_hot(labels)
    bad["oxygen"] = bad["oxygen"][::-1].copy()  # wrong on most rows
    gap = metrics.paired_bootstrap(bad, good, labels, np.arange(300) // 10, n_boot=50)
    assert gap["difference"] > 0 and gap["low"] > 0 and gap["share_b_ahead"] == 1.0
    same = metrics.paired_bootstrap(good, good, labels, np.arange(300) // 10, n_boot=20)
    assert same["difference"] == 0 and same["share_b_ahead"] == 0.0
