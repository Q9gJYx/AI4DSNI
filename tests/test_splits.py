import numpy as np
import pandas as pd
import pytest

from dsni.splits import greedy_stratified_split, random_split


def table(n_genera=200, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.zipf(1.6, n_genera).clip(max=150)
    genus = np.repeat([f"g{i:03d}" for i in range(n_genera)], sizes)
    n = len(genus)
    label = lambda options, p: pd.Series(rng.choice(np.array(options, dtype=object), n, p=p))  # noqa: E731
    return pd.DataFrame(
        {
            "genus": genus,
            "media": label(["a", "b", None], [0.5, 0.3, 0.2]),
            "temperature": label(["meso", "thermo"], [0.9, 0.1]),
            "ph": label(["neutral", None], [0.7, 0.3]),
            "oxygen": label(["aerobe", "anaerobe", None], [0.5, 0.2, 0.3]),
        }
    )


def test_split_is_genus_disjoint_and_near_target():
    df = table()
    split = greedy_stratified_split(df)
    assert df.groupby("genus").apply(lambda g: split.loc[g.index].nunique()).max() == 1
    shares = split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
    assert shares["val"] == pytest.approx(0.15, abs=0.03)
    assert shares["test"] == pytest.approx(0.15, abs=0.03)


def test_split_stratifies_labels_and_missingness():
    df = table(n_genera=400, seed=3)
    split = greedy_stratified_split(df)
    for task in ("oxygen", "media"):
        shares = pd.crosstab(df[task].fillna("missing"), split, normalize="columns")
        assert (shares.max(axis=1) - shares.min(axis=1)).max() < 0.06


def test_split_is_deterministic_and_ignores_row_order():
    df = table()
    first = greedy_stratified_split(df)
    shuffled = df.sample(frac=1.0, random_state=1)
    second = greedy_stratified_split(shuffled)
    assert first.equals(greedy_stratified_split(df))
    assert first.equals(second.loc[df.index])


def test_split_rejects_missing_groups_and_bad_ratios():
    df = table(n_genera=20)
    with pytest.raises(ValueError, match="sum to 1"):
        greedy_stratified_split(df, ratios=(0.5, 0.2, 0.2))
    df.loc[0, "genus"] = None
    with pytest.raises(ValueError, match="missing"):
        greedy_stratified_split(df)


def test_random_split_shares():
    shares = random_split(table(n_genera=400), seed=0).value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.03)
