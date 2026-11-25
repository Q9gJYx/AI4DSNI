"""Genus-disjoint train/val/test split with greedy stratification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dsni.schema import SPLIT_RATIOS, SPLITS, TASKS


def greedy_stratified_split(
    table: pd.DataFrame,
    ratios: Sequence[float] = SPLIT_RATIOS,
    tasks: Sequence[str] = TASKS,
    group_col: str = "genus",
    label_weight: float = 0.1,
) -> pd.Series:
    """Assign every group (genus) to one split, largest group first.

    The deviation of a split from its targets is a chi-square statistic: squared difference
    between observed and expected counts over the expected count, for the number of rows
    plus ``label_weight`` times the same for every label class (missing counts as a class,
    so missingness is stratified too). Each group goes to the split where adding it lowers
    the total deviation most. The procedure has no randomness: ties in group size are
    broken by group name.

    Args:
        table: One row per strain with ``group_col`` and the ``tasks`` columns.
        ratios: Target shares of train, val and test; must sum to 1.
        tasks: Label columns to stratify over.
        group_col: Column whose values must not cross splits.
        label_weight: Weight of the per-class deviation relative to the size deviation.

    Returns:
        The split name of every row, aligned to ``table.index``.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    if table[group_col].isna().any():
        raise ValueError(f"{group_col} has missing values; drop or fill them before splitting")

    onehot = pd.get_dummies(table[list(tasks)].astype(object), dummy_na=True, dtype=float)
    onehot.insert(0, "_rows", 1.0)
    per_group = onehot.groupby(table[group_col].to_numpy()).sum().sort_index()
    per_group = per_group.sort_values("_rows", ascending=False, kind="stable")

    counts = per_group.to_numpy()
    total = counts.sum(axis=0)
    weight = np.r_[1.0, np.full(counts.shape[1] - 1, label_weight)]
    target = np.outer(ratios, np.maximum(total, 1.0))
    filled = np.zeros_like(target)
    choice = np.empty(len(counts), dtype=int)
    for i, row in enumerate(counts):
        # Change in sum((filled - target)^2 / target) if this group joins each split.
        change = (row * (row + 2 * (filled - target)) / target * weight).sum(axis=1)
        choice[i] = int(change.argmin())
        filled[choice[i]] += row

    group_to_split = dict(zip(per_group.index, np.asarray(SPLITS)[choice], strict=True))
    return table[group_col].map(group_to_split).rename("split")


def random_split(
    table: pd.DataFrame, ratios: Sequence[float] = SPLIT_RATIOS, seed: int = 0
) -> pd.Series:
    """Row-level random split that ignores taxonomy; only for measuring leakage."""
    rng = np.random.default_rng(seed)
    names = rng.choice(np.asarray(SPLITS), size=len(table), p=list(ratios))
    return pd.Series(names, index=table.index, name="split")
