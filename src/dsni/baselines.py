"""The classical ``k-mer + RF`` arm: k-mer counts into one random forest per task."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from dsni.data import encode_labels, load_splits, num_classes
from dsni.metrics import reusable, write_result
from dsni.schema import IGNORE_INDEX, TASKS

KMERS = ["".join(p) for k in (3, 4, 5) for p in product("ACGT", repeat=k)]  # 1,344 features


def kmer_features() -> Pipeline:
    """Counts of every 3-, 4- and 5-mer over A, C, G, T, L2-normalised per sequence."""
    counter = CountVectorizer(
        analyzer="char", ngram_range=(3, 5), vocabulary=KMERS, lowercase=False
    )
    return make_pipeline(counter, Normalizer(norm="l2"))


def nearest_train_similarity(table: pd.DataFrame, split: pd.Series | None = None) -> np.ndarray:
    """Cosine similarity of every test strain's k-mer profile to its closest training strain.

    Args:
        table: A DSNI table.
        split: Split names aligned to ``table``; defaults to the released ``split`` column.
    """
    split = table["split"] if split is None else split
    features = kmer_features()
    train = features.fit_transform(table.loc[split == "train", "sequence"]).toarray()
    test = features.transform(table.loc[split == "test", "sequence"]).toarray()
    return (test @ train.T).max(axis=1)  # rows have unit length, so the product is the cosine


def run(
    run_dir: Path = Path("out/dev/kmer_rf/s0"),
    path: str | None = None,
    seed: int = 0,
    n_estimators: int = 500,
    split: str = "genus",
    train_fraction: float = 1.0,
) -> Path:
    """Fit ``k-mer + RF`` and write the same result file as the neural arms.

    Args:
        run_dir: ``out/<version>/<arm>/s<seed>``; a run whose result exists is not repeated
            (see :func:`dsni.metrics.reusable`).
        path: Parquet table; ``None`` resolves as in :func:`dsni.data.load`.
        seed: Seed of the forests, the random split and the training subsample.
        n_estimators: Trees per forest.
        split: ``"genus"`` or ``"random"``, see :func:`dsni.data.load_splits`.
        train_fraction: Share of the training rows kept.

    Returns:
        The path of ``result.pt``.
    """
    config = {"n_estimators": n_estimators, "split": split, "train_fraction": train_fraction}
    tables = load_splits(path, split, train_fraction, seed)
    if done := reusable(run_dir, seed, config, tables):
        return done
    n_classes = num_classes(tables["train"])
    features = kmer_features()
    x_train = features.fit_transform(tables["train"]["sequence"])
    y_train = encode_labels(tables["train"]).numpy()
    forests = {}
    for i, task in enumerate(TASKS):
        known = y_train[:, i] != IGNORE_INDEX
        forest = RandomForestClassifier(n_estimators, n_jobs=-1, random_state=seed)
        forests[task] = forest.fit(x_train[known], y_train[known, i])

    proba: dict[str, dict[str, np.ndarray]] = {}
    for name in ("val", "test"):
        x = features.transform(tables[name]["sequence"])
        proba[name] = {}
        for task, forest in forests.items():  # a forest only knows the classes it has seen
            proba[name][task] = np.zeros((x.shape[0], n_classes[task]))
            proba[name][task][:, forest.classes_] = forest.predict_proba(x)
    return write_result(run_dir, seed, config, tables, proba)
