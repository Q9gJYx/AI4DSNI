"""Metrics shared by every arm, and the result artefact every arm writes.

A result lives at ``out/<version>/<arm>/s<seed>/result.pt`` and holds ``meta``, ``metrics``
and ``preds`` for the val and test splits. Evaluation code reads nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from dsni.data import encode_labels
from dsni.schema import IGNORE_INDEX, TASKS

TOP_K = (5, 10, 25)
RESULT = "result.pt"  # file name of a run's result inside its run directory


def joint_rank(proba: Mapping[str, np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Rank (0 is best) of the true condition tuple among all joint combinations.

    The joint score of a combination is the product of the four per-task probabilities.
    Combinations that tie with the true tuple share the average of the ranks they span, so a
    tie is never resolved in the model's favour. Only rows with all four labels are ranked;
    the result has one entry per such row.
    """
    complete = (labels != IGNORE_INDEX).all(axis=1)
    if not complete.any():
        return np.empty(0, dtype=float)
    y = labels[complete]
    logp = [np.log(np.clip(proba[t][complete], 1e-12, None)).astype(np.float32) for t in TASKS]
    joint = (
        logp[0][:, :, None, None, None]
        + logp[1][:, None, :, None, None]
        + logp[2][:, None, None, :, None]
        + logp[3][:, None, None, None, :]
    )
    true = joint[np.arange(len(y)), y[:, 0], y[:, 1], y[:, 2], y[:, 3]][:, None]
    flat = joint.reshape(len(y), -1)
    better, tied = (flat > true).sum(axis=1), (flat == true).sum(axis=1)  # tied counts itself
    return better + (tied - 1) / 2


def evaluate(proba: Mapping[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    """The paper's metrics for one split.

    Args:
        proba: Class probabilities ``(n, n_classes)`` of every task.
        labels: Class codes ``(n, 4)`` in ``TASKS`` order, ``IGNORE_INDEX`` where missing.

    Returns:
        Per task ``<task>_acc`` and ``<task>_f1`` (macro) over the rows that have the label,
        ``media_top3``, ``joint_acc`` (all four correct, over the ``n_complete`` rows that
        have all four labels) and ``joint_recall@k``.
    """
    out: dict[str, float] = {}
    hit = np.ones(len(labels), dtype=bool)
    for i, task in enumerate(TASKS):
        y, pred = labels[:, i], proba[task].argmax(axis=1)
        known = y != IGNORE_INDEX
        hit &= pred == y
        if not known.any():
            continue
        out[f"{task}_acc"] = float(accuracy_score(y[known], pred[known]))
        out[f"{task}_f1"] = float(f1_score(y[known], pred[known], average="macro"))
        if task == "media":
            classes = np.arange(proba[task].shape[1])
            top3 = top_k_accuracy_score(y[known], proba[task][known], k=3, labels=classes)
            out["media_top3"] = float(top3)
    complete = (labels != IGNORE_INDEX).all(axis=1)
    out["n_complete"] = float(complete.sum())
    out["joint_acc"] = float(hit[complete].mean()) if complete.any() else float("nan")
    rank = joint_rank(proba, labels)
    for k in TOP_K:
        out[f"joint_recall@{k}"] = float((rank < k).mean()) if len(rank) else float("nan")
    return out


def majority_baseline(
    train_labels: np.ndarray, n_classes: Mapping[str, int]
) -> dict[str, np.ndarray]:
    """Class priors of the training labels, as constant probabilities for one row."""
    priors = {}
    for i, task in enumerate(TASKS):
        y = train_labels[:, i]
        counts = np.bincount(y[y != IGNORE_INDEX], minlength=n_classes[task]).astype(float)
        priors[task] = counts / max(counts.sum(), 1.0)
    return priors


def write_result(
    run_dir: str | Path,
    seed: int,
    config: dict[str, Any],
    tables: Mapping[str, pd.DataFrame],
    proba: Mapping[str, Mapping[str, np.ndarray]],
) -> Path:
    """Score one run and save it as ``<run_dir>/result.pt``.

    Args:
        run_dir: ``out/<version>/<arm>/s<seed>``; the arm name is read from it.
        seed: Seed of the run.
        config: What produced the run; plain values only.
        tables: The rows of every split, as returned by :func:`dsni.data.load_splits`.
        proba: For every evaluated split, the class probabilities of every task.

    Returns:
        The path of the result, which holds ``meta`` (arm, seed, config, and the file, hash
        and row counts of the table the run used), ``metrics`` and ``preds`` per split.
    """
    metrics, preds = {}, {}
    for split, by_task in proba.items():
        labels = encode_labels(tables[split])
        metrics[split] = evaluate(by_task, labels.numpy())
        preds[split] = {
            "proba": {task: torch.as_tensor(by_task[task]) for task in TASKS},
            "labels": labels,
            "strain_id": tables[split]["strain_id"].tolist(),
        }
    attrs = next(iter(tables.values())).attrs
    data = {
        "source": attrs.get("source"),
        "hash": attrs.get("fingerprint"),
        "rows": {split: len(part) for split, part in tables.items()},
    }
    meta = {"arm": Path(run_dir).parent.name, "seed": seed, "config": config, "data": data}
    path = Path(run_dir) / RESULT
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "metrics": metrics, "preds": preds}, path)
    return path


def reusable(
    run_dir: str | Path, seed: int, config: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> Path | None:
    """The existing result of a run, if this seed, configuration and table produced it.

    Returns ``None`` when the run has no result yet. Raises when a result exists but came
    from another seed, configuration or table: skipping the run would silently mix
    experiments. ``config`` lists the keys to compare; other stored keys are ignored.
    """
    path = Path(run_dir) / RESULT
    if not path.exists():
        return None
    meta = torch.load(path, weights_only=True)["meta"]
    asked = {
        "seed": seed,
        "config": dict(config),
        "table": next(iter(tables.values())).attrs.get("fingerprint"),
    }
    found = {
        "seed": meta["seed"],
        "config": {key: meta["config"].get(key) for key in config},
        "table": meta["data"].get("hash"),
    }
    if asked != found:
        changed = [key for key in asked if asked[key] != found[key]]
        raise RuntimeError(
            f"{path} exists but was produced with a different {' and '.join(changed)}. "
            "Use a new version directory, or delete the old result."
        )
    print(f"skip: {path} exists")
    return path


def read_result(path: str | Path) -> dict[str, Any]:
    """Load one run; probabilities and labels come back as numpy arrays."""
    result = torch.load(path, weights_only=True)
    for p in result["preds"].values():
        p["proba"] = {t: v.numpy() for t, v in p["proba"].items()}
        p["labels"] = p["labels"].numpy()
    return result


def aggregate(version_dir: Path) -> pd.DataFrame:
    """Collect every ``<arm>/s<seed>/result.pt`` under ``version_dir`` in long format.

    Also writes ``results.parquet`` and ``leaderboard.md`` (test metrics, mean and standard
    deviation over seeds) next to the runs. Raises if the runs do not all come from the same
    table, because their numbers would not be comparable.
    """
    frames, tables = [], set()
    for path in sorted(version_dir.glob(f"*/s*/{RESULT}")):
        result = read_result(path)
        tables.add(result["meta"]["data"].get("hash"))
        long = pd.DataFrame(result["metrics"]).rename_axis("metric").reset_index()
        long = long.melt("metric", var_name="split")
        frames.append(long.assign(arm=result["meta"]["arm"], seed=result["meta"]["seed"]))
    if len(tables) > 1:
        raise RuntimeError(f"runs under {version_dir} come from {len(tables)} different tables")
    results = pd.concat(frames, ignore_index=True)[["arm", "seed", "split", "metric", "value"]]
    results.to_parquet(version_dir / "results.parquet", index=False)
    test = results[results["split"] == "test"]
    board = test.pivot_table(index="arm", columns="metric", values="value", aggfunc=["mean", "std"])
    (version_dir / "leaderboard.md").write_text(board.round(4).to_markdown() + "\n")
    return results


def bootstrap(
    proba: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Percentile intervals of :func:`evaluate` from resampling whole groups (genera).

    Returns:
        One row per metric with ``value``, ``low`` and ``high`` (2.5th and 97.5th percentile).
    """
    rng = np.random.default_rng(seed)
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.integers(len(names), size=len(names))])
        draws.append(evaluate({t: p[rows] for t, p in proba.items()}, labels[rows]))
    spread = pd.DataFrame(draws).quantile([0.025, 0.975]).T
    table = pd.DataFrame({"value": evaluate(proba, labels)})
    return table.assign(low=spread[0.025], high=spread[0.975])


def paired_bootstrap(
    proba_a: Mapping[str, np.ndarray],
    proba_b: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    metric: str = "joint_acc",
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Difference ``b - a`` in one metric, with both models scored on the same resampled groups.

    Returns:
        ``difference`` on the full data, its 2.5th and 97.5th percentiles over the draws
        (``low``, ``high``), and ``share_b_ahead``, the share of draws in which ``b`` is ahead.
    """
    rng = np.random.default_rng(seed)
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    gaps = []
    for _ in range(n_boot):
        rows = np.concatenate([members[g] for g in rng.integers(len(names), size=len(names))])
        score = [
            evaluate({t: p[rows] for t, p in m.items()}, labels[rows])[metric]
            for m in (proba_a, proba_b)
        ]
        gaps.append(score[1] - score[0])
    full = evaluate(proba_b, labels)[metric] - evaluate(proba_a, labels)[metric]
    low, high = np.percentile(gaps, [2.5, 97.5])
    return {
        "difference": full,
        "low": low,
        "high": high,
        "share_b_ahead": float(np.mean(np.array(gaps) > 0)),
    }
