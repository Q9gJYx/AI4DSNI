"""The paper's experimental protocol as data: every arm is a run directory and its options.

:func:`run` executes the arms over seeds. Nothing in this repository has run it at full
scale; ``scripts/smoke_test.py`` runs this same table at toy size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from dsni import baselines, lit, metrics
from dsni.data import FULL
from dsni.features import GATE_FEATURES

FLAT = {"model": "LitTransformer", "model.gate": "none"}
GATED = {"model": "LitTransformer", "model.gate": "bio"}
FRACTIONS = (0.05, 0.10, 0.25, 0.50)

# Neural arms: options of scripts/train.py on top of the paper-scale defaults.
NEURAL: dict[str, dict[str, Any]] = {
    # Main table.
    "cnn": {"model": "LitCNN"},
    "transformer_flat": FLAT,
    "transformer_gated": GATED,
    # Gate ablations: drop one biological feature, or learn the gate from position bins only.
    **{
        f"ablate_no_{dropped}": {
            **GATED,
            "model.gate_features": [f for f in GATE_FEATURES if f != dropped],
        }
        for dropped in GATE_FEATURES
    },
    "learned_gate": {"model": "LitTransformer", "model.gate": "learned"},
    # Few-shot: share of the training strains kept.
    **{f"fewshot_flat_{f:.2f}": {**FLAT, "data.train_fraction": f} for f in FRACTIONS},
    **{f"fewshot_gated_{f:.2f}": {**GATED, "data.train_fraction": f} for f in FRACTIONS},
    # Leakage: the same arms on a row-level random split.
    "random_flat": {**FLAT, "data.split": "random"},
    "random_gated": {**GATED, "data.split": "random"},
    # Truncation: a 440-token centre crop.
    "crop440_flat": {**FLAT, "data.max_len": 440, "data.crop": "center"},
    "crop440_gated": {**GATED, "data.max_len": 440, "data.crop": "center"},
}

# Classical arms: keyword arguments of dsni.baselines.run.
BASELINE: dict[str, dict[str, Any]] = {
    "kmer_rf": {},
    "random_kmer_rf": {"split": "random"},
}


def run(
    version_dir: Path = Path("out/v1"),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    arms: Sequence[str] | None = None,
    path: Path = FULL,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run the protocol and aggregate it into ``results.parquet`` and ``leaderboard.md``.

    Every run writes ``<version_dir>/<arm>/s<seed>/result.pt`` and is skipped when that file
    exists, so a campaign can be stopped and resumed.

    Args:
        version_dir: Where the runs go.
        seeds: Seeds of every arm.
        arms: Arms to run; ``None`` runs all of :data:`NEURAL` and :data:`BASELINE`.
        path: The table to run on. It defaults to the full table and must exist: the protocol
            never falls back to the tracked sample on its own.
        overrides: Extra options applied on top of every matching arm, keyed by ``"baseline"``,
            ``"neural"`` or a model class name. The smoke test shrinks every run this way.

    Returns:
        The long results table of :func:`dsni.metrics.aggregate`.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Build the full table with scripts/build_dsni.py, or pass "
            "another table explicitly with `path`."
        )
    extra = overrides or {}
    for seed in seeds:
        for arm, options in BASELINE.items():
            if arms is None or arm in arms:
                kwargs = {"path": str(path), **options, **extra.get("baseline", {})}
                baselines.run(version_dir / arm / f"s{seed}", seed=seed, **kwargs)
        for arm, options in NEURAL.items():
            if arms is None or arm in arms:
                flags = {"data.path": str(path), **options, **extra.get("neural", {})}
                flags |= extra.get(options["model"], {})
                lit.fit(version_dir / arm / f"s{seed}", **flags, seed_everything=seed)
    return metrics.aggregate(version_dir)
