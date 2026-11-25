"""Run every stage of the pipeline at toy size and check that it produces finite numbers.

The numbers mean nothing: the models are tiny and train for one epoch on a few hundred
strains. The only question is whether every stage runs. Needs no network.

Usage:
    uv run python scripts/smoke_test.py               # parser, data, every arm, aggregation
    uv run python scripts/smoke_test.py --notebooks   # also execute the demo notebook in place
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from dsni import build, data, protocol
from dsni.schema import TASKS

ROOT = Path(__file__).resolve().parents[1]


def shrink() -> dict[str, dict[str, object]]:
    """Overrides that turn every arm of the paper's protocol into a run of a few seconds."""
    return {
        "baseline": {"n_estimators": 20},
        "neural": {
            "data.max_len": 128,
            "data.batch_size": 32,
            "model.d_model": 16,
            "trainer.max_epochs": 1,
            "trainer.accelerator": "cpu",
            "trainer.enable_progress_bar": False,
            "trainer.enable_model_summary": False,
        },
        "LitTransformer": {"model.n_layers": 1, "model.n_heads": 2, "model.d_ff": 32},
        "LitCNN": {"model.channels": [8, 16], "model.kernels": [5, 3]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", action="store_true", help="also execute the notebooks")
    args = parser.parse_args()

    print("[1/4] parse the fixture BacDive records")
    records = json.loads((ROOT / "tests" / "fixtures" / "bacdive_records.json").read_text())
    parsed = pd.DataFrame(build.parse_record(r) for r in records)
    assert parsed["bacdive_id"].tolist() == [100, 654, 9001]
    assert parsed["oxygen"].tolist()[:2] == ["anaerobe", "facultative"]

    print("[2/4] load the tracked sample and check the release invariants")
    sample = data.load(data.SAMPLE)
    build.validate(sample)
    subset = build.make_sample(sample, n_rows=400, seed=0)

    n_arms = len(protocol.NEURAL) + len(protocol.BASELINE)
    print(f"[3/4] the paper's protocol at toy size: {n_arms} arms, one seed")
    with tempfile.TemporaryDirectory() as tmp:
        path, version = str(Path(tmp) / "subset.parquet"), Path(tmp) / "smoke"
        subset.to_parquet(path, index=False)
        results = protocol.run(version, seeds=[0], path=Path(path), overrides=shrink())
        assert results["arm"].nunique() == n_arms
        scores = results[results["metric"].isin([f"{t}_acc" for t in TASKS])]
        assert scores["value"].map(math.isfinite).all()
        assert (version / "leaderboard.md").exists()

    print("[4/4] notebooks" if args.notebooks else "[4/4] notebooks skipped (--notebooks)")
    if args.notebooks:
        for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
            start = time.time()
            command = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace"]
            subprocess.run([sys.executable, "-m", *command, str(notebook)], check=True)
            print(f"      executed {notebook.name} in {time.time() - start:.0f} s")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
