"""Fit the ``k-mer + RF`` arm and write ``<run_dir>/result.pt``.

Usage:
    uv run python scripts/run_baseline.py --run_dir out/v1/kmer_rf/s0
"""

from jsonargparse import auto_cli

from dsni.baselines import run

if __name__ == "__main__":
    auto_cli(run)
