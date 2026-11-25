"""Run the paper's protocol on the full DSNI table (data/dsni.parquet).

Nothing here has been run at this scale in this repository: the protocol is the table in
``dsni.protocol`` and ``scripts/smoke_test.py`` runs that same table at toy size. Expect GPU
hours per neural run. Runs whose result exists are skipped, so the campaign can be stopped
and resumed.

Usage:
    uv run python scripts/reproduce_paper.py --version_dir out/v1
    uv run python scripts/reproduce_paper.py --arms '[transformer_gated]' --seeds '[0]'
"""

from jsonargparse import auto_cli

from dsni.protocol import run

if __name__ == "__main__":
    auto_cli(run)
