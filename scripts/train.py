"""Train one neural arm with LightningCLI and write ``<run_dir>/result.pt``.

Usage:
    uv run python scripts/train.py --model LitTransformer --model.gate bio \
        --run_dir out/v1/transformer_gated/s0
"""

from dsni.lit import cli_main

if __name__ == "__main__":
    cli_main()
