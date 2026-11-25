"""Rebuild the DSNI dataset from BacDive, MediaDive and NCBI.

Inputs:  public APIs, no registration. Everything fetched is cached under ``--raw`` and reused.
Outputs: in ``--out``: dsni.parquet, dsni.fasta.gz, media_map.csv, media_compositions.json,
         manifest.json and dsni_sample.parquet.

Usage:
    uv run --extra build python scripts/build_dsni.py                 # full rebuild
    uv run --extra build python scripts/build_dsni.py --max_id 2000 --out out/dry --raw out/dry/raw
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from jsonargparse import auto_cli

from dsni import build
from dsni.schema import TASKS

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = ("dsni.parquet", "dsni.fasta.gz", "media_map.csv", "media_compositions.json")
SAMPLE_FILE = "dsni_sample.parquet"


def main(
    out: Path = ROOT / "data",
    raw: Path = ROOT / "data" / "raw",
    max_id: int = 200_000,
    workers: int = 2,
    min_len: int = build.MIN_LEN,
    max_len: int = build.MAX_LEN,
    max_ambiguous: float = build.MAX_AMBIGUOUS,
    n_media: int = build.N_MEDIA,
    ph_fallback: bool = True,
    sample_rows: int = 5000,
    sample_bytes: int = 1_000_000,
    seed: int = 0,
) -> None:
    """Fetch (or reuse the cache), assemble, validate and write a DSNI release.

    Args:
        out: Output directory.
        raw: Download cache.
        max_id: Highest BacDive id to request.
        workers: Parallel BacDive shard downloads.
        min_len: Shortest 16S sequence kept.
        max_len: Longest 16S sequence kept.
        max_ambiguous: Share of non-ACGT bases at which a sequence is dropped.
        n_media: Media classes including ``Other``.
        ph_fallback: Use the pH of the primary medium when a strain has no pH of its own.
        sample_rows: Rows aimed for in the tracked sample.
        sample_bytes: Size cap of the tracked sample; it is shrunk until it fits.
        seed: Seed of the sample draw.
    """
    parameters = {k: str(v) for k, v in locals().items() if k not in ("out", "raw")}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    build.fetch_bacdive(raw / "bacdive", max_id, workers)
    records = build.read_bacdive(raw / "bacdive")
    strains = pd.DataFrame(build.parse_record(r, min_len, max_len) for r in records)
    strains = strains[strains["bacdive_id"] <= max_id].drop_duplicates("bacdive_id")
    media = build.fetch_mediadive(raw)
    wanted = strains.loc[strains["domain"] == "Bacteria", "accession"].dropna()
    sequences, versions = build.fetch_sequences(wanted, raw)

    table, media_map, steps = build.assemble(
        strains,
        sequences,
        media.set_index("media_id")["medium_ph"].dropna().to_dict(),
        versions=versions,
        min_len=min_len,
        max_len=max_len,
        max_ambiguous=max_ambiguous,
        n_media=n_media,
        ph_fallback=ph_fallback,
    )
    build.validate(table)

    media_map = media_map.merge(media, on="media_id", how="left")
    classes = [m for m in table["media"].cat.categories if m != "Other"]
    recipes = build.fetch_recipes(classes, raw)
    manifest = {
        "built": dt.date.today().isoformat(),
        "sources": {
            "bacdive": {
                "url": build.BACDIVE_FETCH,
                "releases": dict(Counter(strains["release"]).most_common(5)),
            },
            "mediadive": {"url": build.MEDIADIVE_REST, "media": len(media)},
            "ncbi": {"db": "nuccore", "accessions_cached": len(sequences)},
        },
        "parameters": parameters,
        "attrition": steps,
        "rows": len(table),
        "genera": int(table["genus"].nunique()),
        "species": int(table["species"].nunique()),
        "splits": {
            s: {"rows": len(g), "genera": int(g["genus"].nunique())}
            for s, g in table.groupby("split")
        },
        "labels": {
            t: {str(k): int(v) for k, v in table[t].value_counts(dropna=False).items()}
            for t in TASKS
        },
    }
    # The table describes itself: provenance and media names travel inside the parquet file.
    names = media_map.dropna(subset=["name"]).set_index("media_id")["name"].to_dict()
    table.attrs = {"manifest": manifest, "media_names": names}

    table.to_parquet(out / "dsni.parquet", compression="zstd", index=False)
    build.write_fasta(table, out / "dsni.fasta.gz")
    media_map.to_csv(out / "media_map.csv", index=False)
    (out / "media_compositions.json").write_text(json.dumps(recipes, ensure_ascii=False))

    while True:  # shrink the sample in memory until it fits the size cap, then write it once
        sample = build.make_sample(table, sample_rows, seed)
        sample.attrs = table.attrs
        buffer = io.BytesIO()
        sample.to_parquet(buffer, compression="zstd", compression_level=19, index=False)
        if buffer.getbuffer().nbytes <= sample_bytes:
            break
        sample_rows = int(sample_rows * 0.9)
    (out / SAMPLE_FILE).write_bytes(buffer.getvalue())

    manifest["sample_rows"] = len(sample)
    manifest["files"] = {
        f: {"bytes": (out / f).stat().st_size, "sha256": build.sha256(out / f)}
        for f in (*RELEASE_FILES, SAMPLE_FILE)
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("attrition: %s", steps)
    logging.info("wrote %d strains, %d genera to %s", len(table), manifest["genera"], out)


if __name__ == "__main__":
    auto_cli(main)
