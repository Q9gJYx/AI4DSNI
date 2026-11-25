"""Build the DSNI table from BacDive, MediaDive and NCBI.

The parsing and label rules come first and are pure, so they are unit-tested offline. The
network stages at the bottom cache what they fetch and skip work that is already on disk.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from dsni.schema import (
    CLASSES,
    COLUMNS,
    PH_BINS,
    RANKS,
    SPLIT_RATIOS,
    SPLITS,
    TASKS,
    TEMP_BINS,
)
from dsni.splits import greedy_stratified_split

logger = logging.getLogger(__name__)

BACDIVE_FETCH = "https://api.bacdive.dsmz.de/v2/fetch/"
MEDIADIVE_REST = "https://mediadive.dsmz.de/rest/"
USER_AGENT = "dsni-build/1.0"

# Release defaults; scripts/build_dsni.py exposes them as options.
MIN_LEN, MAX_LEN = 500, 1600  # accepted 16S sequence length
MAX_AMBIGUOUS = 0.01  # share of non-ACGT bases at which a sequence is dropped
N_MEDIA = 42  # media classes including "Other"

SHARD = 10_000  # BacDive ids per cache file
PAUSE = 0.2  # seconds between calls to the DSMZ services
NCBI_BATCH = 200  # accessions per efetch call
MAX_MISSING = 0.01  # share of accessions NCBI may fail to return before the build stops

# BacDive oxygen-tolerance vocabulary -> the four DSNI classes.
OXYGEN_MAP = {
    "aerobe": "aerobe",
    "obligate aerobe": "aerobe",
    "anaerobe": "anaerobe",
    "obligate anaerobe": "anaerobe",
    "facultative anaerobe": "facultative",
    "facultative aerobe": "facultative",
    "microaerophile": "microaerophile",
}

_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-\s*(-?\d+(?:\.\d+)?))?\s*$")
_MEDIADIVE_ID = re.compile(r"mediadive\.dsmz\.de/medium/(\w+)")
_NOT_ACGT = re.compile("[^ACGT]")


# --------------------------------------------------------------------------- parsing


def _as_list(value: Any) -> list[dict[str, Any]]:
    """BacDive stores a single entry as a dict and several as a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def parse_range(value: object) -> float | None:
    """Return the midpoint of ``"37"``, ``"30-55"``, ``"-2-10"``, ``"-5--2"`` or ``"25--30"``.

    A second number below the first can only be a doubled dash (``"25--30"`` means 25 to 30),
    so its sign is dropped.
    """
    match = _RANGE.match(str(value))
    if match is None:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    return (low + (high if high >= low else -high)) / 2


def representative(entries: list[dict[str, Any]], value_key: str, flag_key: str) -> float | None:
    """Median midpoint of the positive ``optimum`` entries, else of the ``growth`` entries."""
    for wanted in ("optimum", "growth"):
        values = [
            parse_range(e.get(value_key))
            for e in entries
            if e.get("type") == wanted and e.get(flag_key) == "positive"
        ]
        kept = [v for v in values if v is not None]
        if kept:
            return float(np.median(kept))
    return None


def oxygen_class(values: Iterable[str | None]) -> str | None:
    """Collapse the reported oxygen tolerances of one strain into a single class.

    Any facultative report gives ``facultative``. Reports of both aerobic and anaerobic
    growth with no facultative report contradict each other, and the strain is left without
    an oxygen label rather than given an inferred one. Otherwise the most frequent class
    wins, first reported on ties.
    """
    mapped = [OXYGEN_MAP[v] for v in values if v in OXYGEN_MAP]
    kinds = set(mapped)
    if "facultative" in kinds:
        return "facultative"
    if not mapped or {"aerobe", "anaerobe"} <= kinds:
        return None
    return Counter(mapped).most_common(1)[0][0]


def media_ids(entries: list[dict[str, Any]]) -> list[str]:
    """MediaDive ids of the media with recorded growth, in BacDive order, without repeats."""
    ids: list[str] = []
    for entry in entries:
        match = _MEDIADIVE_ID.search(entry.get("link") or "")
        if match and entry.get("growth") == "yes" and match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def pick_accession(entries: list[dict[str, Any]], min_len: int, max_len: int) -> str | None:
    """Longest 16S accession whose stated length is in range; the first wins ties."""
    in_range = [
        e for e in entries if e.get("accession") and min_len <= (e.get("length") or 0) <= max_len
    ]
    return max(in_range, key=lambda e: e["length"])["accession"] if in_range else None


def parse_record(
    record: dict[str, Any], min_len: int = MIN_LEN, max_len: int = MAX_LEN
) -> dict[str, Any]:
    """Flatten one BacDive v2 record into the fields DSNI uses."""
    general = record.get("General") or {}
    names = record.get("Name and taxonomic classification") or {}
    lpsn = names.get("LPSN") or {}
    lpsn = lpsn[0] if isinstance(lpsn, list) else lpsn
    culture = record.get("Culture and growth conditions") or {}
    physiology = record.get("Physiology and metabolism") or {}
    s16 = _as_list((record.get("Sequence information") or {}).get("16S sequences"))
    media = media_ids(_as_list(culture.get("culture medium")))
    oxygen = (e.get("oxygen tolerance") for e in _as_list(physiology.get("oxygen tolerance")))
    return {
        "bacdive_id": int(general["BacDive-ID"]),
        "dsm_number": general.get("DSM-Number"),
        "type_strain": names.get("type strain") == "yes",
        **{rank: lpsn.get(rank) or names.get(rank) for rank in RANKS},
        "temp_c": representative(_as_list(culture.get("culture temp")), "temperature", "growth"),
        "ph_value": representative(_as_list(culture.get("culture pH")), "pH", "ability"),
        "oxygen": oxygen_class(oxygen),
        "media_id": media[0] if media else None,
        "media_all": media,
        "accession": pick_accession(s16, min_len, max_len),
        "accessions_all": list(dict.fromkeys(e["accession"] for e in s16 if e.get("accession"))),
        "release": ".".join(str(general.get("doi", "")).split(".")[-2:]),
    }


def clean_sequence(sequence: str) -> str:
    """Upper-case DNA with U as T and every other symbol as N."""
    return _NOT_ACGT.sub("N", sequence.upper().replace("U", "T"))


# --------------------------------------------------------------------------- assembly


def _bin(values: pd.Series, bins: tuple[float, float], classes: tuple[str, ...]) -> pd.Categorical:
    """Three ordered classes: below ``bins[0]``, inside the closed interval, above ``bins[1]``."""
    codes = np.select([values.isna(), values < bins[0], values > bins[1]], [-1, 0, 2], default=1)
    return pd.Categorical.from_codes(codes, categories=list(classes))


def media_classes(
    media_id: pd.Series, n_classes: int = N_MEDIA
) -> tuple[pd.Categorical, pd.DataFrame]:
    """Keep the ``n_classes - 1`` most frequent media as classes and pool the rest as ``Other``.

    Returns:
        The class of every row (missing stays missing) and the media map, one row per medium
        id with its strain count and class.
    """
    counts = media_id.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    top = list(counts.index[: n_classes - 1])
    labels = media_id.where(media_id.isin(top), "Other").where(media_id.notna())
    media_map = counts.rename("n_strains").rename_axis("media_id").reset_index()
    media_map["media"] = media_map["media_id"].where(media_map["media_id"].isin(top), "Other")
    return pd.Categorical(labels, categories=[*top, "Other"]), media_map


def assemble(
    strains: pd.DataFrame,
    sequences: Mapping[str, str],
    medium_ph: Mapping[str, float],
    *,
    versions: Mapping[str, str] | None = None,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
    max_ambiguous: float = MAX_AMBIGUOUS,
    n_media: int = N_MEDIA,
    ph_fallback: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Join labels with sequences, filter, deduplicate, derive classes and split.

    Args:
        strains: Output of :func:`parse_record`, one row per BacDive record.
        sequences: Accession (without version) to raw sequence.
        versions: Accession (without version) to the versioned accession that was actually
            retrieved, for example ``DQ297928`` to ``DQ297928.1``; recorded in ``accession``.
        medium_ph: MediaDive medium id to the pH of the medium.
        min_len: Shortest sequence kept.
        max_len: Longest sequence kept.
        max_ambiguous: Sequences with this share of non-ACGT bases or more are dropped.
        n_media: Number of media classes including ``Other``.
        ph_fallback: Use the pH of the primary medium when the strain has no pH of its own.

    Returns:
        The DSNI table, the media map, and the number of rows left after every step.
    """
    steps = {"bacdive_records": len(strains)}
    df = strains[strains["domain"] == "Bacteria"].copy()
    steps["bacteria"] = len(df)
    steps["with_16s_accession"] = int(df["accessions_all"].str.len().gt(0).sum())
    df = df[df["accession"].notna() & df["genus"].notna()]
    steps["accession_in_length_range"] = len(df)

    base = df["accession"].str.split(".").str[0]
    df["sequence"] = base.map(sequences)
    if versions is not None:
        df["accession"] = base.map(versions).fillna(df["accession"])
    df = df[df["sequence"].notna()]
    df["sequence"] = df["sequence"].map(clean_sequence)
    steps["sequence_retrieved"] = len(df)
    df["seq_len"] = df["sequence"].str.len()
    ambiguous = df["sequence"].str.count("N") / df["seq_len"]
    df = df[df["seq_len"].between(min_len, max_len) & (ambiguous < max_ambiguous)]
    steps["passes_sequence_qc"] = len(df)

    df["medium_ph"] = df["media_id"].map(medium_ph)
    ph = df["ph_value"].fillna(df["medium_ph"]) if ph_fallback else df["ph_value"]
    df["ph_source"] = np.where(df["ph_value"].notna(), "strain", "medium")
    df["ph_source"] = df["ph_source"].where(ph.notna())
    df["temperature"] = _bin(df["temp_c"], TEMP_BINS, CLASSES["temperature"])
    df["ph"] = _bin(ph, PH_BINS, CLASSES["ph"])
    df["oxygen"] = pd.Categorical(df["oxygen"], categories=list(CLASSES["oxygen"]))

    n_labels = df[["media_id", "temperature", "ph", "oxygen"]].notna().sum(axis=1)
    df = df[n_labels > 0]
    steps["with_any_label"] = len(df)

    # Exact-duplicate sequences collapse to the best-annotated record (DSMZ records on ties).
    rank = pd.DataFrame(
        {"n": n_labels.loc[df.index], "dsm": df["dsm_number"].notna(), "id": df["bacdive_id"]}
    ).sort_values(["n", "dsm", "id"], ascending=[False, False, True])
    df = df.loc[rank.index].drop_duplicates("sequence").sort_values("bacdive_id")
    steps["after_exact_dedup"] = len(df)

    df["media"], media_map = media_classes(df["media_id"], n_media)
    df["split"] = greedy_stratified_split(df)
    df["strain_id"] = "BacDive:" + df["bacdive_id"].astype(str)
    df["dsm_number"] = pd.to_numeric(df["dsm_number"], errors="coerce").astype("Int64")
    return df[list(COLUMNS)].reset_index(drop=True), media_map, steps


def validate(table: pd.DataFrame, tolerance: float = 0.05) -> None:
    """Raise if ``table`` breaks an invariant of a DSNI release.

    Checks the column set, the uniqueness of ``strain_id``, that every strain has a label,
    that no genus crosses splits and that the split shares are within ``tolerance`` of the
    targets.
    """
    shares = table["split"].value_counts(normalize=True)
    off = {s: round(shares.get(s, 0.0), 3) for s, r in zip(SPLITS, SPLIT_RATIOS, strict=True)}
    problems = {
        "columns differ from schema.COLUMNS": list(table.columns) != list(COLUMNS),
        "strain_id is not unique": not table["strain_id"].is_unique,
        "a strain has no label": not table[list(TASKS)].notna().any(axis=1).all(),
        "a genus crosses splits": table.groupby("genus")["split"].nunique().max() > 1,
        f"split shares off target: {off}": any(
            abs(off[s] - r) > tolerance for s, r in zip(SPLITS, SPLIT_RATIOS, strict=True)
        ),
    }
    failed = [name for name, bad in problems.items() if bad]
    if failed:
        raise ValueError("; ".join(failed))


def make_sample(table: pd.DataFrame, n_rows: int = 1800, seed: int = 0) -> pd.DataFrame:
    """Draw whole genera at random inside each split until about ``n_rows`` rows are kept.

    Genera larger than a quarter of a split's quota are left out so that no single genus
    dominates the sample; the split ratios of the full table are preserved.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, part in table.groupby("split", sort=True):
        quota = n_rows * len(part) / len(table)
        sizes = part.groupby("genus").size().sort_index()
        small = sizes[sizes <= quota / 4]
        sizes = small if len(small) else sizes
        sizes = sizes.loc[rng.permutation(sizes.index.to_numpy())]
        keep = sizes.index[: int(np.searchsorted(sizes.cumsum().to_numpy(), quota)) + 1]
        parts.append(part[part["genus"].isin(keep)])
    return pd.concat(parts).sort_values("bacdive_id").reset_index(drop=True)


def write_fasta(table: pd.DataFrame, path: Path) -> None:
    """Write ``>strain_id accession species`` records to a gzip-compressed FASTA file."""
    with gzip.open(path, "wt", encoding="ascii") as fh:
        for row in table.itertuples(index=False):
            fh.write(f">{row.strain_id} {row.accession} {row.species}\n{row.sequence}\n")


def sha256(path: Path) -> str:
    """Hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- network


def _session() -> Any:
    """A requests session that retries transient failures with backoff."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(total=6, backoff_factor=2.0, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _fetch_shard(start: int, stop: int, path: Path) -> None:
    """Write the BacDive records with ids in ``[start, stop)`` to one gzip JSONL file."""
    session = _session()
    partial = path.with_suffix(".part")
    n = 0
    with gzip.open(partial, "wt", encoding="utf-8") as fh:
        for low in range(start, stop, 100):  # the API accepts at most 100 ids per call
            ids = ";".join(str(i) for i in range(low, min(low + 100, stop)))
            response = session.get(BACDIVE_FETCH + ids, timeout=180)
            response.raise_for_status()
            results = response.json().get("results") or {}
            for record in results.values() if isinstance(results, dict) else results:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
            time.sleep(PAUSE)
    partial.replace(path)
    logger.info("bacdive ids %d to %d: %d records", start, stop - 1, n)


def fetch_bacdive(raw_dir: Path, max_id: int = 200_000, workers: int = 2) -> None:
    """Download every BacDive record with id up to ``max_id``; finished shards are skipped."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for start in range(1, max_id + 1, SHARD):
        stop = min(start + SHARD, max_id + 1)
        path = raw_dir / f"bacdive_{start:06d}_{stop - 1:06d}.jsonl.gz"
        if not path.exists():
            jobs.append((start, stop, path))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(_fetch_shard, *job) for job in jobs]:
            future.result()


def read_bacdive(raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield every cached BacDive record."""
    for path in sorted(raw_dir.glob("bacdive_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def fetch_mediadive(raw_dir: Path) -> pd.DataFrame:
    """The MediaDive media list: id, name, source and the pH of each medium."""
    cache = raw_dir / "mediadive_media.json"
    if not cache.exists():
        response = _session().get(MEDIADIVE_REST + "media", timeout=180)
        response.raise_for_status()
        cache.write_text(json.dumps(response.json()["data"], ensure_ascii=False))
    media = pd.DataFrame(json.loads(cache.read_text()))
    media["media_id"] = media["id"].astype(str)
    media["medium_ph"] = media[["min_pH", "max_pH"]].mean(axis=1)
    return media[["media_id", "name", "source", "medium_ph", "link"]]


def fetch_recipes(ids: Iterable[str], raw_dir: Path) -> dict[str, Any]:
    """MediaDive recipes (solutions, ingredients, steps) of the given media."""
    cache = raw_dir / "mediadive_recipes.json"
    recipes: dict[str, Any] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [i for i in ids if i not in recipes]
    if todo:
        session = _session()
        for media_id in todo:
            response = session.get(f"{MEDIADIVE_REST}medium/{media_id}", timeout=180)
            response.raise_for_status()
            recipes[media_id] = response.json().get("data")
            time.sleep(PAUSE)
        cache.write_text(json.dumps(recipes, ensure_ascii=False))
    return {i: recipes[i] for i in ids}


def fetch_sequences(
    accessions: Iterable[str], raw_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch nuccore sequences by accession through NCBI E-utilities into a resumable cache.

    ``NCBI_EMAIL`` and ``NCBI_API_KEY`` are read from the environment when set; NCBI asks
    heavy users to identify themselves, and a key raises the rate limit.

    Returns:
        Two maps keyed by the accession without its version: the sequence, and the versioned
        accession NCBI returned (the newest version unless one was requested).
    """
    from Bio import Entrez, SeqIO

    Entrez.tool = "dsni-build"
    Entrez.email = os.environ.get("NCBI_EMAIL")  # type: ignore[assignment]
    Entrez.api_key = os.environ.get("NCBI_API_KEY")  # type: ignore[assignment]
    cache = raw_dir / "ncbi_16s.fasta"
    have: dict[str, str] = {}
    version: dict[str, str] = {}
    if cache.exists():
        for rec in SeqIO.parse(cache, "fasta"):
            have[rec.id.split(".")[0]], version[rec.id.split(".")[0]] = str(rec.seq), rec.id
    wanted = {a.split(".")[0] for a in accessions}
    todo = sorted(wanted - have.keys())

    def fetch(chunk: list[str], fh: Any) -> None:
        try:
            handle = Entrez.efetch(db="nuccore", id=chunk, rettype="fasta", retmode="text")
            records = list(SeqIO.parse(handle, "fasta"))
        except (HTTPError, ValueError) as error:
            # A call that contains an accession NCBI does not know fails as a whole, with
            # HTTP 400 or with an error text that is not FASTA: bisect to isolate it. Any
            # other HTTP status is transient and must stop the build, because a skipped batch
            # would silently drop strains; the cache makes the rerun cheap.
            if isinstance(error, HTTPError) and error.code != 400:
                raise
            if len(chunk) == 1:
                logger.warning("accession %s not retrieved: %s", chunk[0], error)
                return
            fetch(chunk[: len(chunk) // 2], fh)
            fetch(chunk[len(chunk) // 2 :], fh)
            return
        for rec in records:
            have[rec.id.split(".")[0]], version[rec.id.split(".")[0]] = str(rec.seq), rec.id
        SeqIO.write(records, fh, "fasta")

    with cache.open("a") as fh:
        for start in range(0, len(todo), NCBI_BATCH):
            fetch(todo[start : start + NCBI_BATCH], fh)
            done = min(start + NCBI_BATCH, len(todo))
            logger.info("ncbi: %d of %d accessions", done, len(todo))
    missing = wanted - have.keys()
    if len(missing) > MAX_MISSING * max(len(wanted), 1):
        raise RuntimeError(f"{len(missing)} of {len(wanted)} accessions were not retrieved")
    return have, version
