"""Loading, tokenising and batching the DSNI table."""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dsni.schema import IGNORE_INDEX, PAD, SPLITS, TASKS, UNKNOWN, VOCAB
from dsni.splits import random_split

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # repo root for an editable install
FULL = DATA_DIR / "dsni.parquet"
SAMPLE = DATA_DIR / "dsni_sample.parquet"

# The full table ships in data/. For installs without the repository, fetch() downloads the
# copy tagged v1.0.0 (also inside the Zenodo archive of that release, doi:10.5281/zenodo.24524178).
DSNI_URL: str | None = "https://github.com/Q9gJYx/AI4DSNI/raw/v1.0.0/data/dsni.parquet"
DSNI_SHA256: str | None = "705e2abc4b305763647e309476ae185cf87d128fa3deeaaee1d027f3728a6450"

_LUT = np.full(256, UNKNOWN, dtype=np.int64)
for _base, _index in VOCAB.items():
    _LUT[ord(_base)] = _LUT[ord(_base.lower())] = _index


def fetch(url: str | None = DSNI_URL, sha256: str | None = DSNI_SHA256, dest: Path = FULL) -> Path:
    """Download the full table to ``dest`` and verify its checksum."""
    if url is None:
        raise RuntimeError(
            "No download URL given. Build the table with `python scripts/build_dsni.py`, or "
            "use the copy shipped in data/ (`dsni.data.load()`)."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if sha256 is not None and digest != sha256:
        partial.unlink()
        raise RuntimeError(f"checksum mismatch for {url}: got {digest}, expected {sha256}")
    partial.replace(dest)
    return dest


def load(path: str | Path | None = None) -> pd.DataFrame:
    """Read a DSNI table.

    Args:
        path: A parquet file. By default the full table when it is on disk, otherwise the
            small sample tracked in the repository.

    Returns:
        One row per strain. ``table.attrs`` holds ``source`` (the file that was read),
        ``fingerprint`` (see :func:`fingerprint`), ``manifest`` (how the release was built)
        and ``media_names`` (MediaDive id to name).
    """
    if path is None:
        path = FULL if FULL.exists() else SAMPLE
    table = pd.read_parquet(path)
    table.attrs["source"] = Path(path).name
    table.attrs["fingerprint"] = fingerprint(table)
    return table


def fingerprint(table: pd.DataFrame) -> str:
    """Short hash of what a run depends on: ids, sequences, labels and the released split."""
    content = table[["strain_id", "sequence", *TASKS, "split"]]
    hashes = pd.util.hash_pandas_object(content, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()[:16]


def crop_start(length: Any, max_len: int, crop: str) -> Any:
    """Index of the first base kept when sequences of ``length`` are cut to ``max_len``.

    ``crop="start"`` keeps the first bases, ``crop="center"`` the middle ones. Works on a
    single length or an array of lengths.
    """
    if crop not in ("start", "center"):
        raise ValueError(f"crop must be 'start' or 'center', got {crop!r}")
    return np.maximum(length - max_len, 0) // 2 if crop == "center" else np.zeros_like(length)


def encode(sequences: Iterable[str], max_len: int, crop: str = "start") -> torch.Tensor:
    """Token ids of shape ``(n, max_len)``; long sequences are cropped, short ones padded.

    Args:
        sequences: Nucleotide strings; anything outside A, C, G, T/U becomes N.
        max_len: Output length.
        crop: Which part of a longer sequence to keep, see :func:`crop_start`.
    """
    sequences = list(sequences)
    out = np.full((len(sequences), max_len), PAD, dtype=np.int64)
    for i, sequence in enumerate(sequences):
        ids = _LUT[np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)]
        start = int(crop_start(len(ids), max_len, crop))
        ids = ids[start : start + max_len]
        out[i, : len(ids)] = ids
    return torch.from_numpy(out)


def encode_labels(table: pd.DataFrame) -> torch.Tensor:
    """Class codes of shape ``(n, 4)`` in ``TASKS`` order; missing labels are ``IGNORE_INDEX``."""
    codes = np.stack([table[task].cat.codes.to_numpy() for task in TASKS], axis=1).astype(np.int64)
    codes[codes < 0] = IGNORE_INDEX
    return torch.from_numpy(codes)


def num_classes(table: pd.DataFrame) -> dict[str, int]:
    """Number of classes of every task, read from the categorical dtypes."""
    return {task: len(table[task].cat.categories) for task in TASKS}


def load_splits(
    path: str | Path | None = None,
    split: str = "genus",
    train_fraction: float = 1.0,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """The train, val and test rows of a run; every arm selects its rows through here.

    Args:
        path: Parquet file; ``None`` resolves as in :func:`load`.
        split: ``"genus"`` uses the released genus-disjoint split; ``"random"`` redraws a
            row-level split, which leaks taxonomy and exists to measure that leak.
        train_fraction: Share of the training rows kept (few-shot experiments).
        seed: Seed of the random split and of the training subsample.
    """
    table = load(path)
    if split == "random":
        table = table.assign(split=random_split(table, seed=seed))
    elif split != "genus":
        raise ValueError(f"split must be 'genus' or 'random', got {split!r}")
    parts = {name: table[table["split"] == name] for name in SPLITS}
    if train_fraction < 1.0:
        parts["train"] = parts["train"].sample(frac=train_fraction, random_state=seed)
    return parts


class DSNIDataModule(L.LightningDataModule):
    """The DSNI table as train, val and test loaders of ``(tokens, labels, offsets)`` batches.

    ``offsets`` is the position on the gene of the first token of every row: zero unless the
    sequence was centre-cropped. The variability gate needs it to place its region tags.
    """

    def __init__(
        self,
        path: str | None = None,
        max_len: int = 1500,
        crop: str = "start",
        split: str = "genus",
        train_fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        """Configure the data.

        Args:
            path: Parquet file; ``None`` resolves as in :func:`load`.
            max_len: Sequences are cropped or padded to this many tokens.
            crop: ``"start"`` or ``"center"`` (the latter for the truncation experiment).
            split: ``"genus"`` or ``"random"``, see :func:`load_splits`.
            train_fraction: Share of the training rows kept, see :func:`load_splits`.
            batch_size: Batch size of every loader.
            num_workers: Loader workers; the tensors are in memory, so 0 is usually best.
            seed: Seed of the random split and of the training subsample.
        """
        super().__init__()
        self.save_hyperparameters()
        self.tables: dict[str, pd.DataFrame] = {}
        self.sets: dict[str, TensorDataset] = {}
        self.n_classes: dict[str, int] = {}

    def setup(self, stage: str | None = None) -> None:
        """Read the table once and tokenise every split."""
        if self.sets:
            return
        hp = self.hparams
        self.tables = load_splits(hp["path"], hp["split"], hp["train_fraction"], hp["seed"])
        self.n_classes = num_classes(self.tables["train"])
        for name, part in self.tables.items():
            tokens = encode(part["sequence"], hp["max_len"], hp["crop"])
            offsets = crop_start(part["seq_len"].to_numpy(), hp["max_len"], hp["crop"])
            self.sets[name] = TensorDataset(tokens, encode_labels(part), torch.as_tensor(offsets))

    def loader(self, name: str, shuffle: bool = False) -> DataLoader:
        """Batches of one split, in table order unless ``shuffle``."""
        hp = self.hparams
        return DataLoader(
            self.sets[name],
            batch_size=hp["batch_size"],
            shuffle=shuffle,
            num_workers=hp["num_workers"],
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training batches."""
        return self.loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation batches in table order."""
        return self.loader("val")

    def test_dataloader(self) -> DataLoader:
        """Test batches in table order."""
        return self.loader("test")
