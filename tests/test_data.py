import hashlib

import numpy as np
import pandas as pd
import pytest

from dsni import data
from dsni.schema import COLUMNS, IGNORE_INDEX, SPLITS, TASKS


def test_encode_vocabulary_padding_and_cropping():
    tokens = data.encode(["ACGTUN", "acgtX", "AAAACCCCGGGG"], max_len=8)
    assert tokens.tolist()[0] == [1, 2, 3, 4, 4, 5, 0, 0]
    assert tokens.tolist()[1] == [1, 2, 3, 4, 5, 0, 0, 0]
    assert tokens.tolist()[2] == [1, 1, 1, 1, 2, 2, 2, 2]
    centred = data.encode(["AAAACCCCGGGG"], max_len=4, crop="center")
    assert centred.tolist()[0] == [2, 2, 2, 2]


def test_encode_labels_marks_missing(toy_table):
    labels = data.encode_labels(toy_table).numpy()
    assert labels.shape == (len(toy_table), len(TASKS))
    for i, task in enumerate(TASKS):
        assert ((labels[:, i] == IGNORE_INDEX) == toy_table[task].isna().to_numpy()).all()
    assert labels[labels != IGNORE_INDEX].min() >= 0


def test_tracked_sample_matches_the_schema():
    assert data.SAMPLE.stat().st_size <= 1_000_000
    sample = pd.read_parquet(data.SAMPLE)
    assert list(sample.columns) == list(COLUMNS)
    assert sample.groupby("genus")["split"].nunique().max() == 1
    assert set(sample["split"]) == set(SPLITS)
    assert all(isinstance(sample[task].dtype, pd.CategoricalDtype) for task in TASKS)
    assert sample["sequence"].str.fullmatch("[ACGTN]+").all()


def test_load_reports_its_source_and_fingerprint(toy_path, toy_table):
    table = data.load(toy_path)
    assert table.attrs["source"] == "toy.parquet"
    assert table.attrs["fingerprint"] == data.fingerprint(toy_table)
    relabelled = toy_table.assign(split=toy_table["split"].replace({"val": "test"}))
    assert data.fingerprint(relabelled) != table.attrs["fingerprint"]
    assert data.load_splits(toy_path)["test"].attrs["fingerprint"] == table.attrs["fingerprint"]


def test_load_splits_options(toy_path, toy_table):
    parts = data.load_splits(toy_path)
    assert sum(len(p) for p in parts.values()) == len(toy_table)
    few = data.load_splits(toy_path, train_fraction=0.5, seed=1)
    assert len(few["train"]) == round(0.5 * len(parts["train"]))
    assert len(few["test"]) == len(parts["test"])
    leaky = data.load_splits(toy_path, split="random")
    assert set(leaky["train"]["genus"]) & set(leaky["test"]["genus"])  # genera are shared
    with pytest.raises(ValueError, match="split must be"):
        data.load_splits(toy_path, split="family")


def test_datamodule_batches_and_crop_offsets(toy_path, toy_table):
    module = data.DSNIDataModule(toy_path, max_len=128, batch_size=16)
    module.setup()
    tokens, labels, offsets = next(iter(module.train_dataloader()))
    assert tokens.shape == (16, 128) and labels.shape == (16, 4) and offsets.sum() == 0
    assert module.n_classes == {"media": 4, "temperature": 3, "ph": 3, "oxygen": 4}

    centred = data.DSNIDataModule(toy_path, max_len=128, crop="center")
    centred.setup()
    expected = (centred.tables["val"]["seq_len"].to_numpy() - 128) // 2
    assert centred.sets["val"].tensors[2].tolist() == expected.tolist()
    with pytest.raises(ValueError, match="crop must be"):
        data.encode(["ACGT"], 2, crop="end")


def test_fetch_verifies_the_checksum(tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(np.arange(64, dtype=np.uint8).tobytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dest = data.fetch(source.as_uri(), digest, tmp_path / "copy.parquet")
    assert dest.read_bytes() == source.read_bytes()
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.fetch(source.as_uri(), "0" * 64, tmp_path / "bad.parquet")
    assert not (tmp_path / "bad.parquet").exists()
    with pytest.raises(RuntimeError, match="no archive URL"):
        data.fetch(None)


def test_shipped_tables_match_the_manifest():
    import json

    manifest = json.loads((data.DATA_DIR / "manifest.json").read_text())
    for name in ("dsni.parquet", "dsni_sample.parquet", "media_map.csv"):
        digest = hashlib.sha256((data.DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == manifest["files"][name]["sha256"], name
    assert len(pd.read_parquet(data.FULL)) == manifest["rows"]
