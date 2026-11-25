import numpy as np
import pandas as pd
import pytest

from dsni import build
from dsni.schema import COLUMNS
from tests.conftest import synthetic_strains


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("37", 37.0),
        ("30-55", 42.5),
        ("-2-10", 4.0),
        ("-5--2", -3.5),
        ("-5 - -2", -3.5),
        ("25--30", 27.5),
        ("6.5-7.5", 7.0),
        ("abc", None),
        (None, None),
    ],
)
def test_parse_range(text, expected):
    assert build.parse_range(text) == expected


def test_representative_prefers_optimum_and_ignores_negative_growth():
    entries = [
        {"type": "growth", "growth": "positive", "temperature": "20-40"},
        {"type": "growth", "growth": "negative", "temperature": "4"},
        {"type": "optimum", "growth": "positive", "temperature": "37"},
    ]
    assert build.representative(entries, "temperature", "growth") == 37.0
    assert build.representative(entries[:2], "temperature", "growth") == 30.0
    assert build.representative(entries[1:2], "temperature", "growth") is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["aerobe", "anaerobe"], None),
        (["obligate aerobe", "obligate anaerobe", "aerobe"], None),
        (["aerobe", "anaerobe", "facultative aerobe"], "facultative"),
        (["aerobe", "aerobe", "facultative anaerobe"], "facultative"),
        (["anaerobe", "obligate anaerobe"], "anaerobe"),
        (["microaerophile", "aerobe", "obligate aerobe"], "aerobe"),
        (["something new", None], None),
        ([], None),
    ],
)
def test_oxygen_class(values, expected):
    assert build.oxygen_class(values) == expected


def test_media_ids_keeps_order_growth_and_mediadive_links_only():
    entries = [
        {"name": "Columbia agar", "growth": "yes"},
        {"link": "https://mediadive.dsmz.de/medium/514", "growth": "yes"},
        {"link": "https://mediadive.dsmz.de/medium/J22", "growth": "no"},
        {"link": "https://mediadive.dsmz.de/medium/1a", "growth": "yes"},
        {"link": "https://mediadive.dsmz.de/medium/514", "growth": "yes"},
        {"link": "https://catalogue-crbip.pasteur.fr/fiche_milieu.xhtml?crbip=72", "growth": "yes"},
    ]
    assert build.media_ids(entries) == ["514", "1a"]


def test_pick_accession_takes_the_longest_in_range():
    entries = [
        {"accession": "A", "length": 1400},
        {"accession": "B", "length": 1550},
        {"accession": "C", "length": 2900},
        {"accession": "D", "length": 300},
    ]
    assert build.pick_accession(entries, 500, 1600) == "B"
    assert build.pick_accession(entries, 500, 1500) == "A"
    assert build.pick_accession(entries[2:], 500, 1600) is None


def test_parse_record_on_real_records(records):
    by_id = {r["General"]["BacDive-ID"]: build.parse_record(r) for r in records}
    rich = by_id[654]  # several media, conflicting oxygen reports, seven 16S accessions
    assert (rich["genus"], rich["domain"], rich["dsm_number"]) == ("Bacillus", "Bacteria", 1)
    assert rich["media_id"] == "1" and rich["media_all"] == ["1", "453"]
    assert rich["oxygen"] == "facultative"
    assert rich["temp_c"] == pytest.approx(37.75) and rich["ph_value"] == 6.0
    assert rich["accession"] == "DQ297928" and len(rich["accessions_all"]) == 7
    assert rich["release"] == "20260601.11"
    assert by_id[100]["oxygen"] == "anaerobe" and by_id[100]["ph_value"] is None
    assert by_id[9001]["accession"] is None and by_id[9001]["accessions_all"] == []


def test_clean_sequence():
    assert build.clean_sequence("acgu-RYn") == "ACGTNNNN"


def test_bin_boundaries_belong_to_the_middle_class():
    values = pd.Series([19.9, 20.0, 45.0, 45.1, np.nan])
    binned = build._bin(values, (20.0, 45.0), ("low", "mid", "high"))
    assert list(binned[:4]) == ["low", "mid", "mid", "high"] and pd.isna(binned[4])


def test_media_classes_pools_the_tail():
    media_id = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d", None])
    labels, media_map = build.media_classes(media_id, n_classes=3)
    assert list(labels.categories) == ["a", "b", "Other"]
    assert labels.value_counts().to_dict() == {"a": 5, "b": 3, "Other": 3}
    assert pd.isna(labels[-1])
    assert media_map.set_index("media_id")["media"].to_dict() == {
        "a": "a",
        "b": "b",
        "c": "Other",
        "d": "Other",
    }


def test_assemble_schema_filters_and_invariants(toy_table):
    assert list(toy_table.columns) == list(COLUMNS)
    build.validate(toy_table, tolerance=0.10)
    assert set(toy_table["split"]) == {"train", "val", "test"}
    assert len(toy_table["media"].cat.categories) == 4
    strain = toy_table["ph_source"] == "strain"
    assert toy_table.loc[strain, "ph_value"].notna().all()
    medium = toy_table["ph_source"] == "medium"
    assert medium.any() and toy_table.loc[medium, "ph_value"].isna().all()


def test_assemble_attrition_dedup_and_ph_fallback_switch():
    strains, sequences, medium_ph = synthetic_strains(n_genera=12, seed=1)
    strains.loc[0, "domain"] = "Archaea"
    sequences[strains.loc[1, "accession"]] = "ACGT" * 50  # too short
    sequences[strains.loc[2, "accession"]] = "N" * 60 + "ACGT" * 150  # too ambiguous
    sequences[strains.loc[4, "accession"]] = sequences[strains.loc[3, "accession"]]  # duplicate
    del sequences[strains.loc[5, "accession"]]  # not retrieved
    table, _, steps = build.assemble(strains, sequences, medium_ph, n_media=4)
    assert steps["bacteria"] == steps["bacdive_records"] - 1
    assert steps["sequence_retrieved"] == steps["accession_in_length_range"] - 1
    assert steps["passes_sequence_qc"] == steps["sequence_retrieved"] - 2
    assert steps["after_exact_dedup"] <= steps["with_any_label"]
    assert table["sequence"].is_unique and len(table) == steps["after_exact_dedup"]
    strict, _, _ = build.assemble(strains, sequences, medium_ph, n_media=4, ph_fallback=False)
    assert strict["ph"].notna().sum() < table["ph"].notna().sum()
    assert set(strict["ph_source"].dropna()) == {"strain"}


def test_make_sample_keeps_whole_genera_and_all_splits(toy_table):
    sample = build.make_sample(toy_table, n_rows=80, seed=0)
    assert set(sample["split"]) == {"train", "val", "test"}
    kept = sample["genus"].value_counts()
    assert (kept == toy_table["genus"].value_counts().loc[kept.index]).all()
    assert len(sample) < len(toy_table)


def test_validate_names_every_broken_invariant(toy_table):
    broken = toy_table.copy()
    broken.loc[broken.index[0], "split"] = "test" if broken["split"].iloc[0] != "test" else "val"
    broken.loc[broken.index[1], "strain_id"] = broken["strain_id"].iloc[2]
    with pytest.raises(ValueError, match="strain_id is not unique; a genus crosses splits"):
        build.validate(broken, tolerance=0.10)
    with pytest.raises(ValueError, match="split shares off target"):
        build.validate(toy_table[toy_table["split"] != "val"], tolerance=0.10)


def test_assemble_records_the_retrieved_accession_version():
    strains, sequences, medium_ph = synthetic_strains(n_genera=12, seed=2)
    first, second = strains.loc[0, "accession"], strains.loc[1, "accession"]
    versions = {first: f"{first}.3"}
    table, _, _ = build.assemble(strains, sequences, medium_ph, n_media=4, versions=versions)
    by_id = table.set_index("bacdive_id")["accession"]
    assert by_id[1] == f"{first}.3"  # the version NCBI returned
    assert by_id[2] == second  # unknown version: the listed accession is kept
