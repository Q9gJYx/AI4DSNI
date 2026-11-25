"""Shared fixtures: real BacDive records trimmed to the parsed fields, and a synthetic table."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsni import build

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_strains(n_genera: int = 30, seed: int = 0):
    """Parsed-record rows, sequences and medium pH for genera of 2 to 8 random strains."""
    rng = np.random.default_rng(seed)
    pick = lambda options: rng.choice(np.array(options, dtype=object))  # noqa: E731
    rows, sequences = [], {}
    for g in range(n_genera):
        for _ in range(int(rng.integers(2, 9))):
            i = len(rows) + 1
            accession = f"AB{i:06d}"
            sequences[accession] = "".join(
                rng.choice(list("ACGT"), size=int(rng.integers(520, 700)))
            )
            media_id = pick(["1", "65", "514", "830", "92", None])
            rows.append(
                {
                    "bacdive_id": i,
                    "dsm_number": i if rng.random() < 0.7 else None,
                    "type_strain": bool(rng.random() < 0.5),
                    "domain": "Bacteria",
                    "phylum": f"Phylum{g % 3}",
                    "class": "Class",
                    "order": "Order",
                    "family": "Family",
                    "genus": f"Genus{g:02d}",
                    "species": f"Genus{g:02d} species{i}",
                    "temp_c": pick([10.0, 30.0, 37.0, 60.0, None]),
                    "ph_value": pick([5.0, 7.0, 9.0, None, None]),
                    "oxygen": pick(["aerobe", "anaerobe", "facultative", "microaerophile", None]),
                    "media_id": media_id,
                    "media_all": [media_id] if media_id else [],
                    "accession": accession,
                    "accessions_all": [accession],
                    "release": "20260601.11",
                }
            )
    strains = pd.DataFrame(rows).astype({"temp_c": float, "ph_value": float})
    return strains, sequences, {"1": 7.0, "65": 7.2, "514": 7.6}


@pytest.fixture(scope="session")
def records():
    return json.loads((FIXTURES / "bacdive_records.json").read_text())


@pytest.fixture(scope="session")
def toy_table():
    strains, sequences, medium_ph = synthetic_strains()
    table, _, _ = build.assemble(strains, sequences, medium_ph, n_media=4)
    return table


@pytest.fixture(scope="session")
def toy_path(toy_table, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "toy.parquet"
    toy_table.to_parquet(path, index=False)
    return str(path)
