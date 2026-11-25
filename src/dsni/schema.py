"""Column and label definitions for the DSNI table, documented here once."""

from __future__ import annotations

TASKS: tuple[str, ...] = ("media", "temperature", "ph", "oxygen")
SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_RATIOS: tuple[float, ...] = (0.70, 0.15, 0.15)  # target share of rows per split
RANKS: tuple[str, ...] = ("domain", "phylum", "class", "order", "family", "genus", "species")

# Media classes are data-defined (top media by strain count plus "Other"); read them from the
# categorical dtype of the ``media`` column or from ``media_map.csv``.
CLASSES: dict[str, tuple[str, ...]] = {
    "temperature": ("psychrophile", "mesophile", "thermophile"),
    "ph": ("acidophile", "neutrophile", "alkaliphile"),
    "oxygen": ("aerobe", "facultative", "microaerophile", "anaerobe"),
}
TEMP_BINS: tuple[float, float] = (20.0, 45.0)  # < 20 C psychrophile, > 45 C thermophile
PH_BINS: tuple[float, float] = (6.0, 8.0)  # < 6 acidophile, > 8 alkaliphile
LOSS_WEIGHTS: dict[str, float] = {"media": 1.0, "temperature": 0.5, "ph": 0.5, "oxygen": 0.75}

ARM_NAMES: dict[str, str] = {  # run directory -> name used in the paper
    "kmer_rf": "k-mer + RF",
    "cnn": "CNN Encoder",
    "transformer_flat": "Transformer (Flat)",
    "transformer_gated": "Transformer + Gating",
}

PAD, UNKNOWN = 0, 5
VOCAB_SIZE = 6  # PAD, A, C, G, T, N
VOCAB: dict[str, int] = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4, "N": UNKNOWN}
IGNORE_INDEX = -100  # label value for a missing annotation; every loss and metric masks it

COLUMNS: dict[str, str] = {
    "strain_id": "Row key, `BacDive:<id>`.",
    "bacdive_id": "BacDive strain identifier.",
    "dsm_number": "DSMZ catalogue number, when the strain is held by DSMZ.",
    "type_strain": "True if BacDive marks the strain as the type strain of its species.",
    "phylum": "Phylum (LPSN name when available).",
    "class": "Class.",
    "order": "Order.",
    "family": "Family.",
    "genus": "Genus; the grouping unit of the splits.",
    "species": "Species binomial.",
    "accession": "INSDC accession, with version, of the 16S rRNA sequence used for this strain.",
    "accessions_all": "Every 16S accession BacDive lists for the strain.",
    "sequence": "16S rRNA gene sequence over A, C, G, T; any other base is N.",
    "seq_len": "Sequence length in nucleotides.",
    "temp_c": "Cultivation temperature in Celsius: median of optimum, else of growth, entries.",
    "temperature": "psychrophile (< 20), mesophile (20 to 45) or thermophile (> 45).",
    "ph_value": "Strain-level cultivation pH from BacDive, same precedence as temperature.",
    "medium_ph": "pH of the primary medium from MediaDive (mean of its pH range).",
    "ph_source": "`strain` if `ph` comes from `ph_value`, `medium` if from `medium_ph`.",
    "ph": "acidophile (< 6), neutrophile (6 to 8) or alkaliphile (> 8).",
    "oxygen": "aerobe, facultative, microaerophile or anaerobe.",
    "media_id": "MediaDive id of the primary medium (first DSMZ-linked medium with growth).",
    "media_all": "MediaDive ids of every medium on which the strain grows.",
    "media": "Media class: one of the most frequent media ids, or `Other`.",
    "split": "train, val or test; genera never cross splits.",
}
