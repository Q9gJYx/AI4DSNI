# DSNI data card

The release is tracked in git: the full table, the sample, the media map, the recipes and the
build manifest. Only the FASTA copy and the download cache are left out; `scripts/build_dsni.py`
rebuilds them. Release v1.0.0 is archived on Zenodo with these files:
[10.5281/zenodo.24524178](https://doi.org/10.5281/zenodo.24524178) (all versions: [10.5281/zenodo.24524177](https://doi.org/10.5281/zenodo.24524177)).

## Contents

| File | Tracked | Rows | What |
|:--|:--|--:|:--|
| `dsni_sample.parquet` | yes (0.8 MB) | 5,079 | whole genera drawn at random inside each split; same schema and metadata as the full table |
| `dsni.parquet` | yes (3.9 MB) | 20,059 | the full table, one row per strain |
| `dsni.fasta.gz` | no (3.2 MB) | 20,059 | the sequences as FASTA, header `>strain_id accession species` |
| `media_map.csv` | yes | 1,169 | every primary medium: MediaDive id, name, strain count, media class, medium pH |
| `media_compositions.json` | yes | 41 | MediaDive recipes (solutions, ingredients, steps) of the media classes |
| `manifest.json` | yes | | sources and versions, build parameters, row counts per step, checksums |
| `raw/` | no | | download cache of the build: BacDive records, MediaDive media, NCBI FASTA |

The parquet files describe themselves: `pandas.read_parquet(...).attrs` holds the build
manifest and the MediaDive id to name map. The schema and the label rules are in
[`docs/dataset.md`](../docs/dataset.md).

## Provenance

| Source | Access | Contributes |
|:--|:--|:--|
| BacDive, release 2026-06-01 | `https://api.bacdive.dsmz.de/v2/fetch/` | strain records: taxonomy, culture media, culture temperature and pH, oxygen tolerance, 16S accessions |
| MediaDive | `https://mediadive.dsmz.de/rest/` | medium names, medium pH, recipes |
| NCBI nuccore | E-utilities `efetch` | the 16S rRNA sequences, by INSDC accession |

Built on 2026-11-24. From 102,187 BacDive records to 20,059 strains:

| Step | Strains left |
|:--|--:|
| BacDive records | 102,187 |
| Bacteria | 99,746 |
| with a 16S accession | 26,970 |
| accession of 500 to 1,600 nt | 25,846 |
| sequence retrieved from NCBI | 25,844 |
| passes sequence QC (under 1% ambiguous bases) | 25,402 |
| at least one of the four labels | 20,957 |
| after removing exact-duplicate sequences | 20,059 |

Label caveats worth knowing before use: 62% of the pH labels are the pH of the cultivation
medium rather than a measured optimum of the strain (`ph_source` says which), and a strain with
contradictory oxygen reports is left without an oxygen label.

## Rebuild

```bash
uv sync --extra build
uv run --extra build python scripts/build_dsni.py
```

About 2,000 BacDive calls, one MediaDive listing, 41 recipes and 130 NCBI calls; roughly
15 minutes and 140 MB of cache under `data/raw/`. Stages are cached, so a second run takes
seconds. Set `NCBI_EMAIL` (and optionally `NCBI_API_KEY`) in the environment, as NCBI asks of
E-utilities users. A rebuild at a later date follows the then-current BacDive release, so the
counts will differ; `manifest.json` records what was used.

## Licence and attribution

The DSNI tables and the media recipes are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are derived from BacDive and
MediaDive (Leibniz Institute DSMZ), both licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and from INSDC sequence records
distributed by NCBI, which carry no use restrictions. If you use DSNI, cite the DSNI manuscript and the upstream
databases listed in [`docs/dependencies.md`](../docs/dependencies.md).

## Superseded files

`metadata.csv`, `sequences.fasta` and the earlier `media_compositions.json` (7,470 strains,
added in May 2026) are superseded by this release and removed. Their oxygen column used
`microaerophile` as a fill-in value, and their media column held raw names rather than
classes. They remain in the git history.
