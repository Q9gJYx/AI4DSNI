# Dependencies and Attribution

DSNI is a derived dataset. This page says what each upstream source contributes, under which
terms, and how to cite it.

## Data sources

**BacDive** (Leibniz Institute DSMZ). Strain-level records: taxonomy (LPSN names), culture
media with MediaDive links, culture temperature and pH, oxygen tolerance, and the INSDC
accessions of 16S rRNA sequences. Read through the public API v2
(`https://api.bacdive.dsmz.de/v2/fetch/`), which needs no registration since February 2026.
Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on bacdive.dsmz.de.

> Schober I, Koblitz J, Sardà Carbasse J, et al. BacDive in 2025: the core database for
> prokaryotic strain data. *Nucleic Acids Research* 53(D1):D748-D756, 2025.
> doi:10.1093/nar/gkae959

**MediaDive** (Leibniz Institute DSMZ). Medium names, the pH of each medium, and recipes
(solutions, ingredients, preparation steps). Read through the public REST interface
(`https://mediadive.dsmz.de/rest/`). Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as stated on the MediaDive imprint page ("MediaDive © 2022 by DSMZ is licensed under CC BY 4.0").

> Koblitz J, Halama P, Spring S, et al. MediaDive: the expert-curated cultivation media
> database. *Nucleic Acids Research* 51(D1):D1531-D1538, 2023. doi:10.1093/nar/gkac803

**NCBI nuccore** (National Library of Medicine, NIH). The 16S rRNA sequences, fetched by
accession with E-utilities. INSDC sequence records carry no use restrictions.

> Sayers EW, Beck J, Bolton EE, et al. Database resources of the National Center for
> Biotechnology Information in 2025. *Nucleic Acids Research* 53(D1):D20-D29, 2025.
> doi:10.1093/nar/gkae979

## Software

PyTorch and Lightning (training, `LightningCLI`), scikit-learn (k-mer features, random
forests, metrics), pandas and pyarrow (tables), Biopython and requests (build only),
matplotlib (figures). Versions are pinned in `uv.lock`.

## Summary

| Component | Authored by | Status | Licence |
|:--|:--|:--|:--|
| Strain records, cultivation labels, taxonomy | BacDive, DSMZ | read at build time, derived fields redistributed | CC BY 4.0 |
| Media names, pH, recipes | MediaDive, DSMZ | read at build time, redistributed for the media classes | CC BY 4.0 |
| 16S rRNA sequences | INSDC submitters, via NCBI | fetched by accession, redistributed | no restrictions |
| Label rules, media classes, split, DSNI tables | this repository | released | CC BY 4.0 |
| Code | this repository | released | MIT |

## Not included

- **BEI Resources** and other NCBI-side cultivation records are not part of this build; they
  have no programmatic interface. See `ROADMAP.md`.
- **SILVA / SINA**. Region tags are fixed E. coli coordinates, not an alignment against
  SILVA, so SILVA is neither used nor redistributed.
