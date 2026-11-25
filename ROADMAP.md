# Roadmap

What this repository does not do yet, in the order it matters for the manuscript.

## Dataset

- **DOI.** Tag a GitHub release with the Zenodo integration switched on; the archived snapshot
  includes the tracked data. Then add the DOI badge and `CITATION.cff` `doi`, and point
  `dsni.data.DSNI_URL` / `DSNI_SHA256` at the archived `dsni.parquet` for installs without the
  repository.
- **Species-level extension.** 20,059 strains have a 16S sequence of their own. About 24,500 more
  labelled strains have none; letting them borrow the sequence of another strain of their species would
  give about 45,500 strains. It must be an opt-in table flagged per row (`sequence_source`), because
  strains of one species then share an input while agreeing on the medium only 71% of the time
  (oxygen 80%, temperature and pH above 99%). Sources beyond BacDive (BEI Resources, NCBI-side
  cultivation records) have no programmatic interface and are not rebuilt here.
- **97% OTU dereplication.** Not applied. It needs an external clustering tool (for example
  `vsearch --cluster_fast`) and would shrink the table several-fold.
- **Alignment-based region tags.** The variability gate uses fixed E. coli coordinates.
  Annotating regions by alignment (SINA against SILVA, or covariance models) would be exact
  for partial sequences.
- **Composition-based media classes.** Classes are media ids ranked by frequency. The recipes
  in `media_compositions.json` allow grouping by base composition and an ingredient-level
  similarity for evaluation.
- **Held-out families or orders.** `family` and `order` are in the table; a stricter split is
  one `group_col` argument away in `dsni.splits.greedy_stratified_split`.

- **Residual near-duplicates across splits.** A fraction of a percent of test strains keep a near-identical
  training sequence under another genus name. Audit taxonomic synonyms (LPSN names are missing for some
  records), or split over sequence-similarity clusters instead of genus names.
- **Input snapshot.** Deposit the raw download cache (`data/raw/`) with the release so a build can be
  repeated exactly; a rebuild at a later date follows the then-current BacDive release.

## Evaluation

- **Acceptable media.** Media accuracy scores only the primary medium. Count a prediction as correct when
  it is any medium in `media_all`.
- **pH by source.** Report the pH task separately for strain-level values and for medium-derived values.
- **Macro-F1 class universe.** Fix the set of classes per task instead of taking the classes present in
  each evaluated subset, so bootstrap draws share one denominator.

## Models with no code here

These rows of the manuscript's tables cannot be produced from this repository:

- `RNABERT` and `RNABERT + Gating` (the earlier wrapper did not train and was removed)
- `GROWREC`
- `k-NN (Sequence)`
- `k-mer + XGBoost`

## Experiments

- `scripts/reproduce_paper.py` has not been run at paper scale. The result tables of the
  manuscript need to be regenerated on this release before they are quoted against it.
