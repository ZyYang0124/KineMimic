# MODEL CARD — MOTIONSCAPE Encoders (V0.4 baseline)

## What the models are for

Behavioral retrieval: given a movement episode, find the reference
episodes / motifs / taxa that MOVE most like it. This is explicitly **not**
a species classifier — no model in this pipeline outputs P(species|video).

## Models

### Representation A — `kinematic-pca` (v1)

- **Input**: one movement episode's 16 interpretable kinematic features
  (`kinematics-v1`): speed stats, turning stats, stop–go rhythm,
  sinuosity, straightness, intermittency.
- **Process**: standardize (reference mean/sd) → PCA → z ∈ R^dim
  (dim configurable, shipped at 10).
- **Properties**: fully interpretable; translation/rotation-invariant by
  construction; physical units preserved (needs calibration for cross-
  dataset comparability).

### Representation B — `shape-series` (v2)

- **Input**: resampled (length 64) per-episode series of speed, |turning
  rate|, and moving-state, each **z-scored within the episode**.
- **Process**: concatenation → standardize (reference) → PCA → z ∈ R^dim.
- **Properties**: encodes movement SHAPE (burst structure, rhythm
  phrasing); robust to dataset unit differences (px/s vs cm/s vs mm/s)
  because absolute scale is normalized away. Physical magnitude questions
  belong to Representation A.

## Training data

No training in the learned sense: both encoders are **fit by PCA on the
reference episodes** of a given index version (shamble 2017 + Zeng 2023
episodes for `atlas_reference_v*`). Species/genus/family labels are never
input to fitting or transform (§21 of the brief; enforced by the encoder
API having no label parameter).

## Provenance & versioning

Every encoder file stores software version, feature version,
preprocessing version, parameters, and a weights checksum. Every
retrieval result records the `reference_atlas_version` it used. Index
updates create NEW versions; nothing is overwritten in place.

## Evaluation (see docs/RETRIEVAL.md)

- Augmentation consistency: rotation/translation/spatial-jitter neighbor
  Jaccard@6 = 1.0 on the current reference (invariance by construction,
  verified by test).
- Leave-video-out: neighbor sets recomputed without the query's own
  video; same-video-fraction reported against chance (with single-episode
  videos this metric is currently uninformative — it becomes meaningful
  as multi-episode field videos enter the atlas).
- Positive controls: synthetic ant / siler archetypes (known-different
  movements) separate in retrieval; no taxonomy clustering is forced.
- Human evaluation: workbench-free rating buttons in the Find Similar
  panel write `query_evaluations.jsonl` (separate data; never rewrites
  embeddings).

## Known confounds & limitations

- **Scale/units**: Representation A mixes datasets with different
  spatial calibration; uncalibrated queries may be flagged OOD in A while
  behaving normally in B (this is the intended division of labor).
- **Camera/background**: detection quality depends on footage; query QC
  flags low-coverage episodes ("low-confidence behavioral retrieval").
- **Coverage**: OOD detection only knows what the atlas has seen; a
  well-sampled hull can still be biologically narrow.
- **Not for**: species identification, individual identification, or any
  claim about animal identity from appearance — the encoders never see
  appearance.

## Out-of-distribution policy

A query whose distance to the nearest reference episode exceeds the 95th
percentile of reference-internal nearest-neighbor distances is flagged:
"This movement lies outside the well-sampled region of the current
atlas." Novel movement is a valid result, not an error.
