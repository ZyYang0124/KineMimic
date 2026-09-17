# RETRIEVAL — Find what moves like it

The user-facing promise:

> **Drop in a video. See what moves like it.**

This document explains what "moves like it" means here, exactly how each
number is computed, and what the system refuses to claim.

## What retrieval is for

Retrieval locates a candidate ant mimic **within the ant-mimicry
behavioral reference space**: the nearest ant episodes (models), nearest
mimic episodes from independent lineages, nearest controls, and closest
motifs. It is not a universal animal identifier, and it does not rank all
of Animalia — the reference space is the biologically relevant one.

## What is behavioral similarity?

Two episodes are behaviorally similar when their **movement descriptions**
are close in a fixed vector space:

```
f(video) → episodes → z_behavior ∈ R^d      distance(z_query, z_ref)
```

`f` never sees species labels, appearance, color, or background — only
the trajectory. Similar movements land close together; different movements
land far apart. Whether the two animals belong to the same species is a
SEPARATE question, answered downstream by looking up annotation metadata
of the retrieved episodes.

## The two representations

| | A — physical (`kinematic-pca`) | B — shape (`shape-series`) |
|---|---|---|
| input | 16 interpretable kinematic scalars | resampled speed/turn/moving series, per-episode z-scored |
| keeps | real speeds & distances | movement SHAPE: burst structure, rhythm |
| robust to | — (needs calibration) | unit & camera scale differences |
| answers | "does it move as fast/far?" | "does it move with the same phrasing?" |

Both are PCA baselines (no learned weights yet); both are label-blind by
API. Neighbors from the two spaces are shown side by side, and their
agreement (Jaccard@k) is reported as a stability check.

## How nearest neighbors are computed

Exact search in the chosen representation space: Euclidean distance by
default, cosine optional. `similarity = exp(-distance/3)` is a MONOTONE
display mapping, not a probability. The search backend is an interface —
exact brute force now, FAISS/HNSW drop-in later (the index version
records which backend produced a result).

## How species ranking is computed ("behaviorally similar taxa")

1. The query's episodes form a small behavioral CLOUD (multi-episode
   queries are never averaged into one point — §33/34 of the brief).
2. For each taxon, its reference episodes form another cloud.
3. Two distances are computed and both are shown:
   - **centroid distance** — query mean to taxon mean;
   - **distribution distance** — mean per-dimension Wasserstein-1
     (standardized), which sees multi-modal structure that centroids hide.
4. Per-episode best similarity is aggregated and **shrunk by sample
   size**: `score = mean_best_similarity × n/(n+5)`. A taxon with 3
   reference episodes cannot outrank a well-sampled one by luck alone.
5. The UI reports `n` (reference episode count), the human-confirmed
   fraction, and the label: **behaviorally most similar taxa — not
   predicted species identity.**

## Why similarity ≠ identity probability

The similarity number measures geometric closeness in a behavioral space
built from trajectories. It carries no calibrated probability that the
animal "is" anything. Two different species genuinely can be behaviorally
near-identical (that is what mimicry is), and one species can span
several behavioral modes. Any identity claim belongs to human annotation.

## Why taxonomic similarity and behavioral similarity are different

Phylogeny is deliberately kept OUT of the movement representation
(docs/MODEL_CARD.md). The reward is the ability to ask: *which
phylogenetically distant animals have converged on similar locomotion?*
High behavioral similarity + large taxonomic distance = candidate
behavioral convergence — a future analysis module that this architecture
reserves an interface for.

## Out-of-distribution detection

Reference episodes have their own nearest-neighbor distance distribution.
A query farther than the 95th percentile of that distribution is flagged:

> This movement lies outside the well-sampled region of the current atlas.

Novel behavior is a **valid result** — it may mean the database lacks
coverage or the animal shows an unrecorded phenotype. Queries can be kept
as candidate novel movements (they already are saved as query bundles).
The system never forces an OOD movement onto the nearest taxon.

## Query QC

Uploaded videos are not assumed valid. Each query episode reports
duration, tracking coverage, and warnings (`short episode`, `low tracking
coverage`, `no measurable motion`). A flagged episode displays
**"Low-confidence behavioral retrieval"** — results are shown for
exploration, explicitly not as evidence.

## Where results appear

In the atlas: your video's episodes become white-outlined query particles
that glide to their **real projected positions** (the atlas build's own
saved PCA projection — never a decorative location), nearby reference
movements illuminate, and the Find Similar panel lists closest episodes
(both spaces), closest motifs, behaviorally similar taxa, the similarity
breakdown, and the OOD banner. Every neighbor can be played side-by-side
with your video (normalized or real-time playback).

## Running it

```bash
python -m kinemimic build-reference RUN_A/episodes.jsonl RUN_B/episodes.jsonl \
    --out kinemimic_runs/reference --dim 10
python -m kinemimic find-similar NEW_VIDEO.mp4 --query-id my_query \
    --reference kinemimic_runs/reference --atlas <atlas_dir>
python -m kinemimic eval-retrieval kinemimic_runs/reference
# or serve with upload enabled:
python -m kinemimic serve <atlas_dir> --reference kinemimic_runs/reference
```
