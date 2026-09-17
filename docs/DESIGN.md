# MOTIONSCAPE — Design Document

## Product shape

MOTIONSCAPE / **The Murmur** / *Ant Mimicry* — an explorable atlas of
animal movement whose first biological showcase is ant-mimicry in *Siler*
jumping spiders. The codebase is species-agnostic: labels, importers, and
the question string are showcase data, not architecture.

## Scientific question

> How does a jumping spider (*Siler*) move like an ant — in which
> behavioral dimensions, and how strongly?

Naturalistic video becomes a large collection of movement episodes; the
episodes organize into an explorable behavioral space enabling quantitative
comparison (per-dimension overlap with confidence from hierarchical
bootstrap) and qualitative discovery (watching why two movements are
similar).

## Pipeline

```
Video → Detection → Movement Episode → Trajectory ┬→ Features (interpretable)
                                                   └→ Motifs (unsupervised)
                              Behavior Space → Comparative ethology → Mimicry
       ↑ human annotation (independent layer)          ↑ labels overlaid, never
       └─ workbench: QC + biological labels              used to build the space
```

## The movement episode (fundamental unit)

`Episode` (motionscape/schema.py): episode_id, source_video_id/path, frame
range, fps, calibration, frames + centroids + bbox elongations +
confidences, trajectory QC, sampling hierarchy (site_id, session_id),
environment, annotation fields, features, embedding, motif, and
`processing_history` (list of Provenance records).

- No long-term identity; reappearance → new episode.
- QC: coverage, confidence, displacement sanity; episodes below
  `min_duration_s` are dropped (configurable).

## Observation ≠ annotation (two label layers)

Observation fields are immutable after extraction. Labels live in two
separate field groups:

- `machine_label` / `machine_confidence` / `machine_source` — model
  predictions. They share features with the movement analysis and are
  **never** ground truth (circularity).
- `human_label` / `human_confidence` / `human_source` / `annotator` /
  `annotation_timestamp` / `annotation_status` — independent biological
  annotation + QC review from the workbench (or dataset metadata).
  `annotation_status` ∈ {unreviewed, accepted, rejected, tracking_failure,
  severe_occlusion, edge_effect, too_short, ambiguous_taxon}. Rejected
  episodes keep their observation data — review never deletes.

`effective_label()` = human if present else machine; the legacy
`bio_label` field mirrors it for v0.1 consumers. Refinements
(ant → Crematogaster) append annotation records via
`EpisodeStore.add_annotation` / the workbench without reprocessing.

## Annotation log (append-only)

`<store>/annotations/annotations.jsonl` — one AnnotationRecord per review
event (label and/or QC state, annotator, timestamp, note). Applying the
log is idempotent and order-respecting (latest wins); corrections are new
records, never edits. `annotation.apply_log` replays it before analysis;
the workbench writes one record per keystroke.

## Interaction Layer (V0)

Layer 2 above movement: `SceneWindow` (sliding window of one video with
all active episodes; configurable window/stride) and `InteractionRecord`
(pairwise geometry of concurrent tracks: distance, bearing, heading
difference, alignment, closing rate, speeds; explicit px/cm units). Only
same-video + same-site episodes with xy trajectories are grouped;
velocity-only episodes are excluded. Per-episode ant context (nearest-ant
distance, count within radius, local ant heading/activity) feeds the
Siler with/without-ant contrasts, an episode-shuffle null, a
distance-response scan, and D_ant (distance to the ant feature-space
distribution). Proximity ≠ interaction; correlation ≠ causation; outputs
in `interactions.jsonl` / `scene_windows.jsonl` / `interaction_summary.json`
with full provenance. See docs/INTERACTIONS.md.

## Sampling hierarchy (anti-pseudo-replication)

Site → Session → Video → Episode is stored explicitly (site_id,
session_id, source_video_id, episode_id). `hierarchy.py` builds the tree
and provides cluster (hierarchical) bootstrap: whole sites resampled when
≥2 sites exist, else sessions, else videos. Statistical claims must report
the design levels, not bare episode counts.

## Provenance contract

Every stage appends a Provenance record: software version, model name and
version, parameters, UTC timestamp, parent ids. Runs live in
`runs/<run_id>/` with `manifest.json`; nothing is overwritten —
re-analysis creates a new run referencing the old as parent. Atlas
directories are likewise versioned (`atlas`, `atlas_v2`, …).

## Blind space contract

The behavioral embedding (PCA of standardized kinematics) and motifs
(k-means) receive **features only** — the API has no label input. Tests
enforce it: re-running `analyze` with different human labels yields
identical embeddings. Labels enter only afterwards: atlas overlays,
fingerprint, motif occupancy.

## Atlas contract (The Murmur)

| Visual channel | Data mapping |
|---|---|
| Particle position | behavioral embedding |
| Drift path | episode's sliding-window embedding path z₁…z_t (static if unsupported) |
| Flutter amplitude | speed_cv |
| Color | hidden until Reveal → effective label |
| Click | real clip + real time series + provenance chain |
| Similar movements | nearest in standardized 16-D feature space (exact, k-d tree at scale) |

Scale: `data.json` ~0.32 KB/episode; per-episode detail in
`meta/<id>.json`, fetched on selection; media generated on demand by
`motionscape serve` (cached); one canvas for all particles. Benchmarks in
`benchmarks/` and `docs/ATLAS.md`.

## Mimicry measurement

Per dimension d, Bhattacharyya coefficient between Siler and ant episode
distributions (0–1): speed dynamics, intermittency, stop–go rhythm,
turning, trajectory geometry, trajectory space. The vector is the
**Behavioral Mimicry Fingerprint** — deliberately multi-dimensional; the
UI never collapses it into a single score. Meaningful comparison requires
human-reviewed labels and the non-mimetic jumping-spider control.
