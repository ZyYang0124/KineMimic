# Murmur — Design Document

## Scientific question

> How does a jumping spider (*Siler*) move like an ant — in which behavioral
> dimensions, and how strongly?

Murmur converts naturalistic video into millions-scale collections of short
movement episodes and builds an explorable behavioral space, enabling both
quantitative hypothesis tests (Siler vs ant overlap per dimension) and
qualitative discovery (watching why two episodes are similar).

## Pipeline

```
Video → Detection → Movement Episode → Trajectory ┬→ Features (interpretable)
                                                   └→ Motifs (unsupervised)
                              Behavior Space → Comparative ethology → Mimicry
```

Later phases add Pose (foreleg-I ↔ antennae comparison), behavioral motifs
from learned representations, and behavioral grammar (motif sequences).

## The movement episode (fundamental unit)

`Episode` (murmur/schema.py): episode_id, source_video_id/path, frame range,
fps, calibration, frames + centroids + bbox elongations + confidences,
trajectory QC, environment, biological annotation, features, embedding,
motif, and `processing_history` (list of Provenance records).

- No long-term identity; reappearance → new episode.
- QC: coverage, confidence, displacement sanity; episodes below
  `min_duration_s` are dropped (configurable).

## Observation vs annotation

Observation fields are immutable after extraction. Biological labels are
annotations: `bio_label`, confidence, source (`model:<name>` / `human`),
plus a free-form `bio_label_detail` dict so `ant` can later become
`ant + {genus: Crematogaster}` via appended annotation records
(`EpisodeStore.add_annotation`) without touching observations.

## Provenance contract

Every stage appends a Provenance record: software version, model name and
version, parameters, UTC timestamp, parent ids. Runs live in
`runs/<run_id>/` with `manifest.json`; nothing is overwritten — re-analysis
creates a new run that references the old as parent.

## Visualization contract (Movement Murmuration)

| Visual channel | Data mapping |
|---|---|
| Particle position | PCA embedding of kinematic features |
| Color | biological label |
| Flutter amplitude | speed_cv (movement intermittency) |
| Trail / replay | real trajectory, true fps |
| Click | provenance chain + feature table |

Flutter is the only non-positional motion and is explicitly mapped to
intermittency; no decorative animation is permitted.

## Mimicry measurement

Per dimension d, compare Siler and ant episode distributions with the
Bhattacharyya coefficient (overlap in [0,1]). Dimensions: speed dynamics,
intermittency, stop–go rhythm, turning, path shape, trajectory space.
The vector of overlaps is the **Behavioral Mimicry Fingerprint**.
Comparisons require the non-mimetic jumping-spider control to be meaningful.
