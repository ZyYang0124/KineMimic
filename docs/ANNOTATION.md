# Annotation & QC Guide

Human review turns a machine-organized atlas into admissible science.
This guide covers the daily loop: review, correct, and trust your labels.

## Why annotate at all

Machine pre-classification shares features with the movement analysis —
using it as truth would be circular (see `SCIENTIFIC_ASSUMPTIONS.md` §3).
Only the human annotation layer is admissible ground truth for mimicry
claims.

## Launch

```bash
python -m motionscape annotate motionscape_runs/fieldtest/runs/<run_id>/episodes.jsonl
# → http://127.0.0.1:8692
```

The store root (where the append-only log lives) is auto-detected; override
with `--store PATH` if your directory layout is custom. Review progress
persists across sessions: reopening resumes at the first unreviewed episode.

## The loop: look → judge → next

- **Left**: the episode as a GIF cropped and zoomed to the animal's arena
  (real source-video frames with a faint real-trajectory overlay).
- **Right**: metadata, the machine prediction (with its circularity
  warning), the 17 kinematic features, and the provenance chain.
- **Bottom**: every shortcut, always visible.

### Label shortcuts

| key | label |
|---|---|
| `S` | Siler |
| `A` | Ant |
| `O` | Other spider |
| `R` | Other arthropod |
| `U` | Unknown |

### QC shortcuts

| key | state | meaning |
|---|---|---|
| `X` | rejected | not usable evidence |
| `F` | tracking failure | detector/tracker followed something else |
| `H` | severe occlusion | animal hidden for most of the episode |
| `E` | edge effect | behavior distorted by the arena boundary |
| `T` | too short | too brief to characterize movement |
| `B` | ambiguous taxon | cannot tell what animal this is |

`←`/`→` navigate, `Z` undo (steps back; history is never rewritten — a new
record supersedes the old).

Pressing any label key or QC key **immediately** appends one record to
`<store>/annotations/annotations.jsonl` and advances. There is no save
button because there is nothing to save — persistence happens per
keystroke.

## Guarantees

- **Append-only.** Records are never edited or deleted; the latest record
  for an episode wins. Mistakes are corrected by annotating again.
- **Observations untouched.** Applying annotations changes only label/QC
  fields; trajectories, features, and embeddings are never recomputed.
- **Separation.** `machine_label` and `human_label` live in different
  fields. Effective label = human if present, else machine. A test
  (`test_machine_vs_human_separation`) pins this contract.

## Data model

```json
{"episode_id": "ep_...", "human_label": "ant", "human_confidence": 1.0,
 "qc_state": "accepted", "annotator": "your-name", "note": "",
 "source": "human", "timestamp": "2026-09-17T09:30:00Z"}
```

Taxonomy refinements (`ant → Crematogaster`, `Siler → Siler collingwoodi`)
attach via the `note`/detail layer without touching observations.

## After annotating

Re-run the analysis; human labels now drive the overlays and the mimicry
fingerprint:

```bash
python -m motionscape atlas <run>/episodes.jsonl --out <run>/atlas_v2
python -m motionscape serve <run>/atlas_v2
```

Progress you can quote in a paper: the workbench header shows
Reviewed/Total (e.g. `1,482 / 5,236`) with per-category counts — quote
these together with the site/session/video hierarchy, never the raw
episode count alone.
