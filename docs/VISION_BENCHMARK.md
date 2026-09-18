# VISION BENCHMARK — data, not leaderboard opinions

Per the project rule: backends are chosen on the **KineMimic-specific
benchmark** (Episode Purity, Usable Episode Recall, False Merge Rate,
Trajectory Error, runtime) — never on COCO mAP or MOT leaderboard rank
alone.

## Current benchmark (synthetic, exact ground truth)

`python -m kinemimic vision-benchmark` renders five controlled scenes
(crossing, collision-merge, stationary pause, occlusion gap, fast
animal), runs complete pipelines against exact GT positions, and scores
the KineMimic metric battery. Cost = 5·false-merge + 0.5·fragmentation +
(1 − usable recall) — false merges dominate by design.

### Results (2026-09-18, `benchmarks/tracking_benchmark_v1.json`)

| pipeline | purity ↑ | usable recall ↑ | false merge ↓ | frag/scene ↓ | traj err px ↓ | COST ↓ |
|---|---|---|---|---|---|---|
| A — legacy bgdiff + legacy greedy | 0.904 | 0.671 | **0.20** | 0.6 | 0.28 | 1.63 |
| B — legacy bgdiff + **twostage** | **1.00** | **0.985** | **0.00** | 1.0 | 0.27 | **0.52** |

Reading:

- Pipeline A silently merges the collision pair (20% of episodes
  contaminated; purity drops exactly at the crossing/collision scenes).
- Pipeline B keeps purity at 1.0 across every scene by splitting at the
  ambiguous collision, pays for it with ~1 extra episode per scene
  (fragmentation), and wins the cost function decisively.
- This is the project's core tracking trade-off, quantified:
  **fragmentation is cheap, false merges are toxic.**

Limitations: synthetic scenes have clean backgrounds and known geometry;
they validate tracker *policy*, not detector recall on real footage.

## Planned: real-footage benchmark

A manually annotated set of 10–30 real clips (per docs/VISION_DATASET.md
sampling) covering: single/multiple ants, single/multiple spiders,
ant+spider together, crossings, touching ants, temporary occlusion, edge
entry/exit, stationary animal, very fast animal, dark-on-dark,
bright-on-leaf, camera shake, moving vegetation. Metrics additionally:
detection precision/recall/F1, small-object recall, class confusion,
HOTA/IDF1/MOTA as auxiliary (never sole) criteria, plus frames/s, GPU
memory, and runtime per minute of video at 1080p/4K × 30/120 fps.

## Pipeline ablation matrix (planned, §51-52)

| pipeline | question it answers |
|---|---|
| A detector + ByteTrack-style | baseline |
| B detector + BoT-SORT-class | stronger association help? |
| C detector/seg + TrackTrack-class | do modern online trackers pay off? |
| D C + short-gap point tracking | less harmful fragmentation? |
| E D + mask propagation | does SAM-class refinement justify its cost? |

Adoption is decided by the cost function above on real footage — not by
paper rank.

## Real-footage benchmark (Phase B, in progress)

Audit finding (`docs/AUDIT_V0.7.md`): synthetic scenes alone cannot
clear the vision gate — the only real field clip available exposes two
failure classes the synthetic set does not cover:

1. **field-card / setup segment** (frames 0–520 of GOPR0395): 135–153
   noise detections/frame; the old pipeline chained them into fake
   episodes (the invalid 83-episode baseline). → must become a
   **hard-negative regression scene**: expected output is *zero*
   episodes.
2. **tiny sub-threshold animals** (frames 520–6,891): legacy bgdiff
   sees nothing (median 0 det/frame); COCO YOLO11n sees nothing
   relevant (one "toilet" FP). → requires a fine-tuned detector trained
   on our own annotated clips, plus tiled-inference validation (§18).

### Real-clip GT schema & scenario matrix

Spec moved to `docs/VISION_DATASET.md` stays authoritative for classes;
scene list and layout live in `data/benchmark/README.md` (skeleton
created this phase). Key rules: bbox+centroid+identity first;
`identity_ambiguous` ⇒ expected system behavior is *split*; holdout
clips are never used for threshold tuning.

### Status

- [x] synthetic 5-scene benchmark + metrics frozen (this file, above)
- [x] real-footage failure classes identified & quantified
- [ ] extract 20–50 real clips into `data/benchmark/clips/`
- [ ] manual GT for dev/validation splits
- [ ] fine-tune detector on annotated dev split
- [ ] full-pipeline comparison incl. tiled inference on/off
- [ ] holdout evaluation → production backend decision
- [ ] QC burden measured (min review / min video)
