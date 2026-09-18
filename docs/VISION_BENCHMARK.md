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
