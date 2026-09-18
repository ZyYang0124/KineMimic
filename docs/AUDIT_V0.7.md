# KineMimic V0.7 Audit — Baseline Freeze (Phase A, 2026-09-18)

Audit per the V0.8–V1.0 roadmap (Phase A). Commit at audit time:
`a874f37809fed491fdfcc0a0c9dd41ea0fe35933` + audit-window fixes.
Python 3.14.2, Windows 11, CPU-only.

## 1. Repository & tests

- 18+ commits, ~7,900 LOC, 66 tracked files (excl. data/runs).
- **69/69 pytest tests pass** (pipeline, annotation, roles, retrieval,
  interaction, vision, atlas-scale).
- CLI surface: 15 subcommands (demo, shamble, zeng, ingest, annotate,
  atlas, serve, viz, interact, build-reference, find-similar,
  eval-retrieval, vision, vision-benchmark, benchmark).

## 2. Reproduction results

| Target | Result |
|---|---|
| Synthetic demo | ✅ reproduces: 4 ant / 4 siler correctly separated; atlas + annotate hint built |
| Shamble 2017 atlas (228 ep) | ✅ stored run intact, served correctly |
| Zeng gait universe (64 ep) | ✅ stored run intact |
| Field video (legacy ingest) | ⚠️ **0 usable episodes** (see §3) |
| Field video (COCO YOLO11n) | ❌ useless: 0 relevant detections, 1 "toilet" FP |

## 3. Critical finding: the 83-episode field baseline was invalid

Full sequential profile of `GOPR0395_ns.MP4` (6,891 frames,
`benchmarks/field_dets_profile.json`):

- **Frames 0–520**: field-card / setup segment → 135–153 detections/frame
  of pure noise. The legacy tracker chained these into continuous fake
  tracks; the old run's **83 "episodes" were largely noise artifacts**
  (also explains 66/83 heuristic-"siler" mislabels — random noise has
  high intermittency).
- **Frames 520–6,891**: median **0** detections/frame. The real ants in
  the arena are below the bgdiff threshold / min-area (15 px²).

Conclusions:

1. The V0.7 purity-first tracker is behaving *correctly*: garbage in →
   no episodes out. The regression vs the 83-episode baseline is the
   policy working, not a defect.
2. The legacy bgdiff detector cannot see the actual animals in this
   footage. It remains valid only for controlled-substrate recordings.
3. Off-the-shelf COCO YOLO cannot solve tiny-ant detection (no class,
   scale too small). A fine-tuned model trained on our own annotated
   clips is required → **Phase B benchmark must precede any production
   backend choice** (roadmap §8–16, §64).

## 4. Frozen baseline metrics

Synthetic tracking benchmark (`python -m kinemimic vision-benchmark`,
legacy pipeline, exact GT; cost = 5·FMR + 0.5·frag + (1−recall)):

| Metric | Value |
|---|---|
| Episode purity (mean) | 0.904 |
| False merge rate | **0.20** (2/10 episodes mix ≥2 true animals — worst error class) |
| Usable episode recall | 0.671 |
| Fragmentation (extra/scene) | 0.6 |
| Trajectory error (px, mean) | 0.28 |
| Weighted cost | 1.629 |

Atlas scale: 20k episodes builds (10k → 74 s atlas build;
`benchmarks/benchmark_*.json`).

## 5. Gaps entering Phase B (priority order)

1. **No real-footage benchmark with ground truth** — synthetic-only so
   far; field-card failure not yet a regression test (roadmap §26).
2. **Tiny-ant recall unmeasured** on real footage; needs fine-tuned
   detector + tiled-inference validation (§17–18).
3. Stationary/collision synthetic scenes exist but real collisions
   unquantified (§21).
4. Heuristic species labels still surface as `bio_label` in old runs;
   must be demoted to pre-labels everywhere (§24–25).
5. QC burden (min-video-min of human review) not yet measured (§6.5).

## 6. Decisions taken

- Legacy `ingest` marked **not production-safe** for uncontrolled field
  footage until benchmark says otherwise.
- Field-test video reserved as **hard-negative + tiny-animal benchmark
  source** (its intro is the field-card regression scene).
- No new features until Phase B metrics exist (roadmap §3).
