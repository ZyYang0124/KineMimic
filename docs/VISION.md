# VISION — the modern multi-animal frontend

V0.7 upgrades the original background-subtraction prototype into a
**modular, ambiguity-aware, multi-animal vision frontend**. The scientific
goal is unchanged and narrow:

> 在每一段可可靠观察的时间里，准确知道这个动物如何运动。
> (In every window of reliable observation, know exactly how one animal moves.)

Not: tracking one ant for its whole life. Not: long-term identity.

## Architecture

```
Video
  │  (camera-motion compensation, optional)
  ▼
Frame preprocessing / optional ROI
  │  (tiled high-resolution inference, ACCURATE mode)
  ▼
DetectorBackend            legacy-bgdiff | yolo | rfdetr (benchmark candidate)
  ▼
TrackerBackend             twostage (purity-first) | legacy greedy
  ▼
Short-gap recovery         predicted coast + flagged interpolation
  ▼
Ambiguity policy           prefer splitting over guessing
  ▼
Tracklets ── QC + min duration ──▶ Movement Episodes
  ▼
Existing KineMimic pipeline (features → space → motifs → The Murmur,
interaction, retrieval) — unchanged.
```

## Backends

| backend | kind | dependencies | notes |
|---|---|---|---|
| `legacy-bgdiff` | detector | OpenCV only | CPU fallback; cannot see stationary animals; class always `unknown` |
| `yolo` | detector | `ultralytics` (optional) | needs a KineMimic-trained model — stock COCO weights cannot detect ants (the quality gate will honestly reject such runs) |
| `rfdetr` | detector | `rfdetr` (optional) | benchmark candidate, not yet evaluated |
| `twostage` | tracker | none | ByteTrack-inspired two-stage association, purity-first ambiguity policy — the V0.7 default |
| legacy greedy | tracker | none | preserved for regression and trivial scenes |

Heavy dependencies are optional and import-guarded: the core package
never requires them. GPU is auto-detected; without a GPU the modern
detectors run on CPU (slow) and the legacy backend remains the fast path.

## Running on a CUDA server

```bash
# one-time setup
pip install -e .
pip install ultralytics          # YOLO backend
pip install rfdetr               # optional benchmark candidate

# verify
kinemimic vision-doctor          # GPU + backends + recommendations

# research-data run (tiled high-res, strict ambiguity policy)
kinemimic vision VIDEO.mp4 --id mysite_01 --mode accurate     --detector yolo --tile 1024 --stabilize
```

`vision-doctor` prints GPU availability, every optional dependency, and
the recommended backend/mode for this machine. The YOLO backend selects
`cuda:0` automatically when torch sees a GPU; nothing in the pipeline
needs code changes between a laptop (CPU, legacy backend) and a GPU
server (accurate mode). Weights are a separate decision: train or obtain
a KineMimic detector per docs/VISION_DATASET.md — stock COCO weights
cannot detect ants, and the quality gate will say so rather than return
empty tracks.

## Modes (never silently change scientific meaning)

| mode | detection | tracking | use |
|---|---|---|---|
| `FAST` | full-frame | two-stage | screening footage |
| `ACCURATE` | tiled high-res | two-stage + strict ambiguity + gap recovery | research data |
| `ASSISTED` | ACCURATE | + everything ambiguous flagged for human review | important clips, benchmark annotation |

Every run records mode, backend name/version, weights config, tile
geometry, thresholds, stride, ROI, camera-stabilization shifts and
software version in its provenance (append-only, as everywhere in
KineMimic).

## Quality gate (§56)

A run that cannot support the science is refused, not dressed up. Three
arms, all evaluated on the run's own numbers:

1. **no detections** — fewer than 2% of analyzed frames contain any
   detection (the detector cannot see this footage at all).
2. **low confidence** — mean detection confidence below 0.2.
3. **fragmentation** — enough footage was analyzed to contain an episode
   (>= `min_duration_s`) but no tracklet survived to episode length. The
   yardstick is the run's own `min_duration_s`, not an invented
   constant: if the median tracklet lifetime is far below it, the
   detector is following background noise — debris, leaf flecks, glare,
   a field card's handwriting (docs/VISION_DATASET.md §48) — and every
   downstream number would be fabricated.

The third arm exists because the first two only bound detection
*absence*: a 1920x1080 clip yielding thousands of one-frame tracklets
per second passed arms 1 and 2 while producing zero episodes, and was
reported as a clean run. `median_tracklet_seconds` and
`detections_per_frame` are recorded in the run stats even when a run
passes, so a suspiciously noisy video is visible before it is trusted.

## Input expectations

- Fixed-ish camera or mild shake (compensable by global translation).
- Animals ≥ a few pixels; `--min-area` / `--threshold` tune the legacy
  detector, tile size tunes the modern one.
- Any fps: kinematics always use true frame spacing (§23-24), never an
  assumed 30 fps.

## Output

Movement episodes identical in shape to the legacy pipeline — plus, per
episode: `tracklet_id`, `detector_class`, `tracking_confidence`,
`ambiguity_events`, `gap_events`, and per-point
observed/interpolated/propagated/human-corrected provenance.
