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
