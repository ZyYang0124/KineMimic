"""Modern multi-animal vision frontend (V0.7).

Replaces / augments background-subtraction + greedy tracking with a
modular, ambiguity-aware pipeline:

    Video → tiled detection → two-stage tracking → gap recovery
          → ambiguity-aware tracklet splitting → movement episodes

Scientific contract (docs/TRACKING.md):

- **Episode purity over identity persistence.** When animals cross,
  touch, or merge and identity cannot be judged reliably, tracklets are
  SPLIT, never silently joined. A false merge is far worse than
  fragmentation.
- No long-term re-identification; short-term reliable tracklets only.
- Every point knows what it is: observed / interpolated / propagated.
- Every model, threshold and mode is recorded in provenance.

Backends are pluggable (docs/VISION.md); nothing here is hardwired to a
specific model. Heavy dependencies (ultralytics, rfdetr) are optional
and import-guarded — the legacy background-subtraction backend remains
the documented CPU fallback.
"""
