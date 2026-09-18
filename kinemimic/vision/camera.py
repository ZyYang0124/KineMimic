"""Camera-motion compensation (§9).

Field GoPro footage shakes. A global translation estimate (phase
correlation on consecutive grayscale frames) can be subtracted from
detections so tracking sees animal motion, not camera motion.

Guardrails:

- only a GLOBAL translation is estimated; real animal motion is never
  corrected away (no per-object warping);
- the applied shifts are recorded and land in provenance;
- disabled by default — enabling is a documented config decision.
"""

from __future__ import annotations

import numpy as np


class CameraMotion:
    """Accumulates global frame-to-frame shifts (px) via phase correlation
    and exposes compensated (stabilized) coordinates."""

    def __init__(self, min_shift_px: float = 0.3, max_shift_px: float = 120.0):
        self.min_shift_px = min_shift_px
        self.max_shift_px = max_shift_px
        self.shifts: list[tuple[float, float]] = []   # per processed frame
        self.total = (0.0, 0.0)

    def update(self, prev_gray: np.ndarray | None, cur_gray: np.ndarray) -> tuple[float, float]:
        """Feed the current frame; returns the applied incremental shift."""
        if prev_gray is None or cur_gray is None or prev_gray.shape != cur_gray.shape:
            self.shifts.append((0.0, 0.0))
            return (0.0, 0.0)
        import cv2
        a = np.ascontiguousarray(prev_gray.astype(np.float32))
        b = np.ascontiguousarray(cur_gray.astype(np.float32))
        try:
            (dx, dy), _ = cv2.phaseCorrelate(a, b)
        except cv2.error:
            dx = dy = 0.0
        if not (np.isfinite(dx) and np.isfinite(dy)):
            dx = dy = 0.0
        mag = float(np.hypot(dx, dy))
        if mag < self.min_shift_px or mag > self.max_shift_px:
            dx = dy = 0.0                        # noise or scene cut: ignore
        self.total = (self.total[0] + dx, self.total[1] + dy)
        self.shifts.append((float(dx), float(dy)))
        return (float(dx), float(dy))

    def stabilized(self, x: float, y: float) -> tuple[float, float]:
        """Translate a raw detection into the compensated frame."""
        return (x - self.total[0], y - self.total[1])

    def provenance(self) -> dict:
        return {"enabled": True, "method": "phase_correlate_global_translation",
                "n_frames": len(self.shifts),
                "total_shift_px": [round(v, 2) for v in self.total],
                "min_shift_px": self.min_shift_px, "max_shift_px": self.max_shift_px,
                "note": "global translation only; real animal motion is never "
                        "corrected away"}
