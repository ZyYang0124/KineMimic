"""Frame-differencing detector for small dark arthropods on substrate.

V1 detector: background model (median of sampled frames) + threshold +
morphological cleanup + connected components. Deliberately simple and
swappable -- the pipeline records the model name/version so a future
YOLO-based detector can coexist without invalidating old runs.
"""

from __future__ import annotations

import cv2
import numpy as np

MODEL_NAME = "bgdiff-v1"


class BgDiffDetector:
    def __init__(self, threshold: int = 35, min_area: int = 15, max_area: int = 2500,
                 bg_samples: int = 60):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.bg_samples = bg_samples

    def fit_background(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.unique(np.linspace(0, max(total - 1, 0), min(self.bg_samples, max(total, 1))).astype(int))
        samples = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cap.release()
        if not samples:
            raise RuntimeError(f"no frames readable from {video_path}")
        return np.median(samples, axis=0)

    def detect_frame(self, gray: np.ndarray, bg: np.ndarray) -> list[dict]:
        diff = cv2.absdiff(gray, bg.astype(np.uint8))
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        dets = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (self.min_area <= area <= self.max_area):
                continue
            cy, cx = (labels == i).nonzero()
            # orientation-invariant elongation from the minimum-area rectangle
            rect = cv2.minAreaRect(np.column_stack([cx, cy]).astype(np.float32))
            rw, rh = rect[1]
            elong = max(rw, rh) / max(min(rw, rh), 1e-9)
            # store axes so downstream aspect is the true body elongation
            dets.append(dict(
                centroid=[float(cx.mean()), float(cy.mean())],
                bbox=[float(max(rw, rh)), float(min(rw, rh))],
                elongation=float(elong),
                area=float(area),
                confidence=float(min(area / self.max_area + 0.3, 1.0)),
            ))
        return dets
