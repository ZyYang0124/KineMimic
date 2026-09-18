"""Detector backends (§4, §18, §54-55).

All backends implement the same surface: `detect(frame_bgr, frame_idx)
-> list[Detection]` (pixel coordinates, full frame). Heavy models are
optional and import-guarded; the legacy background-subtraction backend
is always available and is the documented CPU fallback.

Classes are COARSE visual categories (ant / spider / other_arthropod /
unknown) — observation only. Human taxon annotation (Siler collingwoodi,
Crematogaster sp., …) lives in the annotation layer and is never
overwritten by a detector class (Observation ≠ Biological Annotation).
"""

from __future__ import annotations

import numpy as np

from .detections import Detection

DETECTOR_CLASSES = ("ant", "spider", "other_arthropod", "unknown")


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def normalize_class(name: str) -> str:
    n = (name or "").strip().lower()
    if "ant" in n:
        return "ant"
    if "spider" in n:
        return "spider"
    for k in DETECTOR_CLASSES:
        if n == k:
            return k
    return "unknown"


class LegacyBgDiffBackend:
    """The original KineMimic detector (docs/VISION.md §Legacy).

    Background-subtraction, CPU-only, no dependencies beyond OpenCV.
    Class is always 'unknown' (it cannot tell ant from spider) and
    static animals fade into the background — the modern backend exists
    precisely to fix this."""

    name = "legacy-bgdiff"
    version = "1"

    def __init__(self, threshold: int = 35, min_area: int = 15,
                 max_area: int = 2500, bg_samples: int = 60,
                 video_path: str | None = None, **_):
        from ..detect import BgDiffDetector
        if video_path is None:
            raise ValueError("LegacyBgDiffBackend needs video_path to fit its "
                             "background model")
        self.det = BgDiffDetector(threshold=threshold, min_area=min_area,
                                  max_area=max_area, bg_samples=bg_samples)
        self.bg = self.det.fit_background(video_path)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out = []
        for d in self.det.detect_frame(gray, self.bg):
            x1 = d["centroid"][0] - d["bbox"][0] / 2
            y1 = d["centroid"][1] - d["bbox"][1] / 2
            # the legacy score is area-derived, not a calibrated probability;
            # rescale into stage-1 range so ByteTrack-style two-stage
            # association works unchanged. Ordering is preserved.
            conf = 0.5 + 0.45 * (float(d["confidence"]) - 0.3) / 0.7
            out.append(Detection(
                frame_idx=frame_idx,
                bbox=(x1, y1, x1 + d["bbox"][0], y1 + d["bbox"][1]),
                confidence=float(min(max(conf, 0.5), 0.95)), cls="unknown",
                area=float(d["area"])))
        return out


class YoloBackend:
    """Ultralytics YOLO adapter (optional dependency, import-guarded).

    NOTE: stock COCO weights cannot detect ants or spiders. Pass a
    KineMimic-trained model via `weights` (see docs/VISION_DATASET.md) —
    otherwise the quality gate will (honestly) reject the run."""

    name = "yolo"
    version = "1"

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25,
                 imgsz: int = 960, device: str | None = None, **_):
        try:
            from ultralytics import YOLO          # optional dependency
        except ImportError as e:
            raise RuntimeError(
                "YoloBackend requires the optional dependency group: "
                "pip install ultralytics") from e
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or ("cuda:0" if gpu_available() else "cpu")

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            cls_name = res.names[int(b.cls)]
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append(Detection(frame_idx=frame_idx, bbox=(x1, y1, x2, y2),
                                 confidence=float(b.conf), cls=normalize_class(cls_name)))
        return out


class RFDetrBackend:
    """RF-DETR adapter stub — candidate for the KineMimic detection
    benchmark (docs/VISION_BENCHMARK.md). Requires the optional `rfdetr`
    package; not installed in the current environment, so the class
    refuses loudly instead of failing silently mid-run."""

    name = "rfdetr"
    version = "0"

    def __init__(self, *a, **kw):
        raise RuntimeError(
            "RFDetrBackend requires the optional 'rfdetr' package "
            "(pip install rfdetr) and is a benchmark candidate — not yet "
            "evaluated on KineMimic footage (docs/VISION_BENCHMARK.md)")


def detector_backend(name: str, video_path: str | None = None,
                     params: dict | None = None):
    params = dict(params or {})
    if name == "legacy":
        return LegacyBgDiffBackend(video_path=video_path, **params)
    if name == "yolo":
        return YoloBackend(**params)
    if name == "rfdetr":
        return RFDetrBackend(**params)
    raise ValueError(f"unknown detector backend {name!r}; "
                     f"expected legacy | yolo | rfdetr")
