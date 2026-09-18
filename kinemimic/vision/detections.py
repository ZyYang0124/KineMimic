"""Detection & tracklet data model for the vision frontend.

Coordinate contract (docs/TRACKING.md §Coordinates):

- All detector output is in PIXELS of the original (unstabilized) frame.
- Camera-motion compensation, if enabled, is recorded as shifts and the
  compensated coordinates are marked (source="stabilized"); the original
  pixel values always remain recoverable.
- If a spatial calibration exists it is applied downstream (episodes),
  never inside the detector. No fake cm/s without calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

CLASSES = ("ant", "spider", "other_arthropod", "unknown")


@dataclass
class Detection:
    """One object detection in one frame (pixel coordinates)."""
    frame_idx: int
    bbox: tuple[float, float, float, float]     # x1, y1, x2, y2
    confidence: float
    cls: str = "unknown"                        # one of CLASSES (coarse!)
    centroid: tuple[float, float] | None = None
    area: float | None = None
    major_axis: float | None = None             # body axis length (px)
    minor_axis: float | None = None
    orientation: float | None = None            # axis angle (rad, 180° ambiguity)
    mask: Any = None                            # optional instance mask
    tile_id: int | None = None                  # provenance of tiled inference
    source: str = "detector"

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        if self.centroid is None:
            self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if self.area is None:
            self.area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask", None)                      # masks are not serialized inline
        return d


def iou_xyxy(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Confidence-ordered greedy NMS — the tile-merge primitive (§8)."""
    order = sorted(detections, key=lambda d: -d.confidence)
    kept: list[Detection] = []
    for d in order:
        if all(iou_xyxy(d.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(d)
    return kept


# --------------------------------------------------------------------------
# tracklets
# --------------------------------------------------------------------------

POINT_STATES = ("observed", "interpolated", "propagated", "human_corrected")


@dataclass
class TrackPoint:
    """One position sample of a tracklet, with explicit provenance (§57):
    `state` says whether the position was measured (observed), linearly
    filled during a short gap (interpolated), model-propagated
    (propagated), or corrected by a human (human_corrected). Behavioral
    analysis may filter on this — silent mixing is forbidden."""
    frame_idx: int
    x: float
    y: float
    state: str = "observed"
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0                 # detection confidence (0 if not observed)
    association_confidence: float = 0.0     # track-association quality 0..1
    cls: str = "unknown"


@dataclass
class AmbiguityEvent:
    frame_idx: int
    kind: str                               # competition | merge | split
    detail: str = ""
    involved_tracklets: list[str] = field(default_factory=list)


@dataclass
class Tracklet:
    """A short-term, purity-first track: believed to be ONE animal from
    start to end. Not a long-term identity (docs/TRACKING.md)."""
    tracklet_id: str
    source_video_id: str
    detector_class: str = "unknown"
    start_frame: int = -1
    end_frame: int = -1
    points: list[TrackPoint] = field(default_factory=list)
    gap_events: list[dict] = field(default_factory=list)      # {frame, length}
    ambiguity_events: list[AmbiguityEvent] = field(default_factory=list)
    tracking_confidence: float = 1.0         # running mean of association conf
    qc_status: str = "unreviewed"
    source_backend: str = "twostage"

    def add_point(self, pt: TrackPoint):
        if not self.points:
            self.start_frame = pt.frame_idx
        self.points.append(pt)
        self.end_frame = pt.frame_idx

    def update_confidence(self, assoc: float):
        n = len(self.points)
        self.tracking_confidence = (
            (self.tracking_confidence * (n - 1) + assoc) / max(n, 1))

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame + 1) / 30.0  # caller refines with fps

    def to_dict(self) -> dict:
        return {
            "tracklet_id": self.tracklet_id,
            "source_video_id": self.source_video_id,
            "detector_class": self.detector_class,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "n_points": len(self.points),
            "n_observed": sum(1 for p in self.points if p.state == "observed"),
            "n_interpolated": sum(1 for p in self.points if p.state == "interpolated"),
            "gap_events": self.gap_events,
            "ambiguity_events": [asdict(a) for a in self.ambiguity_events],
            "tracking_confidence": round(self.tracking_confidence, 4),
            "qc_status": self.qc_status,
            "source_backend": self.source_backend,
        }
