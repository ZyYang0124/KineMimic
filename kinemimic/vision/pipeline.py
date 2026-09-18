"""Vision frontend orchestration: processing modes, quality gate,
tracklets -> movement episodes (§25-27, §37, §39-40, §56).

Modes (recorded in provenance; none silently changes scientific
meaning):

- FAST      full-frame detection, two-stage tracking, minimal overhead.
            For screening videos.
- ACCURATE  tiled high-resolution detection + strict ambiguity policy +
            gap recovery + strict QC. For real research data.
- ASSISTED  ACCURATE + flagged for human correction in the workbench
            (every ambiguous episode is surfaced for review).

The legacy ingest pipeline (`pipeline.ingest_video`) remains available
as the lightweight backend for synthetic demos and CPU-only quick runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import __version__
from ..detect import BgDiffDetector
from ..schema import Episode, Provenance, TrajectoryQC
from ..track import Track
from .backends import detector_backend, gpu_available
from .camera import CameraMotion
from .detections import Detection, TrackPoint
from .tiling import tiled_detect
from .tracker import TrackerConfig, TwoStageTracker

MODES = ("fast", "accurate", "assisted")


@dataclass
class VisionConfig:
    mode: str = "fast"                    # fast | accurate | assisted
    detector: str = "legacy"              # legacy | yolo | rfdetr
    detector_params: dict = None
    tracker_cfg: TrackerConfig = None
    tile_w: int = 1024
    tile_h: int = 1024
    tile_overlap: float = 0.2
    merge_iou: float = 0.45
    inference_stride: int = 1             # analyze every Nth frame (real dt kept)
    camera_stabilization: bool = False
    min_duration_s: float = 3.0
    max_duration_s: float = 120.0
    roi: tuple[int, int, int, int] | None = None   # x, y, w, h (recorded!)
    max_frames: int | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.detector_params is None:
            self.detector_params = {}
        if self.tracker_cfg is None:
            self.tracker_cfg = TrackerConfig()
        if self.mode in ("accurate", "assisted") and self.detector == "legacy":
            # ACCURATE with the legacy detector still gains the ambiguity-
            # aware tracker; tiling is meaningless for full-frame bgdiff.
            pass

    def provenance(self) -> dict:
        return {
            "mode": self.mode,
            "detector": self.detector,
            "detector_params": self.detector_params,
            "gpu_available": gpu_available(),
            "tiling": {"tile_w": self.tile_w, "tile_h": self.tile_h,
                       "overlap": self.tile_overlap, "merge_iou": self.merge_iou},
            "inference_stride": self.inference_stride,
            "camera_stabilization": self.camera_stabilization,
            "roi": self.roi,
            "tracker": {"max_jump_px": self.tracker_cfg.max_jump_px,
                        "max_gap_frames": self.tracker_cfg.max_gap_frames,
                        "max_interpolated_gap": self.tracker_cfg.max_interpolated_gap,
                        "ambiguity_ratio": self.tracker_cfg.ambiguity_ratio},
            "software_version": __version__,
        }


class VisionQCFailed(RuntimeError):
    """§56: the detector clearly failed on this video — do not dress the
    result up as tracks. Carry suggestions for the user."""


def run_vision_frontend(video_path: str, video_id: str,
                        cfg: VisionConfig) -> dict:
    """Video -> detections -> tracklets -> episodes (legacy Episode objects
    carrying full tracklet provenance). Raises VisionQCFailed when the
    run is obviously unusable."""
    t0 = time.time()
    backend = detector_backend(cfg.detector, video_path=video_path,
                               params=cfg.detector_params)
    cfg.tracker_cfg.source_video_id = video_id
    tracker = TwoStageTracker(cfg.tracker_cfg)
    cam = CameraMotion() if cfg.camera_stabilization else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(cfg.inference_stride))

    prev_gray = None
    n_processed = 0
    n_det_sum = 0
    n_frames_with_det = 0
    conf_sum, conf_n = 0.0, 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (cfg.max_frames and frame_idx > cfg.max_frames):
            break
        if frame_idx % stride == 0:
            work = frame
            if cfg.roi:
                x, y, w, h = cfg.roi
                work = work[y:y + h, x:x + w]
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            if cam is not None:
                cam.update(prev_gray, gray)
            if cfg.mode in ("accurate", "assisted") and cfg.detector != "legacy":
                dets = tiled_detect(work, lambda tile, tid: backend.detect(tile, frame_idx),
                                    tile_w=cfg.tile_w, tile_h=cfg.tile_h,
                                    overlap=cfg.tile_overlap, merge_iou=cfg.merge_iou)
            else:
                dets = backend.detect(work, frame_idx)
            if cam is not None:
                for d in dets:
                    cx, cy = cam.stabilized(*d.centroid)
                    dx, dy = cx - d.centroid[0], cy - d.centroid[1]
                    x1, y1, x2, y2 = d.bbox
                    d.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                    d.centroid = (cx, cy)
                    d.source = "stabilized"
            prev_gray = gray
            n_processed += 1
            n_det_sum += len(dets)
            if dets:
                n_frames_with_det += 1
            for d in dets:
                conf_sum += d.confidence
                conf_n += 1
            tracker.step(frame_idx, dets)
        frame_idx += 1
    cap.release()
    tracklets = tracker.finalize_all()

    # ---- quality gate (§56) ----
    det_frac = n_frames_with_det / max(n_processed, 1)
    mean_conf = conf_sum / max(conf_n, 1)
    total_track_s = sum((t.end_frame - t.start_frame) / fps for t in tracklets)
    if n_processed >= 10 and det_frac < 0.02:
        raise VisionQCFailed(
            f"Vision QC failed: only {det_frac:.1%} of analyzed frames had "
            f"detections. Suggestions: check detector classes against the "
            f"footage (stock COCO weights cannot detect ants — see "
            f"docs/VISION_DATASET.md), lower --threshold / raise resolution, "
            f"try a different backend, or use assisted mode.")
    if tracklets and mean_conf < 0.2:
        raise VisionQCFailed(
            f"Vision QC failed: mean detection confidence {mean_conf:.2f} is "
            f"too low for trustworthy trajectories. Suggestions: assisted "
            f"mode, a trained KineMimic detector, or manual review.")

    episodes = tracklets_to_episodes(tracklets, video_id, video_path, fps,
                                     min_duration_s=cfg.min_duration_s,
                                     max_duration_s=cfg.max_duration_s)
    analyzed_s = n_processed / fps
    tracklet_durations_s = sorted(
        (t.end_frame - t.start_frame) / fps for t in tracklets)
    median_tracklet_s = (tracklet_durations_s[len(tracklet_durations_s) // 2]
                         if tracklet_durations_s else 0.0)
    # Fragmentation check: no invented constant — the yardstick is the
    # run's own min_duration_s. If enough footage was analyzed to contain
    # an episode and none survived, nothing in this video was ever
    # reliably attributable to a single animal.
    if n_processed >= 10 and analyzed_s >= cfg.min_duration_s and not episodes:
        raise VisionQCFailed(
            f"Vision QC failed: {len(tracklets)} tracklets from "
            f"{analyzed_s:.1f}s of footage, median lifetime "
            f"{median_tracklet_s:.2f}s, {n_det_sum / max(n_processed, 1):.0f} "
            f"detections per frame -> no window long enough to belong to one "
            f"animal. The detector is tracking background noise, not animals. "
            f"Suggestions: raise --min-area / --threshold to drop debris and "
            f"leaf flecks, set --roi to exclude moving vegetation, or switch "
            f"to a trained detector (docs/VISION_DATASET.md §48 hard "
            f"negatives).")
    prov = Provenance(
        software_version=__version__, model_name=f"vision-frontend-{cfg.mode}",
        model_version="1",
        parameters=dict(
            video=video_path, video_id=video_id,
            detector_backend=backend.name, detector_version=backend.version,
            **cfg.provenance(),
            n_source_frames=n_source_frames, n_frames_analyzed=n_processed,
            inference_stride_note="kinematics use true frame spacing, not "
                                  "assumed 30 fps",
            runtime_s=round(time.time() - t0, 1),
            fps=fps))
    for ep in episodes:
        ep.processing_history.append(prov.to_dict())
    return {
        "episodes": episodes, "tracklets": tracklets, "provenance": prov.to_dict(),
        "stats": {"n_frames_analyzed": n_processed,
                  "detection_frame_fraction": round(det_frac, 4),
                  "mean_detection_confidence": round(mean_conf, 3),
                  "n_tracklets": len(tracklets),
                  "n_episodes": len(episodes),
                  "analyzed_seconds": round(analyzed_s, 1),
                  "median_tracklet_seconds": round(median_tracklet_s, 3),
                  "detections_per_frame": round(n_det_sum / max(n_processed, 1), 1),
                  "total_track_seconds": round(total_track_s, 1),
                  "runtime_s": round(time.time() - t0, 1), "fps": fps},
    }


def tracklets_to_episodes(tracklets, video_id: str, video_path: str, fps: float,
                          min_duration_s: float = 3.0,
                          max_duration_s: float = 120.0) -> list[Episode]:
    """Tracklet -> movement episode bridge (§37, §39-40, §57).

    A tracklet is not automatically an episode: minimum duration, maximum
    duration, and interpolated-fraction filtering apply, and the full
    tracklet provenance (purity-relevant metadata, point states) travels
    with the episode. Interpolated points are flagged, never silently
    mixed with observed ones."""
    episodes = []
    for trk in tracklets:
        obs = [p for p in trk.points if p.state == "observed"]
        interp = [p for p in trk.points if p.state == "interpolated"]
        if len(obs) < 2:
            continue
        duration = (trk.end_frame - trk.start_frame) / fps
        if duration < min_duration_s or duration > max_duration_s:
            continue
        n_interpolated = len(interp)
        if n_interpolated / max(len(trk.points), 1) > 0.5:
            continue                     # majority-invented track: reject
        frames = [p.frame_idx for p in trk.points]
        centroids = [[p.x, p.y] for p in trk.points]
        confs = [p.confidence for p in trk.points]
        boxes = [[p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]]
                 if p.bbox else [6.0, 6.0] for p in trk.points]
        qc = TrajectoryQC(
            mean_detection_confidence=float(np.mean([c for c in confs if c > 0]) if any(
                c > 0 for c in confs) else 0.0),
            coverage=len(obs) / max(len(trk.points), 1),
            mean_displacement_px=float(np.mean(np.hypot(
                *np.diff(np.asarray(centroids, float), axis=0).T)) ) if len(centroids) > 1 else 0.0,
            flags=(["has_interpolated_points"] if n_interpolated else []) +
                  ([f"ambiguity_events:{len(trk.ambiguity_events)}"]
                   if trk.ambiguity_events else []))
        ep = Episode(
            episode_id=f"tk_{trk.tracklet_id}",
            source_video_id=video_id, source_video_path=video_path,
            start_frame=trk.start_frame, end_frame=trk.end_frame, fps=fps,
            frames=frames, centroids_px=centroids, bbox_sizes_px=boxes,
            detection_confidence=confs, qc=qc)
        ep.metadata["tracklet_id"] = trk.tracklet_id
        ep.metadata["detector_class"] = trk.detector_class
        ep.metadata["tracking_confidence"] = round(trk.tracking_confidence, 4)
        ep.metadata["ambiguity_events"] = [
            {"frame": a.frame_idx, "kind": a.kind, "detail": a.detail}
            for a in trk.ambiguity_events]
        ep.metadata["gap_events"] = trk.gap_events
        ep.metadata["n_interpolated_points"] = n_interpolated
        ep.metadata["point_states"] = {"observed": len(obs),
                                       "interpolated": n_interpolated}
        ep.metadata["source_backend"] = trk.source_backend
        ep.processing_history.append(Provenance(
            software_version=__version__, model_name="tracklet-to-episode",
            model_version="1",
            parameters=dict(tracklet_id=trk.tracklet_id,
                            n_points=len(trk.points),
                            n_observed=len(obs), n_interpolated=n_interpolated,
                            point_state_note="observed/interpolated flags "
                                             "preserved; analysis may filter"),
            parent_ids=[trk.tracklet_id]).to_dict())
        episodes.append(ep)
    return episodes
