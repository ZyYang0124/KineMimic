"""Short-term greedy tracker and episode extraction.

Tracking here exists only to assemble *movement episodes*: continuous,
reliable observations ending when the animal disappears, is occluded, or
exits the frame. Long-term identity is intentionally not maintained --
if the same animal returns, it becomes a new episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import Episode, TrajectoryQC


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)
    centroids: list[list[float]] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    @property
    def last_pos(self) -> np.ndarray:
        return np.array(self.centroids[-1])

    @property
    def last_frame(self) -> int:
        return self.frames[-1]


class GreedyTracker:
    """Nearest-neighbour association within a max jump distance."""

    def __init__(self, max_jump_px: float = 60.0):
        self.max_jump_px = max_jump_px
        self.tracks: list[Track] = []
        self._next_id = 0

    def step(self, frame_idx: int, dets: list[dict]) -> None:
        live = [t for t in self.tracks if t.last_frame == frame_idx - 1]
        used = set()
        for t in live:
            if not dets:
                break
            dists = [np.hypot(*(np.array(d["centroid"]) - t.last_pos)) for d in dets]
            j = int(np.argmin(dists))
            if j not in used and dists[j] <= self.max_jump_px:
                t.frames.append(frame_idx)
                t.centroids.append(dets[j]["centroid"])
                t.bboxes.append(dets[j]["bbox"])
                t.confidences.append(dets[j]["confidence"])
                used.add(j)
        for j, d in enumerate(dets):
            if j not in used:
                tr = Track(self._next_id); self._next_id += 1
                tr.frames.append(frame_idx)
                tr.centroids.append(d["centroid"])
                tr.bboxes.append(d["bbox"])
                tr.confidences.append(d["confidence"])
                self.tracks.append(tr)


def tracks_to_episodes(tracks: list[Track], video_id: str, video_path: str, fps: float,
                       min_duration_s: float = 3.0, max_gap_frames: int = 0,
                       px_per_cm: float | None = None) -> list[Episode]:
    """Convert short tracks to episodes; drop fragments shorter than the
    minimum reliable duration. (With max_gap_frames=0 any dropout ends the
    episode, which matches the episode definition: reliability first.)"""
    episodes = []
    min_frames = int(min_duration_s * fps)
    for tr in tracks:
        if len(tr.frames) < min_frames:
            continue
        qc = TrajectoryQC(
            mean_detection_confidence=float(np.mean(tr.confidences)),
            coverage=1.0,
            mean_displacement_px=float(np.mean(np.hypot(np.diff(tr.centroids, axis=0)[:, 0],
                                                        np.diff(tr.centroids, axis=0)[:, 1]))),
        )
        episodes.append(Episode(
            source_video_id=video_id, source_video_path=video_path,
            start_frame=tr.frames[0], end_frame=tr.frames[-1], fps=fps,
            px_per_cm=px_per_cm,
            frames=tr.frames, centroids_px=tr.centroids, bbox_sizes_px=tr.bboxes,
            detection_confidence=tr.confidences, qc=qc,
        ))
    return episodes
