"""Core data model: the Movement Episode.

Design rules (see docs/DESIGN.md):
- The episode, not the individual animal, is the fundamental unit.
- Observation data (what the video shows) is strictly separated from
  biological annotation (what a human/model believes the animal is), so
  labels can be refined later without reprocessing video.
- Every analysis stage appends to ``processing_history``; nothing is
  silently overwritten.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Canonical biological label taxonomy. Coarse now (V1); refinements such as
# "ant -> Crematogaster" attach via annotation records, not by editing
# observations.
BIO_LABELS = ("siler", "ant", "other_spider", "other_arthropod", "unknown")

# QC / annotation-review states. Reviewing never deletes or alters the
# observation; it only sets this state (append-only annotation log).
QC_STATES = ("unreviewed", "accepted", "rejected", "tracking_failure",
             "severe_occlusion", "edge_effect", "too_short", "ambiguous_taxon")

# Sampling hierarchy: statistics must respect site > session > video > episode
# (episodes from one video are not independent replicates).
SITE_LEVELS = ("site", "session", "video", "episode")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Provenance:
    """Records how a result was produced. Nested per processing step."""

    software_version: str
    model_name: str
    model_version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    parent_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Environment:
    """Optional context metadata. All fields optional by design."""

    site: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    camera: Optional[str] = None
    fps: Optional[float] = None
    resolution: Optional[list[int]] = None
    px_per_cm: Optional[float] = None  # spatial calibration
    substrate: Optional[str] = None
    habitat: Optional[str] = None
    temperature_c: Optional[float] = None
    weather: Optional[str] = None
    condition: Optional[str] = None  # experimental condition
    field_or_lab: Optional[str] = None
    distance_to_nest_m: Optional[float] = None
    ant_activity: Optional[str] = None


@dataclass
class TrajectoryQC:
    """Quality metrics for the centroid trajectory of one episode."""

    mean_detection_confidence: float = 0.0
    coverage: float = 1.0  # fraction of frames with a valid centroid
    mean_displacement_px: float = 0.0  # per-frame jump size (occlusion check)
    flags: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.coverage >= 0.5 and "large_gaps" not in self.flags


@dataclass
class Episode:
    """One continuous, sufficiently reliable observation of one animal."""

    episode_id: str = field(default_factory=lambda: new_id("ep"))
    source_video_id: str = ""
    source_video_path: str = ""
    start_frame: int = 0
    end_frame: int = 0
    fps: float = 30.0
    px_per_cm: Optional[float] = None

    # observation payload (frames indexed from start_frame)
    frames: list[int] = field(default_factory=list)
    centroids_px: list[list[float]] = field(default_factory=list)
    bbox_sizes_px: list[list[float]] = field(default_factory=list)  # [w, h]
    detection_confidence: list[float] = field(default_factory=list)

    qc: TrajectoryQC = field(default_factory=TrajectoryQC)
    environment: Environment = field(default_factory=Environment)

    # sampling hierarchy (episode != independent biological replicate)
    site_id: Optional[str] = None
    session_id: Optional[str] = None

    # ---- MACHINE prediction (may participate in circularity; never ground truth)
    machine_label: Optional[str] = None
    machine_confidence: float = 0.0
    machine_source: Optional[str] = None      # e.g. "model:heuristic-v1"

    # ---- HUMAN annotation (independent of movement features; the only
    # ---- admissible ground truth for mimicry comparisons)
    human_label: Optional[str] = None
    human_confidence: float = 0.0
    human_source: Optional[str] = None        # "human" | "dataset:<name>"
    annotator: Optional[str] = None
    annotation_timestamp: Optional[str] = None
    annotation_status: str = "unreviewed"     # one of QC_STATES
    annotation_note: str = ""

    # legacy combined field (v0.1): kept in sync as the *effective* label so
    # older consumers (viz.py, old runs) keep working. New code should use
    # effective_label().
    bio_label: str = "unknown"
    bio_label_confidence: float = 0.0
    bio_label_source: str = "unannotated"  # unannotated | model:<name> | human
    bio_label_detail: dict[str, Any] = field(default_factory=dict)  # e.g. {"ant_genus": "Crematogaster"}

    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory_features: dict[str, float] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    motif: Optional[int] = None

    processing_history: list[dict[str, Any]] = field(default_factory=list)

    # ---- derived ----
    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps if self.fps else 0.0

    def effective_label(self) -> str:
        """The label analysis overlays may use: human annotation if present,
        else machine prediction. Embeddings never use either."""
        if self.human_label:
            return self.human_label
        if self.machine_label:
            return self.machine_label
        return self.bio_label or "unknown"

    def is_reviewed(self) -> bool:
        return self.annotation_status != "unreviewed"

    def provenance_chain(self) -> list[str]:
        """Human-readable trace from embedding back to video frames."""
        chain = [
            f"episode {self.episode_id}",
            f"video {self.source_video_id} ({self.source_video_path})",
            f"frames {self.start_frame}-{self.end_frame} @ {self.fps} fps",
        ]
        for step in self.processing_history:
            chain.append(
                f"{step.get('model_name', '?')} v{step.get('model_version', '?')} "
                f"@ {step.get('timestamp', '?')}"
            )
        return chain

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        d = dict(d)
        d.pop("duration_s", None)
        qc = TrajectoryQC(**d.pop("qc")) if d.get("qc") else TrajectoryQC()
        env = Environment(**d.pop("environment")) if d.get("environment") else Environment()
        # migrate v0.1 runs: combined bio_label -> separated machine/human fields
        if d.get("machine_label") is None and d.get("human_label") is None:
            src = d.get("bio_label_source") or "unannotated"
            if src.startswith("human") or src.startswith("dataset"):
                d["human_label"] = d.get("bio_label")
                d["human_confidence"] = d.get("bio_label_confidence", 0.0)
                d["human_source"] = src
                d["annotator"] = src
                d["annotation_status"] = "accepted"
            elif src.startswith("model:"):
                d["machine_label"] = d.get("bio_label")
                d["machine_confidence"] = d.get("bio_label_confidence", 0.0)
                d["machine_source"] = src
        return cls(qc=qc, environment=env, **d)
