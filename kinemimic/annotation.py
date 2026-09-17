"""Independent human annotation + QC review system.

Scientific contract
-------------------
Machine predictions (``classify.py``) are derived from movement features --
the very quantities the mimicry analysis studies. Using them as ground truth
would be circular. This module therefore maintains a *separate*, append-only
log of human annotations; applying it never touches observation data or
machine predictions, and every entry can be superseded (latest record wins),
so label refinements need no reprocessing.

Log record fields: episode_id, human_label, human_confidence, qc_state,
annotator, note, source, timestamp.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .schema import BIO_LABELS, QC_STATES, Episode


@dataclass
class AnnotationRecord:
    """One human review event for one episode. Appended, never edited."""

    episode_id: str
    human_label: str | None = None       # one of BIO_LABELS, or None (QC only)
    human_confidence: float = 1.0
    qc_state: str = "accepted"           # one of QC_STATES
    annotator: str = "anonymous"
    note: str = ""
    source: str = "human"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict:
        return asdict(self)


def annotation_log_path(root: str | Path) -> Path:
    return Path(root) / "annotations" / "annotations.jsonl"


def append_record(root: str | Path, rec: AnnotationRecord) -> Path:
    """Persist one review event. Append-only: history is never rewritten."""
    if rec.human_label is not None and rec.human_label not in BIO_LABELS:
        raise ValueError(f"unknown label {rec.human_label!r}; expected one of {BIO_LABELS}")
    if rec.qc_state not in QC_STATES:
        raise ValueError(f"unknown qc_state {rec.qc_state!r}; expected one of {QC_STATES}")
    path = annotation_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict()) + "\n")
    return path


def read_log(root: str | Path) -> list[AnnotationRecord]:
    path = annotation_log_path(root)
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out.append(AnnotationRecord(**json.loads(line)))
    return out


def apply_record(ep: Episode, rec: AnnotationRecord) -> None:
    """Apply one record to an episode. Only annotation fields change;
    observation data and machine predictions are untouched."""
    if rec.episode_id != ep.episode_id:
        raise ValueError("record targets a different episode")
    if rec.human_label is not None:
        ep.human_label = rec.human_label
        ep.human_confidence = rec.human_confidence
        ep.human_source = rec.source
        ep.annotator = rec.annotator
        ep.annotation_timestamp = rec.timestamp
    if rec.note:
        ep.annotation_note = rec.note
    ep.annotation_status = rec.qc_state
    # bio_label mirrors the effective label for v0.1 consumers (viz.py)
    ep.bio_label = ep.effective_label()
    ep.bio_label_confidence = (ep.human_confidence if ep.human_label
                               else ep.machine_confidence)
    ep.bio_label_source = (ep.human_source or ep.machine_source
                           or ep.bio_label_source)


def apply_log(episodes: list[Episode], root: str | Path) -> int:
    """Replay the whole annotation log (latest record per episode wins).

    Returns the number of episodes whose annotation changed. Episodes are
    matched by id; unknown ids are ignored (the log may span supersets).
    """
    by_id = {e.episode_id: e for e in episodes}
    latest: dict[str, AnnotationRecord] = {}
    for rec in read_log(root):            # chronological; later overwrites earlier
        if rec.episode_id in by_id:
            latest[rec.episode_id] = rec
    for eid, rec in latest.items():
        apply_record(by_id[eid], rec)
    return len(latest)


def summary(episodes: list[Episode]) -> dict:
    """Counts for the workbench progress display (and run summaries)."""
    by_status: dict[str, int] = {s: 0 for s in QC_STATES}
    by_label: dict[str, int] = {}
    machine_only = human = 0
    for ep in episodes:
        by_status[ep.annotation_status if ep.annotation_status in by_status else "unreviewed"] += 1
        lbl = ep.effective_label()
        by_label[lbl] = by_label.get(lbl, 0) + 1
        if ep.human_label:
            human += 1
        elif ep.machine_label:
            machine_only += 1
    reviewed = sum(v for k, v in by_status.items() if k != "unreviewed")
    return {"total": len(episodes), "reviewed": reviewed,
            "by_status": by_status, "by_effective_label": by_label,
            "n_human_labeled": human, "n_machine_only": machine_only}
