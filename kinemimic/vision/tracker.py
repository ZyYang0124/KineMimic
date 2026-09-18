"""TwoStageTracker — a self-contained, purity-first multi-animal tracker.

ByteTrack-inspired (motion-gated two-stage association over high- and
low-confidence detections) but deliberately simple, dependency-free and
governed by the KineMimic ambiguity policy:

    prefer splitting over guessing (docs/TRACKING.md).

Per frame:
  1. predict each active track's position (constant velocity, decayed);
  2. stage 1 — associate HIGH-confidence detections:
       cost = gated motion distance (+ small IoU bonus);
       a match is accepted only if it is clearly better than the
       runner-up; otherwise every involved tracklet is terminated at its
       last confident frame and the contested detections seed NEW
       tracklets (an ambiguity/split event is recorded);
  3. merge ambiguity — if two active tracks both need the same single
       detection, that is a collision the system cannot resolve: both
       tracklets terminate, one new tracklet continues from the
       detection;
  4. stage 2 — remaining LOW-confidence detections may extend tracks that
       lost their stage-1 match (flicker continuity), under a stricter
       gate and never through ambiguity;
  5. unmatched tracks coast (velocity decay) up to `max_gap_frames`;
       beyond that they are finalized. Re-association after a short miss
       inserts `interpolated` points (flagged) up to
       `max_interpolated_gap`; a longer disappearance starts a NEW
       tracklet (gap event recorded).

Association confidence per point = exp(-cost / soft_scale) — a real
quality estimate, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detections import Detection, TrackPoint, Tracklet, AmbiguityEvent, iou_xyxy


@dataclass
class TrackerConfig:
    det_high: float = 0.5            # stage-1 confidence threshold
    det_low: float = 0.15            # stage-2 confidence threshold
    max_jump_px: float = 60.0        # per-frame association gate (scaled by gap)
    max_gap_frames: int = 8          # coast window; beyond -> finalize
    max_interpolated_gap: int = 4    # fill interpolated points up to this gap
    ambiguity_ratio: float = 0.55    # runner-up cost below ratio*best -> ambiguous
    ambiguity_margin_px: float = 12.0  # or runner-up within this absolute margin
    iou_bonus: float = 30.0          # px-equivalent reward for bbox overlap
    soft_scale: float = 25.0         # association confidence softness
    merge_dist_px: float = 18.0      # two tracks this close to one det: collision
    source_video_id: str = "video"
    source_backend: str = "twostage-v1"


class _Active:
    """Internal per-track state."""

    def __init__(self, tracklet: Tracklet, det: Detection, assoc: float):
        self.tracklet = tracklet
        self.last_pos = np.asarray(det.centroid, float)
        self.velocity = np.zeros(2)
        self.last_frame = det.frame_idx
        self.missed = 0
        self.tracklet.add_point(TrackPoint(
            frame_idx=det.frame_idx, x=det.centroid[0], y=det.centroid[1],
            state="observed", bbox=det.bbox, confidence=det.confidence,
            association_confidence=assoc, cls=det.cls))
        self.tracklet.detector_class = det.cls

    def predict(self, frame_idx: int) -> np.ndarray:
        dt = max(frame_idx - self.last_frame, 1)
        decay = 0.9 ** (dt - 1)                       # velocity decays while coasting
        return self.last_pos + self.velocity * (dt - 1) * decay


class TwoStageTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.active: list[_Active] = []
        self.finished: list[Tracklet] = []
        self._next_id = 0

    # -- helpers ----------------------------------------------------------
    def _new_tracklet_id(self) -> str:
        self._next_id += 1
        return f"trk_{self._next_id:05d}"

    def _spawn(self, det: Detection, inherited_velocity: np.ndarray | None = None,
               parent_id: str | None = None) -> _Active:
        tid = self._new_tracklet_id()
        trk = Tracklet(tracklet_id=tid,
                       source_video_id=self.cfg.source_video_id,
                       source_backend=self.cfg.source_backend)
        act = _Active(trk, det, assoc=1.0)
        if inherited_velocity is not None:
            act.velocity = np.asarray(inherited_velocity, float)
        if parent_id:
            act.tracklet.ambiguity_events.append(AmbiguityEvent(
                frame_idx=det.frame_idx, kind="spawn",
                detail=f"continues motion after ambiguity split of {parent_id}",
                involved_tracklets=[parent_id]))
        self.active.append(act)
        return act

    def _finalize(self, act: _Active):
        self.finished.append(act.tracklet)
        if act in self.active:
            self.active.remove(act)

    def _terminate_with_split(self, act: _Active, frame_idx: int, kind: str,
                              detail: str, involved: list[str] | None = None):
        """Purity policy: end this tracklet at its last confident frame."""
        ev = AmbiguityEvent(frame_idx=frame_idx, kind=kind, detail=detail,
                            involved_tracklets=[act.tracklet.tracklet_id]
                                               + (involved or []))
        act.tracklet.ambiguity_events.append(ev)
        self._finalize(act)

    def _cost(self, act: _Active, frame_idx: int, det: Detection) -> float | None:
        pred = act.predict(frame_idx)
        gap = max(frame_idx - act.last_frame, 1)
        d = float(np.hypot(det.centroid[0] - pred[0], det.centroid[1] - pred[1]))
        gate = self.cfg.max_jump_px * gap
        if d > gate:
            return None
        c = d
        if det.bbox and act.tracklet.points:
            last = act.tracklet.points[-1].bbox
            if last:
                c -= self.cfg.iou_bonus * iou_xyxy(last, det.bbox)
        return max(c, 0.0)

    def _associate(self, tracks: list[_Active], dets: list[Detection],
                   frame_idx: int) -> tuple[dict, list, dict]:
        """Greedy best-cost assignment with ambiguity veto. Returns
        (assign {track_index: det_index}, leftover_detections,
        ambiguous {track_index: contested det}). Tracks with NO candidate
        are simply absent — they coast; only genuinely contested tracks
        are split by the caller."""
        pairs = []
        for ti, act in enumerate(tracks):
            costs = []
            for di, det in enumerate(dets):
                c = self._cost(act, frame_idx, det)
                if c is not None:
                    costs.append((c, di))
            costs.sort()
            if not costs:
                continue
            best, best_di = costs[0]
            runner = costs[1][0] if len(costs) > 1 else float("inf")
            # two tests: multiplicative ratio AND an absolute margin — the
            # ratio alone fails when the best cost is ~0 (coincident
            # detections), which would silently swap identities
            ambiguous = (runner < self.cfg.ambiguity_ratio * max(best, 1e-6)
                         or (runner - best) < self.cfg.ambiguity_margin_px)
            pairs.append((best, ti, best_di, ambiguous))
        pairs.sort(key=lambda p: p[0])
        assign: dict[int, int] = {}
        used_dets: set[int] = set()
        ambiguous: dict[int, int] = {}
        for best, ti, di, amb in pairs:
            if ti in assign or ti in ambiguous:
                continue
            if di in used_dets:
                competing = [t2 for t2, d2 in assign.items() if d2 == di]
                if competing:
                    ambiguous[ti] = di           # contested detection
                    continue
            if amb and len(dets) > 1:
                ambiguous[ti] = di               # runner-up nearly as good
                continue
            assign[ti] = di
            used_dets.add(di)
        leftover = [d for di, d in enumerate(dets) if di not in used_dets]
        return assign, leftover, ambiguous

    # -- main step ---------------------------------------------------------
    def step(self, frame_idx: int, detections: list[Detection]):
        cfg = self.cfg
        high = [d for d in detections if d.confidence >= cfg.det_high]
        low = [d for d in detections if cfg.det_low <= d.confidence < cfg.det_high]

        # ---- stage 1: high-confidence association -----------------------
        assign, leftover_high, ambiguous = self._associate(self.active, high,
                                                           frame_idx)
        # resolve EVERYTHING to object references before any mutation
        assign_acts = [(self.active[ti], high[di]) for ti, di in assign.items()]
        ambiguous_pairs = [(self.active[ti], high[di]) for ti, di in ambiguous.items()]

        # ambiguity splits (contested detections only — tracks with no
        # candidate simply coast): terminate at the last confident frame;
        # the contested detection seeds a fresh tracklet that INHERITS the
        # terminated track's motion state — motion continuity, not identity
        # guessing (the new tracklet still carries its ambiguity event)
        claimed: dict[int, tuple[_Active, Detection]] = {}
        assigned_dets = {id(d) for _, d in assign_acts}
        for act, det in ambiguous_pairs:
            # a track vetoed because its best detection is already claimed
            # by a surviving track = collision/merge; a vetoed track with
            # its own contested detection = crossing competition
            kind = "merge" if id(det) in assigned_dets else "competition"
            claimed[id(det)] = (act, det)
            self._terminate_with_split(
                act, frame_idx, kind,
                "identity ambiguous at collision -> split (prefer splitting "
                "over guessing)" if kind == "merge" else
                "runner-up association nearly as good as best; identity "
                "ambiguous -> split (prefer splitting over guessing)")
        self.active = [a for a in self.active
                       if a not in [act for act, _ in ambiguous_pairs]]

        # merge ambiguity: >=2 surviving tracks converging on the SAME
        # leftover detection -> collision the system cannot resolve
        survivors = [a for a in self.active if a.last_frame != frame_idx]
        if len(survivors) >= 2:
            for det in leftover_high:
                converging = [a for a in survivors
                              if self._cost(a, frame_idx, det) is not None and
                              np.hypot(*(np.asarray(a.predict(frame_idx)) -
                                         np.asarray(det.centroid))) <= cfg.merge_dist_px]
                if len(converging) >= 2:
                    ids = [a.tracklet.tracklet_id for a in converging]
                    for a in converging:
                        self._terminate_with_split(
                            a, frame_idx, "merge",
                            f"{len(converging)} tracks converge on one detection",
                            involved=ids)
                        self.active.remove(a)
                    survivors = [a for a in survivors if a not in converging]

        for act, det in assign_acts:
            self._commit(act, frame_idx, det, gap_filled=act.missed > 0)
            act.missed = 0

        # births: contested dets continue with inherited motion; truly new
        # dets start fresh
        for di, det in enumerate(high):
            if di in assign.values():
                continue
            inh, parent = None, None
            if di in claimed:
                parent_act = claimed[di]
                inh = parent_act.velocity
                parent = parent_act.tracklet.tracklet_id
            self._spawn(det, inherited_velocity=inh, parent_id=parent)

        # ---- stage 2: low-confidence flicker continuity -----------------
        unmatched = [a for a in self.active if a.last_frame != frame_idx]
        if low and unmatched:
            assign2, _, _ = self._associate(unmatched, low, frame_idx)
            for ti, di in assign2.items():
                act = unmatched[ti]
                det = low[di]
                self._commit(act, frame_idx, det,
                             gap_filled=act.missed > 0, low_conf=True)
                act.missed = 0

        # ---- coast / finalize -------------------------------------------
        for act in list(self.active):
            if act.last_frame != frame_idx:
                act.missed += 1
                if act.missed > cfg.max_gap_frames:
                    self._finalize(act)

        # ---- births -------------------------------------------------------
        for det in leftover_high:
            self._spawn(det)

    def _commit(self, act: _Active, frame_idx: int, det: Detection,
                gap_filled: bool, low_conf: bool = False):
        cost = self._cost(act, frame_idx, det)
        assoc = float(np.exp(-(cost if cost is not None else 999.0)
                             / self.cfg.soft_scale))
        assoc = min(1.0, max(assoc, 0.05))
        if gap_filled:
            gap = frame_idx - act.last_frame
            act.tracklet.gap_events.append(
                {"frame": act.last_frame, "length": int(gap)})
            if gap <= self.cfg.max_interpolated_gap:
                p0 = np.array([act.tracklet.points[-1].x, act.tracklet.points[-1].y])
                p1 = np.asarray(det.centroid, float)
                for k in range(1, gap):
                    q = p0 + (p1 - p0) * k / gap
                    act.tracklet.add_point(TrackPoint(
                        frame_idx=act.last_frame + k, x=float(q[0]), y=float(q[1]),
                        state="interpolated", confidence=0.0,
                        association_confidence=assoc * 0.5, cls=det.cls))
            new_pt = TrackPoint(
                frame_idx=frame_idx, x=det.centroid[0], y=det.centroid[1],
                state="observed", bbox=det.bbox, confidence=det.confidence,
                association_confidence=assoc, cls=det.cls)
            prev = act.tracklet.points[-1]
            act.velocity = (p1 - np.array([prev.x, prev.y])) / gap
        else:
            new_pt = TrackPoint(
                frame_idx=frame_idx, x=det.centroid[0], y=det.centroid[1],
                state="observed", bbox=det.bbox, confidence=det.confidence,
                association_confidence=assoc, cls=det.cls)
            prev = act.tracklet.points[-1]
            dt = max(frame_idx - prev.frame_idx, 1)
            act.velocity = (np.asarray(det.centroid, float) -
                            np.array([prev.x, prev.y])) / dt
        act.tracklet.add_point(new_pt)
        act.tracklet.update_confidence(assoc)
        act.last_pos = np.asarray(det.centroid, float)
        act.last_frame = frame_idx

    def finalize_all(self) -> list[Tracklet]:
        for act in list(self.active):
            self._finalize(act)
        return list(self.finished)
