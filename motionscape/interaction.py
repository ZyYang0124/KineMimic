"""Interaction Layer (V0): scenes, pairwise geometry, social context.

Layer 2 of MOTIONSCAPE's science stack:

    Movement  (how does one animal move?            — episodes, features)
    Interaction  (how does movement depend on another animal? — this module)
    Behavior  (how are movements organized?          — future)

Core objects:

- ``SceneWindow``: a sliding time window of ONE source video with all
  episodes active in it — the scene an episode lives in.
- ``InteractionRecord``: pairwise geometry of two concurrently tracked
  episodes (distance, relative bearing, heading difference, approach...).

Hard design rules (docs/INTERACTIONS.md):

- No long-term identity: only short-term concurrent tracks within one
  video. Episodes of different videos never interact.
- Spatial interaction needs xy trajectories; velocity-only episodes
  (e.g. Zeng 2023) are excluded and reported as such.
- Units are explicit: pixels unless ``px_per_cm`` calibration exists,
  never silently mixed.
- Proximity is NOT interaction; synchrony is NOT causation. All outputs
  are descriptive; the with/without comparison is tested against an
  episode-shuffle null and labeled as predictive association only.
- Scene identity (site > session > video > scene window) is preserved so
  pairs are never treated as independent biological replicates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import __version__
from .features import feature_matrix
from .hierarchy import key_of
from .schema import Episode, Provenance

SERIES_PTS = 160          # downsample resolution for stored series
DEFAULT_RADIUS_PX = 200.0  # configurable; never hardcode thresholds in analyses


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass
class SceneWindow:
    """All animals active in one video during one time window."""
    scene_window_id: str
    source_video_id: str
    site_id: str
    session_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    active_episode_ids: list[str] = field(default_factory=list)
    animal_count: int = 0
    taxon_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InteractionRecord:
    """Pairwise geometry of two episodes while both are tracked.

    One record per (focal, neighbor) ordered pair. Descriptive only:
    co-occurrence in a scene is not evidence of interaction.
    """
    focal_episode_id: str
    neighbor_episode_id: str
    source_video_id: str
    start_frame: int = 0
    end_frame: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    n_frames_aligned: int = 0
    units: str = "px"                      # "px" | "cm" (with calibration)
    px_per_cm: float | None = None

    distance_min: float = 0.0
    distance_mean: float = 0.0
    distance_max: float = 0.0
    min_distance_time_s: float = 0.0

    # neighbor position relative to focal heading (fractions of aligned time)
    frac_front: float = 0.0                # |bearing| <= 60 deg
    frac_side: float = 0.0                 # 60 < |bearing| <= 120
    frac_rear: float = 0.0                 # |bearing| > 120

    heading_difference_mean_deg: float = 0.0   # 0..180
    alignment: float = 0.0                     # mean cos(dHeading), -1..1
    closing_rate_mean: float = 0.0             # + = approaching, units/s
    focal_speed_mean: float = 0.0
    neighbor_speed_mean: float = 0.0

    distance_series: list[float] = field(default_factory=list)
    series_t_s: list[float] = field(default_factory=list)

    focal_label: str = "unknown"
    neighbor_label: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# track helpers
# --------------------------------------------------------------------------

def _xy_by_frame(ep: Episode) -> dict[int, tuple[float, float]]:
    """frame index -> centroid, keeping finite samples only."""
    out: dict[int, tuple[float, float]] = {}
    for f, c in zip(ep.frames, ep.centroids_px):
        if c is not None and np.isfinite(c).all():
            out[int(f)] = (float(c[0]), float(c[1]))
    return out


def _frame_step(ep: Episode) -> int:
    fr = np.asarray(ep.frames, int)
    if len(fr) < 2:
        return 1
    d = np.diff(fr)
    d = d[d > 0]
    return int(np.median(d)) if len(d) else 1


def _speeds_by_frame(ep: Episode) -> dict[int, float]:
    """per-frame speed (calibrated units/s) from neighboring valid samples."""
    xy = _xy_by_frame(ep)
    fr = sorted(xy)
    scale = (1.0 / ep.px_per_cm) if ep.px_per_cm else 1.0
    out: dict[int, float] = {}
    for i in range(1, len(fr)):
        f0, f1 = fr[i - 1], fr[i]
        if f1 - f0 == 0:
            continue
        v = np.hypot(xy[f1][0] - xy[f0][0], xy[f1][1] - xy[f0][1]) * scale / (f1 - f0) * ep.fps
        out[f1] = float(v)
        out.setdefault(f0, float(v))
    return out


def _headings_by_frame(ep: Episode) -> dict[int, float]:
    """per-frame heading (radians) = direction of recent motion."""
    xy = _xy_by_frame(ep)
    fr = sorted(xy)
    out: dict[int, float] = {}
    for i in range(1, len(fr)):
        f0, f1 = fr[i - 1], fr[i]
        dx, dy = xy[f1][0] - xy[f0][0], xy[f1][1] - xy[f0][1]
        if dx == 0 and dy == 0:
            continue
        out[f1] = float(np.arctan2(dy, dx))
        out.setdefault(f0, float(np.arctan2(dy, dx)))
    return out


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _ds(a: list, n: int = SERIES_PTS) -> list:
    if len(a) <= n:
        return a
    idx = np.unique(np.linspace(0, len(a) - 1, n).astype(int))
    return [a[i] for i in idx]


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------

def concurrent_groups(episodes: list[Episode]) -> dict[str, list[Episode]]:
    """Group episodes into scenes: same source video AND same site (scene
    identity, §14); keep only xy-tracked episodes.

    Velocity-only episodes (no centroids) cannot carry spatial interaction
    and are excluded (counted in analyze_run's report).
    """
    groups: dict[str, list[Episode]] = {}
    for ep in episodes:
        if len(_xy_by_frame(ep)) < 2:
            continue
        key = f"{ep.source_video_id}@{key_of(ep, 'site')}"
        groups.setdefault(key, []).append(ep)
    return {v: eps for v, eps in groups.items() if len(eps) >= 2}


def _aligned_frames(a: Episode, b: Episode) -> tuple[list[int], dict, dict]:
    xa, xb = _xy_by_frame(a), _xy_by_frame(b)
    f0, f1 = max(a.start_frame, b.start_frame), min(a.end_frame, b.end_frame)
    if f1 < f0:
        return [], xa, xb
    common = [f for f in sorted(set(xa) & set(xb)) if f0 <= f <= f1]
    return common, xa, xb


# --------------------------------------------------------------------------
# pairwise geometry
# --------------------------------------------------------------------------

def pairwise_geometry(focal: Episode, neighbor: Episode,
                      focal_label: str, neighbor_label: str) -> InteractionRecord | None:
    """Geometry of one concurrent pair over their overlapping frames."""
    common, xa, xb = _aligned_frames(focal, neighbor)
    if len(common) < 3:
        return None
    fps = focal.fps or neighbor.fps or 30.0
    scale = focal.px_per_cm or neighbor.px_per_cm
    units = "cm" if scale else "px"
    sc = (1.0 / scale) if scale else 1.0

    d = np.array([np.hypot(xa[f][0] - xb[f][0], xa[f][1] - xb[f][1]) * sc for f in common])
    t = np.array([(f - common[0]) / fps for f in common])

    ha, hb = _headings_by_frame(focal), _headings_by_frame(neighbor)
    sa, sb = _speeds_by_frame(focal), _speeds_by_frame(neighbor)
    hdiff, bearing, cosdh = [], [], []
    for f in common:
        if f in ha and f in hb:
            dh = _wrap_pi(hb[f] - ha[f])
            hdiff.append(abs(np.degrees(dh)))
            cosdh.append(np.cos(dh))
            bearing.append(np.degrees(_wrap_pi(hb[f] - ha[f])))
    dt = np.diff(t)
    closing = -(np.diff(d)) / np.where(dt > 0, dt, np.inf)     # + = approaching

    rec = InteractionRecord(
        focal_episode_id=focal.episode_id, neighbor_episode_id=neighbor.episode_id,
        source_video_id=focal.source_video_id,
        start_frame=common[0], end_frame=common[-1],
        start_time_s=round(float(t[0]), 4), end_time_s=round(float(t[-1]), 4),
        n_frames_aligned=len(common), units=units,
        px_per_cm=scale,
        distance_min=round(float(d.min()), 3), distance_mean=round(float(d.mean()), 3),
        distance_max=round(float(d.max()), 3),
        min_distance_time_s=round(float(t[int(np.argmin(d))]), 4),
        heading_difference_mean_deg=round(float(np.mean(hdiff)), 2) if hdiff else 0.0,
        alignment=round(float(np.mean(cosdh)), 4) if cosdh else 0.0,
        closing_rate_mean=round(float(np.mean(closing)), 4) if len(closing) else 0.0,
        focal_speed_mean=round(float(np.mean([sa.get(f, np.nan) for f in common
                                              if not np.isnan(sa.get(f, np.nan))])), 4)
        if any(not np.isnan(sa.get(f, np.nan)) for f in common) else 0.0,
        neighbor_speed_mean=round(float(np.mean([sb.get(f, np.nan) for f in common
                                                 if not np.isnan(sb.get(f, np.nan))])), 4)
        if any(not np.isnan(sb.get(f, np.nan)) for f in common) else 0.0,
        focal_label=focal_label, neighbor_label=neighbor_label,
        distance_series=[round(float(x), 3) for x in _ds(list(d))],
        series_t_s=[round(float(x), 3) for x in _ds(list(t))],
    )
    if bearing:
        b = np.abs(bearing)
        rec.frac_front = round(float(np.mean(b <= 60)), 4)
        rec.frac_side = round(float(np.mean((b > 60) & (b <= 120))), 4)
        rec.frac_rear = round(float(np.mean(b > 120)), 4)
    return rec


# --------------------------------------------------------------------------
# neighbor context (per focal episode)
# --------------------------------------------------------------------------

def circular_mean_deg(angles_deg: list[float]) -> float | None:
    if not angles_deg:
        return None
    r = np.radians(angles_deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def neighbor_context(focal: Episode, others: list[Episode], labels: dict[str, str],
                     radius: float, fps: float) -> dict:
    """Per-frame social context around one focal episode.

    All distances/units follow the focal episode's calibration (cm if
    calibrated, else px — ``units`` says which). ``radius`` is in the same
    units. Returns downsampled series + scalar summary; frames where the
    focal animal has no valid centroid are skipped.
    """
    scale = focal.px_per_cm
    units = "cm" if scale else "px"
    sc = (1.0 / scale) if scale else 1.0
    xa = _xy_by_frame(focal)
    sa = _speeds_by_frame(focal)
    others_xy = [(_xy_by_frame(o), _headings_by_frame(o), _speeds_by_frame(o),
                  labels.get(o.episode_id, o.effective_label()), o.episode_id)
                 for o in others if o.episode_id != focal.episode_id]

    frames = sorted(xa)
    step = max(_frame_step(focal), 1)
    rec_t, rec_dmin, rec_dant, rec_nant, rec_head, rec_act, rec_nany = [], [], [], [], [], [], []
    for f in frames:
        x, y = xa[f]
        d_any, d_ant = [], []
        ants_near: list[tuple[float, float, float]] = []   # (heading_deg, speed, dist)
        n_any = 0
        for xy, hd, sp, lbl, _eid in others_xy:
            if f not in xy:
                continue
            ox, oy = xy[f]
            dist = np.hypot(ox - x, oy - y) * sc
            d_any.append(dist)
            if dist <= radius:
                n_any += 1
            if lbl == "ant":
                d_ant.append(dist)
                if dist <= radius:
                    hdg = hd.get(f)
                    hdg = np.degrees(hdg) if hdg is not None else None
                    ants_near.append((hdg, sp.get(f, np.nan), dist))
        rec_t.append(round(f / fps, 4))
        rec_dmin.append(round(min(d_any), 3) if d_any else None)
        rec_dant.append(round(min(d_ant), 3) if d_ant else None)
        rec_nant.append(len(ants_near))
        rec_nany.append(n_any)
        heads = [a[0] for a in ants_near if a[0] is not None]
        rec_head.append(circular_mean_deg(heads) if heads else None)
        speeds = [a[1] for a in ants_near if np.isfinite(a[1])]
        rec_act.append(round(float(np.mean(speeds)), 3) if speeds else None)

    def _clean(a):
        return _ds(a)

    dant_valid = [d for d in rec_dant if d is not None]
    nant_valid = rec_nant
    with_ant_frames = sum(1 for n in nant_valid if n > 0)
    ctx = {
        "units": units,
        "radius": radius,
        "fps": fps,
        "n_neighbors": len(others_xy),
        "t_s": _clean(rec_t),
        "nearest_any_dist": _clean(rec_dmin),
        "nearest_ant_dist": _clean(rec_dant),
        "n_ants_within": _clean(rec_nant),
        "n_any_within": _clean(rec_nany),
        "mean_ant_heading_deg": _clean(rec_head),
        "ant_activity_speed": _clean(rec_act),
        "has_ant": bool(nant_valid and with_ant_frames >= 0.5 * len(nant_valid)),
        "frac_frames_with_ant": round(with_ant_frames / max(len(nant_valid), 1), 4),
        "nearest_ant_dist_min": min(dant_valid) if dant_valid else None,
        "nearest_ant_dist_mean": (round(float(np.mean(dant_valid)), 3)
                                  if dant_valid else None),
        "n_ants_mean": round(float(np.mean(nant_valid)), 3),
        "n_ants_max": int(max(nant_valid)) if nant_valid else 0,
        "ant_activity_mean": (round(float(np.mean([a for a in rec_act if a is not None])), 3)
                              if any(a is not None for a in rec_act) else None),
        "label": labels.get(focal.episode_id, focal.effective_label()),
    }
    return ctx


# --------------------------------------------------------------------------
# scene windows
# --------------------------------------------------------------------------

def build_scene_windows(video_episodes: list[Episode], video_id: str,
                        window_s: float, stride_s: float,
                        labels: dict[str, str]) -> list[SceneWindow]:
    """Sliding windows over one video; taxon counts use effective labels
    when no explicit label map entry exists."""
    fps = float(np.median([e.fps for e in video_episodes])) or 30.0
    f0 = min(e.start_frame for e in video_episodes)
    f1 = max(e.end_frame for e in video_episodes)
    win, stride = int(window_s * fps), int(stride_s * fps)
    spans = {}
    for e in video_episodes:
        xy = _xy_by_frame(e)
        if xy:
            spans[e.episode_id] = (min(xy), max(xy))
    out = []
    for w0 in range(f0, f1 + 1, max(stride, 1)):
        w1 = w0 + win
        active = [eid for eid, (a, b) in spans.items() if a <= w1 and b >= w0]
        if not active:
            continue
        by_id = {e.episode_id: e for e in video_episodes}
        counts: dict[str, int] = {}
        for eid in active:
            lbl = labels.get(eid) or by_id[eid].effective_label()
            counts[lbl] = counts.get(lbl, 0) + 1
        first = next(e for e in video_episodes if e.episode_id == active[0])
        sw = SceneWindow(
            scene_window_id=f"scene_{video_id}_{w0}",
            source_video_id=video_id,
            site_id=key_of(first, "site"), session_id=key_of(first, "session"),
            start_frame=w0, end_frame=w1,
            start_time_s=round(w0 / fps, 4), end_time_s=round(w1 / fps, 4),
            active_episode_ids=sorted(active), animal_count=len(active),
            taxon_counts=counts)
        out.append(sw)
        if w1 >= f1:
            break
    return out


# --------------------------------------------------------------------------
# run-level analysis
# --------------------------------------------------------------------------

def _d_ant_scaler(episodes: list[Episode], labels: dict[str, str]):
    """Standardized-feature-space distance-to-ant model.

    D_ant(z) = mean over dims of |z - ant_centroid| / ant_spread, computed
    in the SAME feature space the Movement Atlas embeds (labels enter only
    as the reference distribution, never into the features themselves).
    """
    eps = [e for e in episodes if e.trajectory_features]
    if len(eps) < 5:
        return None
    X, names = feature_matrix(eps)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    idx = {e.episode_id: i for i, e in enumerate(eps)}
    ant_rows = [Z[i] for i, e in enumerate(eps) if labels.get(e.episode_id) == "ant"]
    if len(ant_rows) < 3:
        return None
    A = np.asarray(ant_rows)
    ant_mu, ant_sd = A.mean(0), A.std(0) + 1e-9
    d = np.abs(Z - ant_mu) / ant_sd
    d_ant = {e.episode_id: round(float(d[idx[e.episode_id]].mean()), 4)
             for e in eps if e.episode_id in idx}
    return {"names": names, "feature_mu": mu.tolist(), "feature_sd": sd.tolist(),
            "ant_mu": ant_mu.tolist(), "ant_sd": ant_sd.tolist(),
            "d_ant": d_ant, "n_ant_reference": int(len(ant_rows))}


def analyze_run(episodes: list[Episode], window_s: float = 2.0, stride_s: float = 1.0,
                radius: float | None = None, label_source: str = "effective",
                require_human_labels: bool = False, n_shuffle: int = 200,
                seed: int = 0) -> dict:
    """Full Interaction Layer V0 analysis over one run's episodes.

    Returns records + scene windows + per-episode context + the Siler x Ant
    with/without comparison (descriptive, with an episode-shuffle null).
    """
    labels = {}
    for e in episodes:
        if require_human_labels and not e.human_label:
            labels[e.episode_id] = "unannotated"
        elif label_source == "human":
            labels[e.episode_id] = e.human_label or "unannotated"
        else:
            labels[e.episode_id] = e.effective_label()

    groups = concurrent_groups(episodes)
    n_velocity_excluded = sum(1 for e in episodes if len(_xy_by_frame(e)) < 2)

    records: list[InteractionRecord] = []
    windows: list[SceneWindow] = []
    contexts: dict[str, dict] = {}

    for vid, eps in groups.items():
        vid_raw = vid.split("@")[0]
        for i, a in enumerate(eps):
            for b in eps[i + 1:]:
                for focal, nbr in ((a, b), (b, a)):
                    rec = pairwise_geometry(focal, nbr,
                                            labels.get(a.episode_id, "unknown"),
                                            labels.get(b.episode_id, "unknown"))
                    if rec:
                        records.append(rec)
        if radius is None:
            base_r = DEFAULT_RADIUS_PX
        else:
            base_r = radius
        # calibrate radius if any episode in this video is calibrated
        scales = [e.px_per_cm for e in eps if e.px_per_cm]
        r = base_r / (scales[0] if scales else 1.0) if scales else base_r
        for e in eps:
            contexts[e.episode_id] = neighbor_context(e, eps, labels, r, e.fps)
        windows += build_scene_windows(eps, vid_raw, window_s, stride_s, labels)

    comparison = _siler_ant_comparison(episodes, contexts, labels,
                                       n_shuffle=n_shuffle, seed=seed)
    scaler = _d_ant_scaler(episodes, labels)
    if scaler:
        comparison["d_ant"] = _d_ant_contrast(scaler, contexts, labels,
                                              n_shuffle=n_shuffle, seed=seed)
        comparison["d_ant_model"] = {k: v for k, v in scaler.items()
                                     if k not in ("d_ant",)}

    # distance-response scan (§34): no single hardcoded threshold — how the
    # with/without contrast changes with the radius itself, for each metric
    # incl. D_ant (§9). Radii where one group has <3 episodes are dropped
    # (a real result of scene density, not an error).
    if len(contexts) <= 2000:
        scales = {e.source_video_id: (e.px_per_cm or 1.0) for e in episodes
                  if e.px_per_cm}
        base_r = radius if radius is not None else DEFAULT_RADIUS_PX
        siler_ids = {e.episode_id for e in episodes
                     if labels.get(e.episode_id) == "siler"
                     and e.episode_id in contexts}
        if siler_ids:
            d_ant = scaler.get("d_ant") if scaler else None
            response = []
            for frac in (0.5, 1.0, 2.0, 4.0):
                r_at = base_r * frac
                ctx_r = {}
                for gkey, pool in groups.items():
                    eps_v = [e for e in pool if e.episode_id in siler_ids]
                    if not eps_v:
                        continue
                    sc = scales.get(pool[0].source_video_id, 1.0)
                    for e in eps_v:
                        ctx_r[e.episode_id] = neighbor_context(
                            e, pool, labels, r_at / sc, e.fps)
                grp_w = [e for e in episodes if e.episode_id in siler_ids and ctx_r[e.episode_id]["has_ant"]]
                grp_wo = [e for e in episodes if e.episode_id in siler_ids and not ctx_r[e.episode_id]["has_ant"]]
                if len(grp_w) < 3 or len(grp_wo) < 3:
                    continue
                row = {"radius": round(r_at, 2),
                       "n_with": len(grp_w), "n_without": len(grp_wo)}
                for key, nice in (("speed_mean", "speed"),
                                  ("turn_rate_mean", "turning"),
                                  ("frac_time_moving", "stop_go")):
                    a = np.array([e.trajectory_features.get(key, np.nan) for e in grp_w], float)
                    b = np.array([e.trajectory_features.get(key, np.nan) for e in grp_wo], float)
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    if len(a) >= 3 and len(b) >= 3:
                        row[nice] = _contrast_entry(a, b, n_shuffle, seed)
                if d_ant:
                    da = np.array([d_ant.get(e.episode_id, np.nan) for e in grp_w], float)
                    db = np.array([d_ant.get(e.episode_id, np.nan) for e in grp_wo], float)
                    da, db = da[np.isfinite(da)], db[np.isfinite(db)]
                    if len(da) >= 3 and len(db) >= 3:
                        row["d_ant"] = _contrast_entry(da, db, n_shuffle, seed)
                response.append(row)
            if response:
                comparison["response"] = response

    human_n = sum(1 for e in episodes if e.human_label)
    prov = Provenance(
        software_version=__version__, model_name="interaction-v0", model_version="1",
        parameters=dict(window_s=window_s, stride_s=stride_s,
                        radius=(radius if radius is not None else DEFAULT_RADIUS_PX),
                        radius_units="episode calibration units (cm if calibrated, else px)",
                        label_source=label_source,
                        require_human_labels=require_human_labels,
                        n_shuffle=n_shuffle, seed=seed,
                        n_episodes_input=len(episodes),
                        n_episodes_spatial=len(contexts),
                        n_velocity_excluded=n_velocity_excluded,
                        n_videos_with_concurrency=len(groups),
                        n_human_labeled=human_n,
                        labels_note=("human-confirmed labels" if require_human_labels else
                                     "effective labels (human preferred, machine fallback) "
                                     "— machine labels are pre-screening, not ground truth")),
        parent_ids=[])
    return {
        "records": records, "windows": windows, "contexts": contexts,
        "comparison": comparison,
        "provenance": prov.to_dict(),
        "n_velocity_excluded": n_velocity_excluded,
        "n_videos_with_concurrency": len(groups),
    }


def _d_ant_contrast(scaler: dict, contexts: dict, labels: dict,
                    n_shuffle: int, seed: int) -> dict:
    """§9: does Siler sit closer to the ant behavioral distribution when
    ants are nearby? D_ant(Siler | ants nearby) vs D_ant(Siler | absent) —
    descriptive + episode-shuffle null, no assumed answer."""
    d_ant = scaler["d_ant"]
    with_a = np.array([d_ant[eid] for eid, d in d_ant.items()
                       if labels.get(eid) == "siler" and contexts.get(eid, {}).get("has_ant")],
                      float)
    wo_a = np.array([d_ant[eid] for eid, d in d_ant.items()
                     if labels.get(eid) == "siler" and eid in contexts
                     and not contexts[eid].get("has_ant")], float)
    if len(with_a) < 3 or len(wo_a) < 3:
        return {"note": "not enough Siler episodes in both contexts"}
    entry = _contrast_entry(with_a, wo_a, n_shuffle, seed + 1)
    entry["note"] = ("D_ant = mean standardized distance to the ant feature-space "
                     "centroid; contrast is predictive association, not causation")
    return entry


def _contrast_entry(a: np.ndarray, b: np.ndarray, n_shuffle: int, seed: int) -> dict:
    """Descriptive contrast of two samples: medians, MW-U, and an
    episode-shuffle null on the median difference. Predictive association
    only — no causal reading."""
    rng = np.random.default_rng(seed)
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        mannwhitneyu = None
    obs = float(np.median(a) - np.median(b))
    pool = list(a) + list(b)
    n_a = len(a)
    null = []
    for _ in range(n_shuffle):
        rng.shuffle(pool)
        null.append(float(np.median(pool[:n_a]) - np.median(pool[n_a:])))
    null = np.asarray(null)
    entry = {"median_with_ant": round(float(np.median(a)), 4),
             "median_without_ant": round(float(np.median(b)), 4),
             "n_with_ant": int(len(a)), "n_without_ant": int(len(b)),
             "observed_median_diff": round(obs, 4),
             "shuffle_null_median_diff": [round(float(np.percentile(null, 2.5)), 4),
                                          round(float(np.percentile(null, 97.5)), 4)],
             "p_perm": round(float((np.sum(np.abs(null) >= abs(obs)) + 1)
                                   / (n_shuffle + 1)), 5)}
    if mannwhitneyu:
        try:
            entry["mwu_p"] = round(float(mannwhitneyu(a, b, alternative="two-sided").pvalue), 5)
        except Exception:
            pass
    return entry


def _siler_ant_comparison(episodes: list[Episode], contexts: dict, labels: dict,
                          n_shuffle: int, seed: int) -> dict:
    """Does Siler movement differ with ants nearby vs absent?

    Descriptive contrasts (medians) + Mann-Whitney U + an episode-shuffle
    null (context labels permuted within the same video). This is
    predictive association, NOT causal evidence (docs/INTERACTIONS.md).
    """
    siler = [e for e in episodes if labels.get(e.episode_id) == "siler"
             and e.episode_id in contexts]
    with_ant = [e for e in siler if contexts[e.episode_id]["has_ant"]]
    without_ant = [e for e in siler if not contexts[e.episode_id]["has_ant"]]

    def _vals(eps, key):
        return np.array([e.trajectory_features.get(key, np.nan) for e in eps], float)

    dims = {}
    for key, nice in (("speed_mean", "speed"), ("turn_rate_mean", "turning"),
                      ("frac_time_moving", "stop-go (frac moving)")):
        a, b = _vals(with_ant, key), _vals(without_ant, key)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        dims[nice] = _contrast_entry(a, b, n_shuffle, seed)
    return dims


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def write_outputs(run_dir: str | Path, result: dict) -> dict:
    run_dir = Path(run_dir)
    p_rec = run_dir / "interactions.jsonl"
    with open(p_rec, "w", encoding="utf-8") as f:
        for r in result["records"]:
            f.write(json.dumps(r.to_dict()) + "\n")
    p_win = run_dir / "scene_windows.jsonl"
    with open(p_win, "w", encoding="utf-8") as f:
        for w in result["windows"]:
            f.write(json.dumps(w.to_dict()) + "\n")
    p_sum = run_dir / "interaction_summary.json"
    p_sum.write_text(json.dumps({
        "contexts": result["contexts"],
        "comparison": result["comparison"],
        "provenance": result["provenance"],
        "n_records": len(result["records"]),
        "n_scene_windows": len(result["windows"]),
        "n_velocity_excluded": result["n_velocity_excluded"],
        "n_videos_with_concurrency": result["n_videos_with_concurrency"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"interactions": p_rec, "scene_windows": p_win, "summary": p_sum}


def load_summary(run_dir: str | Path) -> dict | None:
    p = Path(run_dir) / "interaction_summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
