"""Interpretable trajectory kinematics features.

The interpretable route: trajectory -> speed, turning, pauses, sinuosity...
These features answer the first biological question ("does Siler walk like
an ant?") directly, and serve as the input to the behavioral-space embedding.
All features are scale-aware: if ``px_per_cm`` is known, speeds are in cm/s;
otherwise they stay in px/s and are flagged in metadata.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "kinematics-v1"


def trajectory_features(centroids: list[list[float]], fps: float,
                         px_per_cm: float | None = None,
                         frames: list[int] | None = None) -> dict[str, float]:
    def _pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q)) if len(a) else 0.0

    def _mean(a: np.ndarray) -> float:
        return float(a.mean()) if len(a) else 0.0

    xy_all = np.asarray(centroids, float).reshape(-1, 2)
    fr_all = np.asarray(frames, int) if frames is not None and len(frames) == len(xy_all) else np.arange(len(xy_all))
    # keep finite samples; a step only counts inside a continuous frame run,
    # so exit/re-entry teleports never contaminate kinematics
    keep = np.isfinite(xy_all).all(axis=1)
    xy, fr = xy_all[keep], fr_all[keep]
    if len(xy) < 2:
        # degenerate input: zero-filled feature dict keeps matrix columns aligned
        return dict.fromkeys(FEATURE_NAMES, 0.0)
    steps_ = np.diff(fr)
    step0 = int(np.bincount(steps_[steps_ > 0]).argmax()) if (steps_ > 0).any() else 1
    dt0 = step0 / (fps or 30.0)

    scale = px_per_cm if px_per_cm else 1.0  # px_per_cm: pixels per cm -> divide
    xy_c = xy / scale
    # frame spacing: 1 for native tracking, N when high-fps video was analyzed
    # with frame skipping (frames stay in original video time)
    steps = np.diff(fr)
    step = int(np.bincount(steps[steps > 0]).argmax()) if (steps > 0).any() else 1
    dt = step / fps                           # true seconds between samples
    # continuous-run tolerance: gap-bridged tracks (sparse detection) have
    # jittery spacing; +-25% of the median step still counts as a real step,
    # while arbitrary teleports never do
    tol = max(1, int(round(step * 0.25)))
    same_run = np.abs(steps - step) <= tol
    d = np.hypot(*np.diff(xy_c, axis=0).T)   # step length per sample
    v = (d / dt)[same_run]                    # speed within continuous runs
    heading = np.arctan2(*np.diff(xy_c, axis=0).T[::-1])
    step_valid = same_run[:-1] & same_run[1:]
    omega = (np.abs(np.diff(heading)) / dt)[step_valid]  # turning rate rad/s
    with np.errstate(divide="ignore", invalid="ignore"):
        curv = np.abs(np.diff(heading)) / np.maximum(d[1:], 1e-9)  # 1/cm
        curv = curv[step_valid]
    d_in = d[same_run]

    speed_thresh = max(0.1 * np.median(v), 1e-9) if len(v) else 1e-9
    moving = v > speed_thresh
    # move-pause rhythm: runs of consecutive moving / paused frames
    runs = np.diff(np.concatenate(([0], moving.view(np.int8), [0])))
    starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
    move_runs = ends - starts if len(starts) else np.array([0])
    is_move_run = moving[starts] if len(starts) else np.array([True])

    net = np.hypot(*(xy_c[-1] - xy_c[0]))
    path = d_in.sum()
    return {
        "duration_s": len(xy) * dt,
        "distance": float(path),
        "net_displacement": float(net),
        "sinuosity": float(path / max(net, 1e-9)),          # 1 = straight
        "speed_mean": _mean(v),
        "speed_cv": (float(v.std() / max(v.mean(), 1e-9)) if len(v) and v.mean() > 0 else 0.0),
        "speed_p90": _pct(v, 90),
        "accel_rms": float(np.sqrt(_mean(np.diff(v) ** 2)) / dt),
        "turn_rate_mean": _mean(omega),
        "turn_rate_p90": _pct(omega, 90),
        "curvature_mean": float(np.nanmean(curv)) if len(curv) else 0.0,
        "frac_time_moving": _mean(moving),
        "n_pauses": int((~is_move_run).sum()),
        "pause_run_mean_s": float((move_runs[~is_move_run].mean() / fps)
                                  if (~is_move_run).any() else 0.0),
        "move_run_mean_s": float((move_runs[is_move_run].mean() / fps)
                                 if is_move_run.any() else 0.0),
        "straightness_index": float(np.hypot(*(xy_c[-1] - xy_c[0])) / max(path, 1e-9)),
    }


FEATURE_NAMES = list(trajectory_features([[0, 0], [1, 0]], 30.0).keys())


def feature_matrix(episodes) -> tuple[np.ndarray, list[str]]:
    rows, names = [], None
    for ep in episodes:
        feats = ep.trajectory_features or trajectory_features(ep.centroids_px, ep.fps, ep.px_per_cm, ep.frames)
        if names is None:
            names = sorted(feats)
        rows.append([feats[k] for k in names])
    return np.asarray(rows, float), names or []
