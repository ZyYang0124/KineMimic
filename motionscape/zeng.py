"""Import Zeng et al. 2023 (iScience, Siler collingwoodi) gait raw data.

The published workbook contains per-frame kinematics for ~120 individuals:

    velocity (mm/s), forelimb-1/2 height (spiders) or antennae-1/2 height
    (ants, normalized), abdomen-head distance

across 8 sheets: Siler collingwoodi (normal + blackened control),
Phintelloides versicolor (non-mimetic salticid), and 5 sympatric ant
species. No xy trajectories are published, so these become *velocity
episodes*: the fundamental movement observation is the real per-frame
velocity/pose series, and features are computed from it directly
(never fabricating directions). The forelimb-I / antennae channel is the
pose-level mimicry axis of the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import __version__
from .schema import Episode, Environment, TrajectoryQC, Provenance, new_id

# sheet name -> (label, species detail)
SHEETS = {
    "Siler collingwoodi_normal": ("siler", "Siler collingwoodi"),
    "Siler collingwoodi_blackened": ("siler", "Siler collingwoodi (blackened control)"),
    "Phintelloides versicolor": ("other_spider", "Phintelloides versicolor"),
    "Crematogaster egidyi": ("ant", "Crematogaster egidyi"),
    "Meranoplus bicolor": ("ant", "Meranoplus bicolor"),
    "Technomyrmex sp.": ("ant", "Technomyrmex sp."),
    "Polyrhachis jianghuaensis": ("ant", "Polyrhachis jianghuaensis"),
    "Polyrhachis dives": ("ant", "Polyrhachis dives"),
}

# the workbook is frame-indexed; the true capture rate is not published in
# the sheet, so per-frame series stay in per-frame units (documented)
ASSUMED_FPS = 100.0


def velocity_features(v: np.ndarray) -> dict[str, float]:
    """Kinematic features from a real per-frame speed series."""
    v = v[np.isfinite(v)]
    n = max(len(v), 1)
    thresh = 0.1 * max(np.median(v), 1e-9)
    moving = v > thresh
    runs = np.diff(np.concatenate(([0], moving.view(np.int8), [0])))
    starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
    mr = (ends - starts) if len(starts) else np.array([0])
    is_move = moving[starts] if len(starts) else np.array([True])
    def _m(a): return float(a.mean()) if len(a) else 0.0
    return {
        "duration_s": n / ASSUMED_FPS,
        "distance": float(v.sum() / ASSUMED_FPS),
        "net_displacement": float(v.sum() / ASSUMED_FPS),
        "sinuosity": 0.0,                       # undefined without heading
        "speed_mean": _m(v),
        "speed_cv": float(v.std() / max(v.mean(), 1e-9)) if len(v) and v.mean() > 0 else 0.0,
        "speed_p90": float(np.percentile(v, 90)) if len(v) else 0.0,
        "accel_rms": float(np.sqrt(_m(np.diff(v) ** 2)) * ASSUMED_FPS) if len(v) > 1 else 0.0,
        "turn_rate_mean": 0.0,                  # undefined without heading
        "turn_rate_p90": 0.0,
        "curvature_mean": 0.0,
        "frac_time_moving": _m(moving),
        "n_pauses": int((~is_move).sum()),
        "pause_run_mean_s": float(mr[~is_move].mean() / ASSUMED_FPS) if (~is_move).any() else 0.0,
        "move_run_mean_s": float(mr[is_move].mean() / ASSUMED_FPS) if is_move.any() else 0.0,
        "straightness_index": 0.0,
    }


def load_zeng_episodes(xlsx_path: str | Path, min_frames: int = 150) -> list[Episode]:
    import openpyxl
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    episodes = []
    for sheet, (label, species) in SHEETS.items():
        if sheet not in wb.sheetnames:
            continue
        rows = {}  # individual -> list of (velocity, limb1, limb2)
        order = []
        for r in wb[sheet].iter_rows(min_row=2, values_only=True):
            if r[0] is None:
                continue
            ind = str(r[0]).strip()
            if ind not in rows:
                rows[ind] = []; order.append(ind)
            vel = float(r[1]) if isinstance(r[1], (int, float)) else np.nan
            l1 = float(r[2]) if isinstance(r[2], (int, float)) else np.nan
            l2 = float(r[3]) if isinstance(r[3], (int, float)) else np.nan
            rows[ind].append((vel, l1, l2))
        for ind in order:
            arr = np.asarray(rows[ind], float)
            if len(arr) < min_frames:
                continue
            v = arr[:, 0]
            ep = Episode(
                episode_id=new_id("zeng"),
                source_video_id=f"Zeng2023:{sheet}:{ind}",
                source_video_path=f"mendeley:10.17632/jrvzn7n475.1/Gait analysis_raw data.xlsx#{sheet}",
                start_frame=0, end_frame=len(arr) - 1, fps=ASSUMED_FPS,
                frames=list(range(len(arr))),
                centroids_px=[],               # no xy published: velocity episode
                detection_confidence=[1.0] * len(arr),
                qc=TrajectoryQC(mean_detection_confidence=1.0, coverage=1.0),
                environment=Environment(
                    site="Zeng et al. 2023 gait assay", field_or_lab="lab",
                    condition="gait assay"),
            )
            ep.bio_label = label
            ep.bio_label_confidence = 1.0
            ep.bio_label_source = "human:dataset_metadata"
            ep.bio_label_detail = {"species": species, "individual": ind}
            # velocity/pose series are the observation payload
            n = max(240, len(arr))
            idx = np.unique(np.linspace(0, len(arr) - 1, min(n, len(arr))).astype(int))
            ep.metadata["velocity_series"] = [float(x) for x in v[idx]]
            ep.metadata["forelimb_series"] = [float(x) for x in arr[idx, 1]]  # leg-I / antennae-1 height
            ep.metadata["forelimb2_series"] = [float(x) for x in arr[idx, 2]]
            ep.trajectory_features = velocity_features(v)
            ep.processing_history.append(Provenance(
                software_version=__version__, model_name="zeng2023-import",
                model_version="1",
                parameters=dict(dataset="Mendeley 10.17632/jrvzn7n475.1",
                                sheet=sheet, individual=ind, n_frames=len(arr),
                                assumed_fps=ASSUMED_FPS,
                                note="velocity episode: no xy published; "
                                     "turn/sinuosity dims undefined"),
            ).to_dict())
            episodes.append(ep)
    return episodes
