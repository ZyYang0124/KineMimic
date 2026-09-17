"""V1 biological label heuristic + mimicry comparison read-outs.

The classifier separates Siler / ant / other spider / other arthropod
from body geometry + intermittency -- deliberately transparent, with a
confidence score. Labels are annotations, not observations: they can be
superseded by later model runs or human review without reprocessing.

``mimicry_fingerprint`` computes the Siler-vs-ant overlap per behavior
dimension (the Behavioral Mimicry Fingerprint of the project brief).
"""

from __future__ import annotations

import numpy as np

from . import __version__
from .features import trajectory_features
from .schema import Episode, Provenance

MODEL_NAME = "heuristic-v1"


def classify_episode(ep: Episode) -> tuple[str, float]:
    """Transparent heuristic from body aspect + move-pause structure.

    Ants: elongate body (aspect ~2-4), continuous locomotion.
    Siler: rounder body, intermittent (high speed_cv, many pauses).
    Confidence is heuristic distance from the decision boundary.
    """
    w = np.median([b[0] for b in ep.bbox_sizes_px]) if ep.bbox_sizes_px else 0
    h = np.median([b[1] for b in ep.bbox_sizes_px]) if ep.bbox_sizes_px else 0
    aspect = w / max(h, 1e-9)
    f = trajectory_features(ep.centroids_px, ep.fps, ep.px_per_cm, ep.frames)
    intermittent = f["speed_cv"] + f["n_pauses"] / max(f["duration_s"], 1e-9)
    elongate = np.log(max(aspect, 1e-9))
    score = elongate - 0.35 * intermittent  # ants high, siler low
    if score > 0.6:
        return "ant", min(0.5 + 0.5 * (score - 0.6), 0.95)
    if score < 0.4:
        return "siler", min(0.5 + 0.3 * (0.4 - score), 0.9)
    return "unknown", 0.3


def label_episodes(episodes: list[Episode]) -> list[Episode]:
    for ep in episodes:
        label, conf = classify_episode(ep)
        ep.bio_label, ep.bio_label_confidence = label, conf
        ep.bio_label_source = f"model:{MODEL_NAME}"
        ep.processing_history.append(Provenance(
            software_version=__version__, model_name=MODEL_NAME, model_version="1",
            parameters=dict(aspect_note="median bbox w/h, intermittency=speed_cv+pauses/s"),
            parent_ids=[ep.episode_id]).to_dict())
    return episodes


# ---------------- mimicry read-outs ----------------

def overlap(a: np.ndarray, b: np.ndarray, grid: np.ndarray) -> float:
    """Bhattacharyya coefficient between two 1-D histograms."""
    ha, _ = np.histogram(a, bins=grid, density=True)
    hb, _ = np.histogram(b, bins=grid, density=True)
    return float(np.sqrt(np.maximum(ha * hb, 0)).sum() * (grid[1] - grid[0]))


def mimicry_fingerprint(siler: list[Episode], ants: list[Episode],
                        others: list[Episode] | None = None) -> dict:
    """Per-dimension Siler-vs-Ant similarity (0-1). Each dimension uses the
    matching kinematics features; trajectory shape uses the embedding."""
    def vals(eps, key):
        out = []
        for ep in eps:
            if not ep.trajectory_features:
                ep.trajectory_features = trajectory_features(ep.centroids_px, ep.fps, ep.px_per_cm, ep.frames)
            v = ep.trajectory_features.get(key) if key != "embedding_x" else (ep.embedding[0] if ep.embedding else None)
            if key == "embedding_x" and v is None:
                continue
            out.append(v)
        return np.asarray(out, float)

    dims = {
        "speed_dynamics": "speed_mean",
        "speed_intermittency": "speed_cv",
        "stop_go_rhythm": "frac_time_moving",
        "turning": "turn_rate_mean",
        "path_shape": "sinuosity",
        "trajectory_space": "embedding_x",
    }
    fp = {}
    for dim, key in dims.items():
        a, b = vals(siler, key), vals(ants, key)
        if len(a) < 3 or len(b) < 3:
            continue
        # degenerate dims (constant, e.g. undefined for velocity episodes) carry no signal
        if np.ptp(a) < 1e-12 and np.ptp(b) < 1e-12:
            continue
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max()) + 1e-9
        fp[dim] = overlap(a, b, np.linspace(lo, hi, 25))
    prov = Provenance(software_version=__version__, model_name="bhattacharyya-v1",
                      model_version="1",
                      parameters=dict(n_siler=len(siler), n_ant=len(ants),
                                      n_other=len(others or [])))
    return {"fingerprint": fp, "provenance": prov.to_dict()}
