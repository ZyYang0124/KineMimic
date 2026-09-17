"""MOTIONSCAPE — The Atlas of Animal Movement.

Builds the explorable atlas: a self-contained directory with ``index.html``
and per-episode animated GIF clips straight from the source video.

Atlas contract (what every visual channel means):

- particle position  : the episode's position in behavioral space, animated
                       along its sliding-window embedding path z_1..z_t
                       (a real movement drifts through movement states)
- species coloring   : hidden on entry; "Reveal species" interpolates colors
                       in over ~2 s -- the signature reveal moment
- flutter amplitude  : movement intermittency (speed_cv)
- click a particle   : original video clip + live trajectory trace +
                       speed/pause sparkline + most-similar movements
                       (nearest episodes in standardized feature space)
- motif exploration  : motifs discovered by k-means; selecting one fades the
                       rest of the flock and lists representative episodes
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .features import trajectory_features, feature_matrix
from .schema import Episode, Provenance

WINDOW_S = 3.0
STRIDE_S = 1.0
N_NEIGHBORS = 4
CLIP_MAX_S = 8.0
CLIP_W = 360


def extract_clip(ep: Episode, out_gif: str) -> str:
    """Cut the episode's frame range from the source video into a looping GIF."""
    from PIL import Image
    cap = cv2.VideoCapture(ep.source_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {ep.source_video_path}")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = CLIP_W / vw
    hw, hh = CLIP_W, max(1, int(round(vh * scale)))
    n = min(ep.end_frame - ep.start_frame + 1, int(CLIP_MAX_S * ep.fps))
    step = max(1, int(round(ep.fps / 10)))  # sample ~10 fps for the GIF
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, ep.start_frame)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (hw, hh))
        # faint trajectory overlay on the clip (data, not decoration)
        pts = np.asarray(ep.centroids_px) * scale
        for p in pts[max(0, len(pts) * i // max(n, 1) - 40): len(pts) * (i + 1) // max(n, 1)]:
            cv2.circle(g, (int(round(p[0])), int(round(p[1]))), 1, 200, -1)
        frames.append(Image.fromarray(g))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames extracted for {ep.episode_id}")
    dur_ms = int(1000 * step / ep.fps)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=dur_ms, loop=0)
    return out_gif


def _speed_series(ep: Episode, n: int = 240) -> list[float]:
    xy = np.asarray(ep.centroids_px, float)
    v = np.hypot(*np.diff(xy, axis=0).T) * ep.fps
    if len(v) == 0:
        return [0.0]
    idx = np.unique(np.linspace(0, len(v) - 1, min(n, len(v))).astype(int))
    return [float(x) for x in v[idx]]


def _window_embedding_path(ep: Episode, mu, sd, comp) -> list[list[float]]:
    """Sliding-window kinematics projected with the episode-level PCA --
    the episode's true drift through behavioral space."""
    xy = np.asarray(ep.centroids_px, float)
    fr = np.asarray(ep.frames, int)
    win = max(int(WINDOW_S * ep.fps), 2)
    stride = max(int(STRIDE_S * ep.fps), 1)
    path = []
    for s in range(0, max(len(xy) - win, 0) + 1, stride):
        f = trajectory_features(xy[s:s + win].tolist(), ep.fps, ep.px_per_cm,
                                fr[s:s + win].tolist())
        row = np.array([f.get(k.replace(" ", "_"), 0.0) for k in FEATURE_ORDER])
        path.append([float(x) for x in ((row - mu) / sd) @ comp[:2].T])
    if not path:
        f = ep.trajectory_features
        row = np.array([f.get(k, 0.0) for k in FEATURE_ORDER])
        z = ((row - mu) / sd) @ comp[:2].T
        path = [[float(z[0]), float(z[1])]]
    return path


# stable feature order shared by the PCA fit and window projection
FEATURE_ORDER = sorted(trajectory_features([[0, 0], [1, 0], [0, 1]], 30.0).keys())


def build_atlas(episodes: list[Episode], out_dir: str | Path,
                run_provenance: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    X, names = feature_matrix(episodes)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)

    # nearest neighbors in standardized feature space
    dist = ((Z[:, None, :] - Z[None]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argsort(dist, axis=1)[:, :N_NEIGHBORS]

    payload = []
    for i, ep in enumerate(episodes):
        gif = extract_clip(ep, str(out_dir / "clips" / f"{ep.episode_id}.gif"))
        payload.append({
            "episode_id": ep.episode_id,
            "video_id": ep.source_video_id,
            "bio_label": ep.bio_label,
            "motif": ep.motif,
            "embedding": ep.embedding or [0, 0],
            "path": _window_embedding_path(ep, mu, sd, Vt),
            "speed_cv": ep.trajectory_features.get("speed_cv", 0.0),
            "speed_series": _speed_series(ep),
            "frac_moving": ep.trajectory_features.get("frac_time_moving", 0.0),
            "n_pauses": ep.trajectory_features.get("n_pauses", 0),
            "duration_s": ep.duration_s,
            "frames": [ep.start_frame, ep.end_frame],
            "clip": f"clips/{Path(gif).name}",
            "neighbors": [
                {"episode_id": episodes[j].episode_id,
                 "bio_label": episodes[j].bio_label,
                 "clip": f"clips/{episodes[j].episode_id}.gif",
                 "dist": float(np.sqrt(dist[i, j]))}
                for j in nn_idx[i]],
            "features": {k: float(v) for k, v in ep.trajectory_features.items()},
            "provenance_chain": ep.provenance_chain(),
        })

    (out_dir / "data.json").write_text(
        json.dumps({"episodes": payload, "motionscape_version": __version__,
                    "provenance": run_provenance or {}}, ensure_ascii=False),
        encoding="utf-8")
    _write_index(out_dir)
    return out_dir / "index.html"


def _write_index(out_dir: Path) -> None:
    from .atlas_template import TEMPLATE
    (out_dir / "index.html").write_text(TEMPLATE, encoding="utf-8")
