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


def _is_velocity_ep(ep: Episode) -> bool:
    return bool(ep.metadata.get("velocity_series"))


def _series(ep: Episode, n: int = 240) -> dict:
    """Downsampled speed / turn-rate / moving time series for sparkline rows."""
    if _is_velocity_ep(ep):
        v = np.asarray(ep.metadata["velocity_series"], float)
        thresh = 0.1 * max(np.median(v), 1e-9)
        def rs(a):
            idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
            return [float(x) for x in a[idx]]
        return {"speed": rs(v), "turn": [0.0] * min(n, len(v)),
                "moving": rs((v > thresh).astype(float))}
    xy = np.asarray(ep.centroids_px, float)
    fr = np.asarray(ep.frames, int) if ep.frames else np.arange(len(xy))
    keep = np.isfinite(xy).all(axis=1)
    xy, fr = xy[keep], fr[keep]
    out = {"speed": [0.0], "turn": [0.0], "moving": [0.0]}
    if len(xy) > 2:
        same = np.diff(fr) == 1
        d = np.hypot(*np.diff(xy, axis=0).T)
        v = (d * ep.fps)[same]
        heading = np.arctan2(*np.diff(xy, axis=0).T[::-1])
        valid = same[:-1] & same[1:] if len(same) > 1 else np.zeros(0, bool)
        omega = (np.abs(np.diff(heading)) * ep.fps)[valid]
        thresh = 0.1 * np.median(v)
        def rs(a):
            if len(a) == 0:
                return [0.0]
            idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
            return [float(x) for x in a[idx]]
        out = {"speed": rs(v), "turn": rs(omega),
               "moving": rs((v > thresh).astype(float))}
    return out


def gait_replay_clip(ep: Episode, out_gif: str, color: tuple = (79, 209, 197)) -> str:
    """Velocity-episode replay: two stacked real traces growing together --
    speed (top) and foreleg-I / antennae height (bottom, normalized)."""
    from PIL import Image, ImageDraw
    v = np.asarray(ep.metadata["velocity_series"], float)
    f = np.asarray(ep.metadata["forelimb_series"], float)
    W, H = CLIP_W, 320
    vmx = np.nanmax(np.abs(v)) or 1.0
    fmin, fmax = np.nanmin(f), np.nanmax(f)
    frng = max(fmax - fmin, 1e-9)
    n_out = min(100, max(30, len(v) // 6))
    idx = np.unique(np.linspace(0, len(v) - 1, n_out).astype(int))
    frames = []
    for j, _ in enumerate(idx):
        img = Image.new("RGB", (W, H), (7, 11, 16))
        dr = ImageDraw.Draw(img)
        k = idx[:j + 1]
        X = lambda i: 30 + i / (n_out - 1) * (W - 60)
        # top panel: speed
        pts = [(X(i), 80 - v[i] / vmx * 60) for i in k]
        dr.line(pts, fill=(232, 161, 60), width=2)
        # bottom panel: forelimb-I (spiders) / antennae (ants) height
        pts2 = [(X(i), 220 - (f[i] - fmin) / frng * 80) for i in k]
        dr.line(pts2, fill=color, width=2)
        dr.text((30, 12), "speed (mm/s, real)", fill=(120, 140, 160))
        dr.text((30, 150), "foreleg-I / antennae height (real)", fill=(120, 140, 160))
        dr.line([(30, 145), (W - 30, 145)], fill=(25, 35, 48), width=1)
        frames.append(img)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 * CLIP_MAX_S / n_out), loop=0)
    return out_gif


def traj_replay_clip(ep: Episode, out_gif: str, color: tuple = (79, 209, 197)) -> str:
    """Trajectory-replay GIF for episodes without accessible source video:
    the real path grows at (loop-compressed) real time on a dark field."""
    from PIL import Image, ImageDraw
    xy = np.asarray(ep.centroids_px, float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    W, H = CLIP_W, 300
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    sc = min((W - 60) / max(x1 - x0, 1e-9), (H - 60) / max(y1 - y0, 1e-9))
    P = lambda p: (30 + (p[0] - x0) * sc, H - 30 - (p[1] - y0) * sc)
    n_out = min(100, max(24, len(xy) // 8))
    idx = np.unique(np.linspace(0, len(xy) - 1, n_out).astype(int))
    frames = []
    for j, i in enumerate(idx):
        img = Image.new("RGB", (W, H), (7, 11, 16))
        dr = ImageDraw.Draw(img)
        pts = [P(xy[k]) for k in idx[:j + 1]]
        dr.line(pts, fill=color + (200,), width=2)
        for q in pts[::max(1, len(pts) // 40)]:      # waypoint crumbs
            dr.ellipse([q[0] - 1, q[1] - 1, q[0] + 1, q[1] + 1], fill=(60, 90, 110))
        h = pts[-1]
        dr.ellipse([h[0] - 4, h[1] - 4, h[0] + 4, h[1] + 4], fill=color)
        frames.append(img)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 * CLIP_MAX_S / n_out), loop=0)
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
    from .zeng import velocity_features
    for s0 in range(0, max(len(xy) - win, 0) + 1, stride):
        if _is_velocity_ep(ep):
            v = np.asarray(ep.metadata["velocity_series"], float)
            w = max(int(win * len(v) / max(len(xy), 1)), 10)
            f = velocity_features(v[s0 * len(v) // max(len(xy), 1):
                                    (s0 + 1) * len(v) // max(len(xy), 1)])
        else:
            f = trajectory_features(xy[s0:s0 + win].tolist(), ep.fps, ep.px_per_cm,
                                    fr[s0:s0 + win].tolist())
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


def _traj_pts(ep: Episode, n: int = 200) -> list[list[float]]:
    if _is_velocity_ep(ep):
        v = np.asarray(ep.metadata["velocity_series"], float)
        idx = np.unique(np.linspace(0, len(v) - 1, min(n, len(v))).astype(int))
        return [[float(i), float(v[i])] for i in idx]   # speed-space curve
    xy = np.asarray(ep.centroids_px, float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return []
    idx = np.unique(np.linspace(0, len(xy) - 1, min(n, len(xy))).astype(int))
    return [[float(a), float(b)] for a, b in xy[idx]]


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
        gif_name = f"{ep.episode_id}.gif"
        gif_path = out_dir / "clips" / gif_name
        has_video = (not ep.source_video_path.startswith("dryad:")
                     and Path(ep.source_video_path).exists())
        try:
            if has_video:
                extract_clip(ep, str(gif_path))
                clip_kind = "video"
            elif _is_velocity_ep(ep):
                gait_replay_clip(ep, str(gif_path))
                clip_kind = "gait_replay"
            else:
                traj_replay_clip(ep, str(gif_path))
                clip_kind = "traj_replay"
        except Exception:
            gait_replay_clip(ep, str(gif_path))
            clip_kind = "gait_replay"
        s = _series(ep)
        hour = None
        if ep.environment.time:
            try:
                import datetime as _dt
                t = _dt.datetime.strptime(ep.environment.time.strip(), "%I:%M %p")
                hour = t.hour + t.minute / 60.0
            except ValueError:
                pass
        payload.append({
            "episode_id": ep.episode_id,
            "video_id": ep.source_video_id,
            "bio_label": ep.bio_label,
            "species_detail": ep.bio_label_detail.get("species", ""),
            "motif": ep.motif,
            "embedding": ep.embedding or [0, 0],
            "path": _window_embedding_path(ep, mu, sd, Vt),
            "speed_cv": ep.trajectory_features.get("speed_cv", 0.0),
            "series": s,
            "frac_moving": ep.trajectory_features.get("frac_time_moving", 0.0),
            "n_pauses": ep.trajectory_features.get("n_pauses", 0),
            "duration_s": ep.duration_s,
            "frames": [ep.start_frame, ep.end_frame],
            "hour": hour,
            "date": ep.environment.date,
            "clip_kind": clip_kind,
            "clip": f"clips/{gif_name}",
            "trajectory": _traj_pts(ep),
            "neighbors": [
                {"episode_id": episodes[j].episode_id,
                 "bio_label": episodes[j].bio_label,
                 "clip": f"clips/{episodes[j].episode_id}.gif",
                 "dist": float(np.sqrt(dist[i, j])),
                 "similarity": float(np.exp(-np.sqrt(dist[i, j]) / 3))}
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
