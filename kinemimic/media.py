"""Episode media: animated GIF clips, generated on demand and cached.

Three kinds of clip, in order of preference:

- ``video``      : the true source-video frames (cropped to the animal's
                   arena when a crop box is given) -- the strongest evidence;
- ``gait_replay`` : for velocity episodes (no xy published): two stacked real
                   traces -- speed and foreleg-I / antennae height;
- ``traj_replay`` : the real path growing at real relative time.

Nothing here is decoration: every pixel traces back to recorded data. Clip
files are produced lazily (the atlas never pre-renders thousands of GIFs).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schema import Episode

CLIP_MAX_S = 8.0


def has_source_video(ep: Episode) -> bool:
    return (not ep.source_video_path.startswith(("dryad:", "mendeley:"))
            and Path(ep.source_video_path).exists())


def clip_kind(ep: Episode) -> str:
    if has_source_video(ep):
        return "video"
    if ep.metadata.get("velocity_series"):
        return "gait_replay"
    return "traj_replay"


def video_clip_gif(ep: Episode, out_path: str, width: int = 360,
                   crop_margin_px: int | None = None) -> str:
    """Cut the episode's frame range from the source video into a GIF.

    With ``crop_margin_px`` the frame is cropped around the trajectory's
    bounding box (workbench zoom); otherwise the full frame is downscaled.
    """
    from PIL import Image
    cap = cv2.VideoCapture(ep.source_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {ep.source_video_path}")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xy = np.asarray([c for c in ep.centroids_px if c], float)
    box = None
    if crop_margin_px and len(xy):
        x0, y0 = np.floor(xy.min(0) - crop_margin_px).astype(int)
        x1, y1 = np.ceil(xy.max(0) + crop_margin_px).astype(int)
        side = max(x1 - x0, y1 - y0)                     # square crop
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = side // 2
        box = (max(0, cx - half), max(0, cy - half),
               min(vw, cx + half), min(vh, cy + half))
        if box[2] - box[0] < 40 or box[3] - box[1] < 40:
            box = None
    src = box or (0, 0, vw, vh)
    scale = width / max(src[2] - src[0], 1)
    hw, hh = width, max(1, int(round((src[3] - src[1]) * scale)))
    xy_full = np.asarray(ep.centroids_px, float)
    frames = []
    n_total = max(ep.end_frame - ep.start_frame + 1, 1)
    n_take = min(n_total, int(CLIP_MAX_S * ep.fps))
    step = max(1, int(round(ep.fps / 10.0)))          # ~10 fps GIF
    cap.set(cv2.CAP_PROP_POS_FRAMES, ep.start_frame)
    for i in range(n_take):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            continue
        g0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = g0[src[1]:src[3], src[0]:src[2]]
        g = cv2.resize(g, (hw, hh))
        # faint real-trajectory overlay, revealed up to the current frame
        if len(xy_full):
            upto = max(2, int(len(xy_full) * i / n_total))
            for p in xy_full[max(0, upto - 60):upto]:
                px = (p[0] - src[0]) * scale
                py = (p[1] - src[1]) * scale
                if 0 <= px < hw and 0 <= py < hh:
                    cv2.circle(g, (int(px), int(py)), 1, 200, -1)
        frames.append(Image.fromarray(g))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames extracted for {ep.episode_id}")
    dur_ms = int(1000 * step / max(ep.fps, 1))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=dur_ms, loop=0)
    return out_path


def gait_replay_gif(ep: Episode, out_path: str, color: tuple = (79, 209, 197)) -> str:
    """Velocity-episode replay: real speed (top) + foreleg-I/antennae (bottom)."""
    from PIL import Image, ImageDraw
    v = np.asarray(ep.metadata["velocity_series"], float)
    f = np.asarray(ep.metadata["forelimb_series"], float)
    W, H = 360, 300
    vmx = np.nanmax(np.abs(v)) or 1.0
    fmin, fmax = np.nanmin(f), np.nanmax(f)
    frng = max(fmax - fmin, 1e-9)
    n_out = int(min(100, max(30, len(v) // 6)))
    idx = np.unique(np.linspace(0, len(v) - 1, n_out).astype(int))
    frames = []
    for j in range(len(idx)):
        img = Image.new("RGB", (W, H), (7, 11, 16))
        dr = ImageDraw.Draw(img)
        k = idx[:j + 1]
        X = lambda i: 30 + i / (n_out - 1) * (W - 60)
        dr.line([(X(i), 80 - v[i] / vmx * 60) for i in k], fill=(232, 161, 60), width=2)
        dr.line([(X(i), 215 - (f[i] - fmin) / frng * 75) for i in k], fill=color, width=2)
        dr.text((30, 12), "speed (dataset units, real)", fill=(120, 140, 160))
        dr.text((30, 145), "foreleg-I / antennae height (real)", fill=(120, 140, 160))
        dr.line([(30, 140), (W - 30, 140)], fill=(25, 35, 48), width=1)
        frames.append(img)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 * CLIP_MAX_S / n_out), loop=0)
    return out_path


def traj_replay_gif(ep: Episode, out_path: str, color: tuple = (79, 209, 197)) -> str:
    """Real path growing on a dark field (for episodes without source video)."""
    from PIL import Image, ImageDraw
    xy = np.asarray(ep.centroids_px, float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    W, H = 360, 300
    if len(xy) < 2:
        raise RuntimeError(f"no trajectory for {ep.episode_id}")
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    sc = min((W - 60) / max(x1 - x0, 1e-9), (H - 60) / max(y1 - y0, 1e-9))
    P = lambda p: (30 + (p[0] - x0) * sc, H - 30 - (p[1] - y0) * sc)
    n_out = int(min(100, max(24, len(xy) // 8)))
    idx = np.unique(np.linspace(0, len(xy) - 1, n_out).astype(int))
    frames = []
    for j in range(len(idx)):
        img = Image.new("RGB", (W, H), (7, 11, 16))
        dr = ImageDraw.Draw(img)
        pts = [P(xy[k]) for k in idx[:j + 1]]
        dr.line(pts, fill=color + (200,), width=2)
        for q in pts[::max(1, len(pts) // 40)]:
            dr.ellipse([q[0] - 1, q[1] - 1, q[0] + 1, q[1] + 1], fill=(60, 90, 110))
        h = pts[-1]
        dr.ellipse([h[0] - 4, h[1] - 4, h[0] + 4, h[1] + 4], fill=color)
        frames.append(img)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 * CLIP_MAX_S / n_out), loop=0)
    return out_path


def make_clip(ep: Episode, out_path: str, crop_margin_px: int | None = None) -> str:
    """Dispatch to the best available clip kind; returns the kind written."""
    kind = clip_kind(ep)
    if kind == "video":
        try:
            video_clip_gif(ep, out_path, crop_margin_px=crop_margin_px)
            return "video"
        except Exception:
            kind = "gait_replay" if ep.metadata.get("velocity_series") else "traj_replay"
    if kind == "gait_replay":
        gait_replay_gif(ep, out_path)
    else:
        traj_replay_gif(ep, out_path)
    return kind
