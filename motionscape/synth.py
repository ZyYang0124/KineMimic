"""Synthetic demo data: field-style videos of ants and Siler.

Generates top-down videos with two movement archetypes chosen to mirror
the biology being studied:

- **Ants**: smooth, persistent, moderately sinuous walking paths.
- **Siler** (ant-mimicking jumping spiders): intermittent locomotion --
  short bursts of forward motion punctuated by pauses and abrupt turns,
  plus occasional large jumps.

The generator is a stand-in for real field footage so the full pipeline
(and its visualizations) can be exercised end-to-end without data.
"""

from __future__ import annotations

import numpy as np

# movement archetype parameters (per-frame body length ~ px)
ARCHETYPES = {
    "ant":   dict(burst_prob=0.0,  cruise_px=3.0, pause_speed=0.4, turn_sigma=0.25, jump_prob=0.0,   jump_px=0.0,  sinuo_freq=0.05),
    "siler": dict(burst_prob=0.75, cruise_px=6.0, pause_speed=0.2, turn_sigma=0.9,  jump_prob=0.002, jump_px=45.0, sinuo_freq=0.12),
}


def _walk(n: int, start: np.ndarray, rng: np.random.Generator,
          p: dict, w: int, h: int, margin: float = 30.0) -> np.ndarray:
    """Simulate one animal's centroid path for n frames; None rows = out of view."""
    path = np.full((n, 2), np.nan)
    pos = start.astype(float)
    heading = rng.uniform(0, 2 * np.pi)
    paused = False
    for t in range(n):
        if not (margin < pos[0] < w - margin and margin < pos[1] < h - margin):
            # animal has left the scene; stay out for a while, maybe re-enter
            if rng.random() < 0.03:
                pos = np.array([rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)])
                path[t] = np.nan  # re-entry frame stays blank so no teleport step is recorded
                continue
        heading += rng.normal(0, p["turn_sigma"]) + p["sinuo_freq"] * np.sin(t * p["sinuo_freq"])
        if rng.random() < p["burst_prob"]:
            paused = not paused  # siler: intermittent start/stop
        speed = p["pause_speed"] if paused else p["cruise_px"]
        if rng.random() < p["jump_prob"]:
            pos = pos + np.array([np.cos(heading), np.sin(heading)]) * p["jump_px"]
        else:
            pos = pos + np.array([np.cos(heading), np.sin(heading)]) * speed
        # reflect off walls so animals glide along/away instead of sticking
        for ax in (0, 1):
            lo, hi = margin, (w if ax == 0 else h) - margin
            if pos[ax] <= lo or pos[ax] >= hi:
                pos[ax] = pos[ax].clip(lo, hi)
                heading = np.pi - heading if ax == 0 else -heading
        path[t] = pos
    return path


def generate_paths(n_frames: int, w: int, h: int, archetype: str,
                   n_animals: int = 3, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    p = ARCHETYPES[archetype]
    starts = rng.uniform(60, min(w, h) - 60, size=(n_animals, 2))
    return [_walk(n_frames, s, rng, p, w, h) for s in starts]


def render_video(paths: list[np.ndarray], archetype: str, out_path: str,
                 w: int = 960, h: int = 540, fps: int = 30,
                 grain: float = 8.0, seed: int = 0) -> str:
    """Render paths to an .avi with textured background + sensor noise."""
    import cv2
    rng = np.random.default_rng(seed + 1)
    n = paths[0].shape[0]
    # substrate: static textured background (so background subtraction works)
    bg = rng.normal(0.5, 0.12, size=(h, w)).clip(0, 1)
    bg = cv2.GaussianBlur((bg * 255).astype(np.uint8), (0, 0), 3)
    # size/shape: ants elongate small, spiders rounder and larger
    body = (16, 3) if archetype == "ant" else (12, 9)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out_path}")
    for t in range(n):
        frame = bg.copy()
        for path in paths:
            x, y = path[t]
            if np.isnan(x):
                continue
            # body long axis follows the true heading of the path
            t0 = t - 1 if t and np.isfinite(path[t - 1]).all() else t
            if not np.isfinite(path[t0]).all() or (path[t] == path[t0]).all():
                angle = 0.0
            else:
                angle = float(np.degrees(np.arctan2(path[t][1] - path[t0][1],
                                                    path[t][0] - path[t0][0])))
            cv2.ellipse(frame, (int(round(x)), int(round(y))), body,
                        angle, 0, 360, 0, -1)  # dark body on gray substrate
        noise = rng.normal(0, grain, size=(h, w))
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)  # grayscale writer takes the single-channel frame
    writer.release()
    return out_path
