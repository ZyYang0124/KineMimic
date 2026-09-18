"""High-resolution tiled inference (§7-8).

Tiny ants on a 4K frame die when the frame is resized to detector input
size. Tiled inference slices each frame into overlapping tiles, runs the
detector per tile, shifts coordinates back to full-frame pixels, and
merges cross-tile duplicates with NMS.

Tile parameters (tile_w, tile_h, overlap, merge_iou, conf) are
configuration, recorded in provenance — never hard-coded.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .detections import Detection, nms


def tile_windows(frame_w: int, frame_h: int, tile_w: int, tile_h: int,
                 overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tile windows as (x, y, w, h), covering the whole frame."""
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))
    windows = []
    ys = list(range(0, max(frame_h - tile_h, 0) + 1, step_y)) or [0]
    xs = list(range(0, max(frame_w - tile_w, 0) + 1, step_x)) or [0]
    if ys[-1] + tile_h < frame_h:
        ys.append(max(0, frame_h - tile_h))
    if xs[-1] + tile_w < frame_w:
        xs.append(max(0, frame_w - tile_w))
    for y in ys:
        for x in xs:
            windows.append((x, y, min(tile_w, frame_w - x), min(tile_h, frame_h - y)))
    return windows


def tiled_detect(frame: np.ndarray,
                 detect_fn: Callable[[np.ndarray, int], list[Detection]],
                 tile_w: int = 1024, tile_h: int = 1024, overlap: float = 0.2,
                 merge_iou: float = 0.45) -> list[Detection]:
    """Run `detect_fn(tile, tile_id)` per overlapping tile, translate
    detections to full-frame coordinates, and merge cross-tile duplicates
    with NMS so one animal on a tile boundary never becomes two."""
    h, w = frame.shape[:2]
    all_dets: list[Detection] = []
    for tile_id, (x, y, tw, th) in enumerate(
            tile_windows(w, h, tile_w, tile_h, overlap)):
        tile = frame[y:y + th, x:x + tw]
        for d in detect_fn(tile, tile_id):
            dx1, dy1, dx2, dy2 = d.bbox
            d.bbox = (dx1 + x, dy1 + y, dx2 + x, dy2 + y)
            d.centroid = (d.centroid[0] + x, d.centroid[1] + y)
            d.tile_id = tile_id
            all_dets.append(d)
    return nms(all_dets, iou_threshold=merge_iou)
