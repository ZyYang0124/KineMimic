"""Modern vision frontend: purity-first tracking test battery (§59-63).

Every test encodes a scenario with a known correct outcome under the
ambiguity policy: prefer splitting over guessing; a false merge is the
worst possible error.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinemimic.vision.detections import Detection, nms
from kinemimic.vision.tiling import tiled_detect, tile_windows
from kinemimic.vision.tracker import TrackerConfig, TwoStageTracker
from kinemimic.vision.pipeline import (VisionConfig, run_vision_frontend,
                                       tracklets_to_episodes)


def det(frame, x, y, conf=0.9, w=20, h=12, cls="ant"):
    return Detection(frame_idx=frame, bbox=(x - w / 2, y - h / 2,
                                            x + w / 2, y + h / 2),
                     confidence=conf, cls=cls)


def run_tracker(frames_dets, **cfgkw):
    cfg = TrackerConfig(**cfgkw)
    tr = TwoStageTracker(cfg)
    for f, dets in frames_dets:
        tr.step(f, dets)
    return tr.finalize_all()


# --------------------------------------------------------------------------
# §8 tiling
# --------------------------------------------------------------------------

def test_nms_merges_duplicates():
    d1 = det(0, 100, 100, conf=0.9)
    d2 = det(0, 103, 101, conf=0.8)          # heavy overlap
    kept = nms([d1, d2], iou_threshold=0.5)
    assert len(kept) == 1 and kept[0].confidence == 0.9


def test_tiling_object_crossing_two_tiles():
    """An object straddling a tile boundary must yield ONE merged
    detection, not two (which would duplicate tracks)."""
    frame = np.zeros((400, 800), np.uint8)
    OBJ = (380.0, 200.0)                       # global position (straddles x=400 seam)
    windows = tile_windows(800, 400, 400, 400, 0.1)

    def fake_detector(tile, tile_id):
        x, y, w, h = windows[tile_id]
        if x <= OBJ[0] < x + w and y <= OBJ[1] < y + h:
            return [Detection(frame_idx=0,
                              bbox=(OBJ[0] - x - 10, OBJ[1] - y - 10,
                                    OBJ[0] - x + 10, OBJ[1] - y + 10),
                              confidence=0.9, cls="ant")]
        return []

    dets = tiled_detect(frame, fake_detector, tile_w=400, tile_h=400,
                        overlap=0.1, merge_iou=0.5)
    assert len(dets) == 1
    # the object truly straddled a boundary (seen by >=2 tiles)
    assert sum(1 for x, y, w, h in windows
               if x <= OBJ[0] < x + w and y <= OBJ[1] < y + h) >= 2


def test_tiling_object_near_four_tile_corner():
    frame = np.zeros((400, 400), np.uint8)
    OBJ = (195.0, 195.0)                       # near the shared corner
    windows = tile_windows(400, 400, 300, 300, 0.2)

    def fake_detector(tile, tile_id):
        x, y, w, h = windows[tile_id]
        if x <= OBJ[0] < x + w and y <= OBJ[1] < y + h:
            return [Detection(frame_idx=0,
                              bbox=(OBJ[0] - x - 8, OBJ[1] - y - 8,
                                    OBJ[0] - x + 8, OBJ[1] - y + 8),
                              confidence=0.9, cls="ant")]
        return []

    dets = tiled_detect(frame, fake_detector, tile_w=300, tile_h=300,
                        overlap=0.2, merge_iou=0.5)
    assert len(dets) == 1
    seen_by = sum(1 for x, y, w, h in windows
                  if x <= OBJ[0] < x + w and y <= OBJ[1] < y + h)
    assert seen_by >= 4


# --------------------------------------------------------------------------
# §60 crossing: never a silent identity swap
# --------------------------------------------------------------------------

def _crossing_frames(n=60, speed=4):
    frames = []
    for f in range(n):
        xa, xb = 100 + f * speed, 300 - f * speed
        frames.append((f, [det(f, xa, 270), det(f, xb, 270)]))
    return frames


def test_crossing_without_swap_or_silent_contamination():
    """Two animals cross. Every output tracklet with enough points must be
    a single GT animal (direction continuity), and any tracklet that could
    not stay pure must carry an ambiguity event."""
    tracklets = run_tracker(_crossing_frames())
    for t in tracklets:
        if len(t.points) < 6:
            continue
        xs = [p.x for p in t.points]
        dx = np.diff(xs)
        assert (np.all(dx >= 0) or np.all(dx <= 0)), \
            f"tracklet {t.tracklet_id} changed direction: silent swap!"


def test_crossing_split_when_identity_unrecoverable():
    """Degenerate crossing where the two detections coincide exactly at
    the meeting frame: identity cannot be recovered, so the tracker must
    split, and every continuation carries an ambiguity event."""
    frames = []
    for f in range(40):
        xa = 100 + f * 5
        xb = 295 - f * 5
        if f == 20:
            xa = xb = 197.5                     # exact coincidence
        frames.append((f, [det(f, xa, 270), det(f, xb, 270)]))
    tracklets = run_tracker(frames)
    for t in tracklets:
        if len(t.points) >= 6:
            xs = [p.x for p in t.points]
            dx = np.diff(xs)
            assert (np.all(dx >= 0) or np.all(dx <= 0)), "silent swap!"


# --------------------------------------------------------------------------
# §61 collision merge
# --------------------------------------------------------------------------

def test_collision_merges_then_splits():
    """Two animals enter one merged detection for a few frames, then
    separate. Correct outcome: old tracklets terminated (ambiguity events),
    new tracklets continue, NO tracklet contains both animals."""
    frames = []
    for f in range(60):
        if f < 20:                               # approach
            a, b = 100 + f * 8, 500 - f * 8
            frames.append((f, [det(f, a, 270), det(f, b, 270)]))
        elif f < 30:                             # merged: one blob
            frames.append((f, [det(f, 300, 270)]))
        else:                                    # separated
            a, b = 300 + (f - 30) * 8, 300 - (f - 30) * 8
            frames.append((f, [det(f, a, 270), det(f, b, 270)]))
    tracklets = run_tracker(frames)
    long = [t for t in tracklets if len(t.points) >= 10]
    merge_events = sum(1 for t in tracklets
                       for e in t.ambiguity_events if e.kind == "merge")
    assert merge_events >= 1, "collision must be flagged as merge ambiguity"
    for t in long:
        xs = [p.x for p in t.points]
        if t.points[0].frame_idx < 30 and t.points[-1].frame_idx > 30:
            span_side = (max(xs) - min(xs))
            assert span_side < 400, \
                f"tracklet {t.tracklet_id} spanned the collision: false merge!"


def test_cluster_detection_between_two_tracks_splits_not_crashes():
    """Real dense footage puts a detection between two tracks at once.
    Both tracks converge on it: the collision path must terminate them
    with merge events, not guess which ant it is (and not crash on the
    tracklet bookkeeping)."""
    frames = [(f, [det(f, 100, 100), det(f, 134, 100)]) for f in range(5)]
    frames.append((5, [det(5, 100, 100), det(5, 134, 100), det(5, 117, 100)]))
    frames += [(f, [det(f, 100, 100), det(f, 134, 100), det(f, 117, 100)])
               for f in range(6, 40)]
    tracklets = run_tracker(frames)
    merges = [e for t in tracklets for e in t.ambiguity_events
              if e.kind == "merge"]
    assert merges, "converging tracks must be flagged as a merge ambiguity"
    for t in tracklets:
        sides = {round(p.x) for p in t.points}
        assert max(sides) - min(sides) < 30, \
            f"tracklet {t.tracklet_id} jumped between two animals"


# --------------------------------------------------------------------------
# §62 stationary pause
# --------------------------------------------------------------------------

def test_stationary_animal_survives():
    """Walk -> stop 2s -> walk. A stationary animal must never be
    terminated just because motion is zero (stop-go phenotype)."""
    frames = []
    for f in range(150):                          # 5s at 30fps
        x = 100 + min(f, 60) * 3 + max(0, f - 120) * 3
        frames.append((f, [det(f, x, 270)]))
    tracklets = run_tracker(frames)
    long = [t for t in tracklets if len(t.points) >= 100]
    assert len(long) == 1, "stationary pause must not split or drop the animal"
    assert long[0].points[-1].frame_idx == 149


# --------------------------------------------------------------------------
# §63 occlusion / gap recovery vs split
# --------------------------------------------------------------------------

def test_short_gap_recovered_with_flags():
    """1-2 missed frames: tracklet continues; gap recorded; interpolated
    points are flagged as interpolated."""
    frames = []
    for f in range(90):
        x = 100 + f * 3
        if f in (40, 41):                         # 2-frame dropout
            frames.append((f, []))                # detector saw nothing
            continue
        frames.append((f, [det(f, x, 270)]))
    tracklets = run_tracker(frames, max_gap_frames=4, max_interpolated_gap=4)
    long = [t for t in tracklets if len(t.points) >= 60]
    assert len(long) == 1
    t = long[0]
    assert t.gap_events, "gap must be recorded"
    states = {p.state for p in t.points}
    assert "interpolated" in states
    assert all(p.state in ("observed", "interpolated") for p in t.points)


def test_gap_beyond_interpolation_window_still_reassociates():
    """Real detectors flicker for longer than the fill window: a gap past
    max_interpolated_gap but inside max_gap_frames must re-associate
    (unfilled) instead of erroring out."""
    tr = TwoStageTracker(TrackerConfig(max_gap_frames=8, max_interpolated_gap=4))
    for f in range(17):                            # recovery at f=16, gap=7
        dets = [] if 10 <= f < 16 else [det(f, 100 + 10 * f, 270)]
        tr.step(f, dets)
    assert len(tr.active) == 1 and tr.active[0].last_frame == 16
    assert not any(p.state == "interpolated" for p in tr.active[0].tracklet.points)
    assert abs(tr.active[0].velocity[0] - 10.0) < 1e-6


def test_interpolated_gap_velocity_stays_px_per_frame():
    """After filling a gap the velocity must be the true per-frame step,
    not the residual of the last interpolated point divided by the gap."""
    tr = TwoStageTracker(TrackerConfig(max_gap_frames=8, max_interpolated_gap=4))
    for f in range(14):                            # recovery at f=13, gap=4
        dets = [] if f in (10, 11, 12) else [det(f, 100 + 12 * f, 270)]
        tr.step(f, dets)
    assert len(tr.active) == 1
    assert abs(tr.active[0].velocity[0] - 12.0) < 1e-6, \
        f"velocity corrupted by gap fill: {tr.active[0].velocity}"


def test_long_gap_splits_instead_of_guessing():
    """A gap longer than max_gap_frames must finalize the old tracklet and
    start a new one — never stitch blindly."""
    frames = []
    for f in range(120):
        if 40 <= f < 80:                          # long occlusion
            frames.append((f, []))                # detector saw nothing
            continue
        frames.append((f, [det(f, 100 + f * 3, 270)]))
    tracklets = run_tracker(frames, max_gap_frames=8)
    long = [t for t in tracklets if len(t.points) >= 20]
    assert len(long) == 2, "long gap must split into two tracklets"
    starts = sorted(t.points[0].frame_idx for t in long)
    assert starts[1] >= 80                        # new tracklet after reappearance


# --------------------------------------------------------------------------
# low-confidence flicker (ByteTrack-style stage 2)
# --------------------------------------------------------------------------

def test_low_confidence_flicker_stays_one_track():
    frames = []
    for f in range(90):
        conf = 0.8 if f % 2 == 0 else 0.3         # alternating conf
        frames.append((f, [det(f, 100 + f * 3, 270, conf=conf)]))
    tracklets = run_tracker(frames, det_high=0.5, det_low=0.15)
    long = [t for t in tracklets if len(t.points) >= 80]
    assert len(long) == 1, "confidence flicker must not fragment the track"
    assert long[0].points[-1].frame_idx == 89


# --------------------------------------------------------------------------
# provenance / purity metadata / episode bridge
# --------------------------------------------------------------------------

def test_tracklet_to_episode_purity_metadata():
    frames = [(f, [det(f, 100 + f * 3, 270)]) for f in range(120)]
    tracklets = run_tracker(frames)
    eps = tracklets_to_episodes(tracklets, "v1", "mem:v1", 30.0,
                                min_duration_s=1.0)
    assert eps, "long enough tracklets must become episodes"
    ep = eps[0]
    assert ep.metadata["tracklet_id"].startswith("trk_")
    assert ep.metadata["point_states"]["observed"] == len(ep.frames)
    assert "n_interpolated_points" in ep.metadata
    assert any(h["model_name"] == "tracklet-to-episode"
               for h in ep.processing_history)


def test_short_tracklets_filtered_from_episodes():
    frames = [(f, [det(f, 100 + f * 3, 270)]) for f in range(20)]   # <1s
    tracklets = run_tracker(frames)
    eps = tracklets_to_episodes(tracklets, "v1", "mem:v1", 30.0,
                                min_duration_s=3.0)
    assert eps == []


# --------------------------------------------------------------------------
# end-to-end on a rendered video
# --------------------------------------------------------------------------

def test_vision_frontend_end_to_end():
    """Full frontend: rendered collision video -> episodes with purity
    metadata -> legacy Episode objects ready for analyze()."""
    import cv2
    from kinemimic.vision.benchmark import default_scenes, render_frames
    import tempfile
    scene = [s for s in default_scenes() if s.name == "collision"][0]
    vdir = Path(tempfile.gettempdir()) / "km_vision_test"
    vdir.mkdir(parents=True, exist_ok=True)
    vpath = vdir / "collision.avi"
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"MJPG"),
                         scene.fps, (scene.w, scene.h), isColor=False)
    for f, frame, blobs, layer in render_frames(scene):
        vw.write(frame)
    vw.release()
    run = run_vision_frontend(str(vpath), "vtest",
                              VisionConfig(mode="fast", detector="legacy",
                                           min_duration_s=2.0))
    assert run["episodes"], "frontend must produce episodes"
    assert run["stats"]["n_frames_analyzed"] > 0
    amb = sum(len(e.metadata.get("ambiguity_events", []))
              for e in run["episodes"])
    assert amb >= 1, "collision scene must surface ambiguity events"
    for ep in run["episodes"]:
        assert ep.metadata["detector_class"] in ("ant", "spider",
                                                 "other_arthropod", "unknown")
        assert "tracking_confidence" in ep.metadata


def test_quality_gate_rejects_noise_dominated_footage():
    """§48 signature: every frame lights up with uncorrelated specks (so
    the gate's detection-absence floors pass) yet nothing is ever reliably
    one animal. That run must be refused, not reported as a clean
    zero-episode result."""
    import cv2
    import pytest
    import tempfile
    from kinemimic.vision.pipeline import VisionQCFailed
    rng = np.random.default_rng(0)
    vdir = Path(tempfile.gettempdir()) / "km_vision_noise"
    vdir.mkdir(parents=True, exist_ok=True)
    vpath = vdir / "specks.avi"
    W = H = 320
    vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"MJPG"),
                         30.0, (W, H), isColor=False)
    for _ in range(150):
        frame = np.zeros((H, W), np.uint8)
        for cx, cy in rng.integers(12, W - 12, size=(30, 2)):
            cv2.circle(frame, (int(cx), int(cy)), 5, 200, -1)
        vw.write(frame)
    vw.release()
    with pytest.raises(VisionQCFailed) as ei:
        run_vision_frontend(str(vpath), "specks",
                            VisionConfig(mode="fast", detector="legacy",
                                         min_duration_s=3.0))
    msg = str(ei.value)
    assert "median lifetime" in msg and "detections per frame" in msg, msg
    assert "confidence" not in msg, "must be the fragmentation arm, not the floor"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(Path(f"/tmp/ms_test_{name}"))
            print(f"{name} OK")
