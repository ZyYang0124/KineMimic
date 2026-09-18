"""Field-card hard-negative regression (roadmap §26).

The intro segment of the field footage once produced 83 fake episodes
(noise chained into continuous tracks). Gate: any vision pipeline must
output ZERO episodes on this scene. Skips silently when the benchmark
clip is absent (e.g. fresh clone without data/).
"""

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

BENCH = Path(__file__).resolve().parents[1] / "data" / "benchmark"
CLIP = BENCH / "clips" / "fieldcard_hardneg.mp4"


def _run_legacy_pipeline(clip: Path) -> int:
    """Legacy ingest path, minimal duration: returns number of episodes."""
    from kinemimic.detect import BgDiffDetector
    from kinemimic.track import GreedyTracker, tracks_to_episodes

    det = BgDiffDetector()
    bg = det.fit_background(str(clip))
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = GreedyTracker()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracker.step(idx, det.detect_frame(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bg))
        idx += 1
    cap.release()
    episodes = tracks_to_episodes(tracker.tracks, clip.stem, str(clip), fps,
                                  min_duration_s=2.0)
    return len(episodes)


@pytest.mark.skipif(not CLIP.exists(), reason="benchmark clip not present")
def test_fieldcard_hardneg_zero_episodes():
    """Noise from the field-card segment must not chain into episodes."""
    assert _run_legacy_pipeline(CLIP) == 0


@pytest.mark.skipif(not CLIP.exists(), reason="benchmark clip not present")
def test_fieldcard_noise_is_detected_as_noise():
    """Sanity: the scene really does trigger dense false detections
    (if this fails, the clip changed and the gate above is vacuous)."""
    from kinemimic.detect import BgDiffDetector

    det = BgDiffDetector()
    bg = det.fit_background(str(CLIP))
    cap = cv2.VideoCapture(str(CLIP))
    ok, frame = cap.read()
    cap.release()
    assert ok
    dets = det.detect_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), bg)
    assert len(dets) > 50, "field-card scene no longer produces noise?!"


def test_benchmark_splits_valid():
    """splits.json exists and holdout is disjoint from development."""
    import json
    splits = json.loads((BENCH / "splits.json").read_text(encoding="utf-8"))
    dev, hold = splits["development"], splits["holdout"]
    assert not set(dev) & set(hold)
