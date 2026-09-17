"""End-to-end pipeline: video -> episodes -> features -> labels -> space.

Preserves every intermediate representation as JSON inside a versioned
run directory. Re-analyzing the same video creates a new run; earlier
results are never destroyed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .classify import label_episodes, mimicry_fingerprint
from .detect import BgDiffDetector, MODEL_NAME as DET_MODEL
from .embedding import PCAEmbedder
from .features import trajectory_features, feature_matrix, MODEL_NAME as FEAT_MODEL
from .motifs import find_motifs, motif_profile
from .schema import Episode, Provenance
from .store import EpisodeStore
from .track import GreedyTracker, tracks_to_episodes


def ingest_video(video_path: str, store: EpisodeStore, video_id: str,
                 min_duration_s: float = 3.0, px_per_cm: float | None = None,
                 environment: dict | None = None) -> list[Episode]:
    """Detection + short-term tracking + episode extraction for one video."""
    det = BgDiffDetector()
    bg = det.fit_background(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = GreedyTracker()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracker.step(idx, det.detect_frame(gray, bg))
        idx += 1
    cap.release()
    episodes = tracks_to_episodes(tracker.tracks, video_id, video_path, fps,
                                  min_duration_s=min_duration_s, px_per_cm=px_per_cm)
    prov = Provenance(software_version=__version__, model_name=DET_MODEL, model_version="1",
                      parameters=dict(min_duration_s=min_duration_s, n_tracks=len(tracker.tracks)))
    for ep in episodes:
        ep.processing_history.append(prov.to_dict())
        if environment:
            for k, v in environment.items():
                if v is not None and hasattr(ep.environment, k):
                    setattr(ep.environment, k, v)
    return episodes


def analyze(store: EpisodeStore, episodes: list[Episode], n_motifs: int = 8,
            parent_run: str | None = None) -> Path:
    """Features -> heuristic labels -> embedding -> motifs -> fingerprint."""
    run_dir = store.new_run("pipeline-v1", __version__,
                            parameters=dict(n_episodes=len(episodes), n_motifs=n_motifs),
                            parent_run=parent_run)

    for ep in episodes:
        ep.trajectory_features = trajectory_features(ep.centroids_px, ep.fps, ep.px_per_cm, ep.frames)
        ep.processing_history.append(Provenance(
            software_version=__version__, model_name=FEAT_MODEL, model_version="1").to_dict())

    # external gold labels (e.g. dataset metadata) are never overwritten
    label_episodes([ep for ep in episodes if ep.bio_label_source == "unannotated"])

    X, names = feature_matrix(episodes)
    emb, prov = PCAEmbedder(n_components=2).fit_transform(X, names)
    for ep, e in zip(episodes, emb):
        ep.embedding = [float(e[0]), float(e[1])]
        ep.processing_history.append(prov.to_dict())

    labels, centers, mprov = find_motifs(X, k=min(n_motifs, max(len(episodes), 1)))
    for ep, m in zip(episodes, labels):
        ep.motif = int(m)
        ep.processing_history.append(mprov.to_dict())

    group_names = ("ant", "siler", "mimic", "other_spider", "unknown")
    groups = {g: np.array([ep.bio_label == g for ep in episodes]) for g in group_names}
    profiles = motif_profile(labels, {k: v for k, v in groups.items() if v.any()})

    fp = mimicry_fingerprint([e for e in episodes if e.bio_label in ("siler", "mimic")],
                             [e for e in episodes if e.bio_label == "ant"],
                             others=[e for e in episodes if e.bio_label == "other_spider"])

    store.write_episodes(run_dir, episodes)
    (run_dir / "summary.json").write_text(json.dumps(
        {"n_episodes": len(episodes), "n_videos": len({e.source_video_id for e in episodes}),
         "labels": {k: int(v.sum()) for k, v in groups.items()},
         "motif_profiles": {k: v.tolist() for k, v in profiles.items()},
         "mimicry_fingerprint": fp}, indent=2), encoding="utf-8")
    return run_dir
