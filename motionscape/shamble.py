"""Import Shamble et al. 2017 (Dryad doi:10.5061/dryad.fd612) into episodes.

The Dryad dataset contains 234 usable overhead-arena trials with full
millimeter-scale centroid tracking (``Rcm_mm``), species labels, original
GoPro file names and frame ranges -- a perfect episode source with
end-to-end provenance and no detection step required.

Species mapping (dataset labels -> MOTIONSCAPE labels):
    Myrmarachne formicaria -> mimic (ant-mimicking jumping spider)
    Salticus senicus       -> other_spider (non-mimetic control)
    ANT *                  -> ant
    1.3cm ball             -> skipped (calibration object)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import __version__
from .schema import Episode, Environment, TrajectoryQC, Provenance, new_id

FPS = 120.0  # GoPro overhead arena recordings


def _label(species: str) -> str | None:
    s = str(species)
    if s.startswith("ANT"):
        return "ant"
    if "Myrmarachne" in s:
        return "mimic"          # ant-mimicking jumping spider
    if "Salticus" in s:
        return "other_spider"   # non-mimetic salticid control
    return None                 # ball / empty -> skip


def load_shamble_episodes(mat_path: str | Path,
                          video_dir: str | Path | None = None,
                          min_frames: int = 360) -> list[Episode]:
    import scipy.io as sio
    m = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    trials = m["data"]
    video_dir = Path(video_dir) if video_dir else None
    episodes = []
    for tr in trials:
        lbl = _label(getattr(tr, "species_name", ""))
        R = getattr(tr, "Rcm_mm", None)
        if lbl is None or not hasattr(R, "shape") or R.size < 2 * min_frames // 120:
            continue
        xy = np.atleast_2d(np.asarray(R, float))
        n = len(xy)
        good = np.isfinite(xy).all(axis=1)
        if good.sum() < min_frames // 3:
            continue
        f0 = int(getattr(tr, "data_first_frame", -1) or -1)
        f1 = int(getattr(tr, "data_last_frame", -1) or -1)
        if f1 <= f0:  # frame range unknown: number sequentially from 0
            f0, f1 = 0, n - 1
        video = None
        if video_dir is not None:
            # local example files add suffixes (_ns) and mixed case extensions
            stem = Path(str(getattr(tr, "file_name", ""))).stem.upper()
            if stem:
                for cand in video_dir.glob("*.mp4") if video_dir.exists() else []:
                    if Path(cand).stem.upper().startswith(stem):
                        video = str(cand)
                        break
        ep = Episode(
            episode_id=new_id("sh"),
            source_video_id=str(getattr(tr, "file_name", "unknown")),
            source_video_path=video or f"dryad:10.5061/dryad.fd612/{getattr(tr, 'file_name', '')}",
            start_frame=f0, end_frame=f1, fps=FPS,
            frames=list(range(n)),
            centroids_px=xy.tolist(),           # already in cm-scale field units
            detection_confidence=[0.99] * n,
            # sampling hierarchy: one lab dataset, sessions by date, one video per trial
            site_id="shamble2017_dryad",
            session_id=f"shamble2017_dryad/{str(getattr(tr, 'date', '')) or 'unknown_date'}",
            qc=TrajectoryQC(mean_detection_confidence=0.99,
                            coverage=float(good.mean())),
            environment=Environment(
                site="Dryad fd612 trial", field_or_lab="lab",
                fps=FPS, condition="75cm arena, overhead GoPro",
                date=str(getattr(tr, "date", "")) or None,
                time=str(getattr(tr, "time", "")).strip() or None),
        )
        # publisher/dataset species metadata is an independent (gold) annotation
        ep.human_label = lbl
        ep.human_confidence = 1.0
        ep.human_source = "dataset:shamble2017_metadata"
        ep.annotator = "dataset_metadata"
        ep.annotation_status = "accepted"
        ep.bio_label = lbl
        ep.bio_label_confidence = 1.0
        ep.bio_label_source = "human:dataset_metadata"
        ep.bio_label_detail = {"species": str(getattr(tr, "species_name", ""))}
        ep.processing_history.append(Provenance(
            software_version=__version__, model_name="shamble2017-import",
            model_version="1",
            parameters=dict(doi="10.5061/dryad.fd612",
                            species=str(getattr(tr, "species_name", ""))),
        ).to_dict())
        episodes.append(ep)
    from .roles import RoleRegistry
    RoleRegistry().apply_to_episodes(episodes)
    return episodes
