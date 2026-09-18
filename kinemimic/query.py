"""Find Similar: the query pipeline.

    Drop in a video. See what moves like it.

    video -> ingest (detection/tracking/episodes) -> query QC
          -> encode (Representation A + B) -> reference index search
          -> episode / motif / taxa retrieval + OOD flag
          -> query bundle (query.json) -> The Murmur

The query is a first-class citizen of the atlas: its particle positions
come from the atlas's own saved projection (never invented), neighbors
are real reference episodes, and every result records the reference
index version and encoder provenance it used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .encoder import KinematicPCAEncoder, ShapeSeriesEncoder, BaseEncoder
from .pipeline import ingest_video
from .retrieval import (ReferenceIndex, IndexPolicy, retrieve,
                        similarity_breakdown)
from .schema import Episode, Provenance
from .store import EpisodeStore


def load_projection(atlas_dir: str | Path) -> dict | None:
    p = Path(atlas_dir) / "atlas_projection.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def project_to_atlas(feat: dict, projection: dict | None) -> list[float]:
    """Project one episode's features into the atlas's exact 2-D space.

    Uses the atlas build's own standardization + PCA components, so the
    query particle lands where the data says it belongs."""
    if not projection:
        return [0.0, 0.0]
    row = np.array([feat.get(k, 0.0) for k in projection["feature_order"]], float)
    mu = np.asarray(projection["mu"])
    sd = np.asarray(projection["sd"])
    comp = np.asarray(projection["components"])
    z = ((row - mu) / sd) @ comp.T
    return [round(float(z[0]), 4), round(float(z[1]), 4)]


def qc_screen(episodes: list[Episode], min_duration_s: float = 3.0) -> list[dict]:
    """Query QC (§32): poor tracking or too-short episodes are flagged,
    never silently used for high-confidence claims."""
    out = []
    for ep in episodes:
        cov = getattr(ep.qc, "coverage", 1.0)
        warnings = []
        if ep.duration_s < min_duration_s:
            warnings.append("short episode")
        if cov < 0.7:
            warnings.append("low tracking coverage")
        if not ep.trajectory_features or ep.trajectory_features.get("distance", 0) < 1e-6:
            warnings.append("no measurable motion")
        out.append({"episode_id": ep.episode_id, "duration_s": round(ep.duration_s, 2),
                    "coverage": round(float(cov), 3), "warnings": warnings,
                    "usable": len(warnings) == 0})
    return out


def run_query(video_path: str, query_id: str, reference_dir: str | Path,
              store_dir: str | Path | None = None, out_dir: str | Path | None = None,
              atlas_dir: str | Path | None = None, k: int = 6,
              metric: str = "euclidean", target_fps: float = 30.0,
              min_duration_s: float = 3.0, use_representation_b: bool = True,
              detector_params: dict | None = None,
              tracker_max_gap: int = 3,
              vision_mode: str = "legacy",
              vision_detector: str = "legacy") -> dict:
    """Full Find Similar pipeline for one uploaded video."""
    t0 = time.time()
    ref_dir = Path(reference_dir)
    index = ReferenceIndex.load(ref_dir)
    enc = KinematicPCAEncoder.load(ref_dir, index.encoder_name)
    enc_b = None
    try:
        enc_b = ShapeSeriesEncoder.load(ref_dir, "shape-series")
    except Exception:
        enc_b = None
    if not use_representation_b:
        enc_b = None

    # 1-3. ingest: detection -> tracking -> episodes (existing pipeline)
    store = EpisodeStore(store_dir) if store_dir else None
    if vision_mode != "legacy":
        from .vision.pipeline import VisionConfig, run_vision_frontend, VisionQCFailed
        from .vision.tracker import TrackerConfig
        cfg = VisionConfig(mode=vision_mode, detector=vision_detector,
                           detector_params=detector_params or {},
                           tracker_cfg=TrackerConfig(
                               max_gap_frames=max(tracker_max_gap, 3),
                               source_video_id=query_id),
                           min_duration_s=min_duration_s)
        try:
            frontend = run_vision_frontend(video_path, query_id, cfg)
        except VisionQCFailed as e:
            return {"query_id": query_id, "error": "vision quality gate rejected "
                    "this video", "reason": str(e),
                    "provenance": {"reference_atlas_version": index.version_id}}
        episodes = frontend["episodes"]
    else:
        episodes = ingest_video(video_path, store, query_id,
                                min_duration_s=min_duration_s, target_fps=target_fps,
                                detector_params=detector_params,
                                tracker_max_gap=tracker_max_gap)
    if not episodes:
        return {"query_id": query_id, "error": "no movement episodes detected",
                "provenance": {"reference_atlas_version": index.version_id}}
    # features must exist before QC (the motion check reads them)
    for e in episodes:
        if not e.trajectory_features:
            from .features import trajectory_features
            e.trajectory_features = trajectory_features(
                e.centroids_px, e.fps, e.px_per_cm, e.frames)
    qc = qc_screen(episodes, min_duration_s=min_duration_s)
    usable = [e for e, q in zip(episodes, qc) if q["usable"]] or episodes

    # 4. encode with the reference encoder(s)
    Z = enc.transform(usable)
    Z_b = enc_b.transform(usable) if enc_b else None

    # 5. retrieve at all three levels (primary representation)
    result = retrieve(usable, Z, index, enc, k=k, metric=metric)
    result_b = retrieve(usable, Z_b, index, enc_b, k=k, metric=metric) \
        if (enc_b and Z_b is not None) else None

    # representation B: its own OOD read-out + full taxa, and the A/B
    # neighbor-agreement check (§15) computed over episodes that are inside
    # BOTH hulls (comparing garbage against garbage proves nothing)
    consistency = None
    if result_b:
        from .retrieval import ood_flag as _ood
        result_b["ood"] = _ood(Z_b, index, metric)
        inside = [i for i in range(len(usable))
                  if not result["ood"]["per_episode_flagged"][i]
                  and not result_b["ood"]["per_episode_flagged"][i]]
        j = []
        for i in inside:
            a = {n["episode_id"] for n in result["episode_hits"][i]["neighbors"]}
            b = {n["episode_id"] for n in result_b["episode_hits"][i]["neighbors"]}
            j.append(len(a & b) / max(len(a | b), 1))
        consistency = {"n_compared": len(inside),
                       "jaccard_at_k": [round(x, 3) for x in j],
                       "mean_jaccard": round(float(np.mean(j)), 3) if j else None,
                       "note": "neighbor agreement between the physical "
                               "feature space and the shape-normalized space, "
                               "over episodes inside both hulls"}
        if taxa_b := result_b.get("taxa"):
            result_b["taxa"] = taxa_b

    # 6. attach similarity breakdowns (real standardized contributions)
    feat_sd = getattr(index, "feature_sd", None) or {}
    ref_feats = getattr(index, "features_by_id", {}) or {}
    for hit in result["episode_hits"]:
        top = hit["neighbors"][0]["episode_id"] if hit["neighbors"] else None
        qe = next((e for e in usable if e.episode_id == hit["query_episode_id"]), None)
        if top and top in ref_feats and qe:
            hit["breakdown"] = similarity_breakdown(
                qe.trajectory_features, ref_feats[top], feat_sd)

    # 7. atlas projection for the query particles (real positions)
    projection = load_projection(atlas_dir) if atlas_dir else None
    for ep, q in zip(episodes, qc):
        q["atlas_position"] = project_to_atlas(ep.trajectory_features, projection)

    # 8. bundle
    bundle = {
        "query_id": query_id,
        "video_path": str(video_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_atlas_version": index.version_id,
        "metric": metric, "k": k,
        "qc": qc,
        "episodes": [{
            "episode_id": e.episode_id, "duration_s": round(e.duration_s, 2),
            "fps": e.fps, "frames": [e.start_frame, e.end_frame],
            "trajectory": [[round(float(a), 2) for a in c]
                           for c in (e.centroids_px or [])][:400],
            "series": _query_series(e),
            "atlas_position": project_to_atlas(e.trajectory_features, projection),
            "features": {k2: round(float(v), 5)
                         for k2, v in e.trajectory_features.items()},
        } for e in episodes],
        "episode_dicts": [e.to_dict() for e in episodes],
        "results": result,
        "results_representation_b": result_b,
        "consistency": consistency,
        "build_seconds": round(time.time() - t0, 1),
        "provenance": result["provenance"],
        "pipeline_provenance": Provenance(
            software_version=__version__, model_name="find-similar-v1",
            model_version="1",
            parameters=dict(video=str(video_path), query_id=query_id,
                            target_fps=target_fps,
                            min_duration_s=min_duration_s,
                            detector_params=detector_params or {},
                            n_query_episodes=len(episodes))).to_dict(),
    }
    bundle["encoder_provenance"] = index.encoder_provenance
    written = []
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "query.json").write_text(
            json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        written.append(str(out_dir / "query.json"))
    if atlas_dir:
        a = Path(atlas_dir)
        a.mkdir(parents=True, exist_ok=True)
        (a / "query.json").write_text(
            json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
        written.append(str(a / "query.json"))
    bundle["written"] = written
    return bundle


def _query_series(ep: Episode, n: int = 160) -> dict:
    from .atlas import _series
    return _series(ep, n)


def build_reference(episodes: list[Episode], out_dir: str | Path,
                    dim: int = 10, policy: IndexPolicy | None = None,
                    previous_version: str | None = None,
                    use_representation_b: bool = True,
                    min_episodes: int = 10) -> dict:
    """Fit encoders on reference episodes and write a versioned index."""
    enc = KinematicPCAEncoder(dim=dim).fit(episodes)
    encoders = {"kinematic-pca": enc}
    enc.save(out_dir)
    if use_representation_b:
        enc_b = ShapeSeriesEncoder(dim=dim).fit(episodes)
        enc_b.save(out_dir)
        encoders["shape-series"] = enc_b
    idx = ReferenceIndex.build(episodes, enc, policy=policy,
                               previous_version=previous_version,
                               out_dir=out_dir, min_episodes=min_episodes)
    # attach per-episode features + feature sd for similarity breakdowns
    from .features import feature_matrix
    kept = [e for e in episodes if
            (policy or IndexPolicy()).accepts(e)]
    X, names = feature_matrix(kept)
    sd = {k: float(v) for k, v in zip(names, X.std(0) + 1e-9)}
    idx.feature_sd = sd
    idx.features_by_id = {e.episode_id: {k: round(float(v), 5) for k, v in
                                         e.trajectory_features.items()}
                          for e in kept}
    # re-save with the attached features
    d = Path(out_dir) / idx.version_id
    state = json.loads((d / "index.json").read_text(encoding="utf-8"))
    state["feature_sd"] = sd
    state["features_by_id"] = idx.features_by_id
    (d / "index.json").write_text(json.dumps(state, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    # the index is self-contained: kept episodes allow later evaluation
    # (leave-video-out, augmentation tests) without the original run
    with open(d / "episodes.jsonl", "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e.to_dict()) + "\n")
    return {"index": idx, "encoders": encoders,
            "n_kept": len(kept), "n_input": len(episodes)}
