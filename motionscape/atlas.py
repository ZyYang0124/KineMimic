"""MOTIONSCAPE — The Murmur: build the explorable Movement Atlas.

Scale contract (V0.2):

- ``data.json`` holds only what the flock needs: embedding, window path,
  motif, intermittency, minimal metadata. ~1 KB per episode.
- Per-episode detail (time series, trajectory, features, provenance,
  neighbors) lives in ``meta/<episode_id>.json`` and is fetched only when
  an episode is selected (lazy media loading).
- Source-video / replay GIFs are generated on demand by
  ``python -m motionscape serve`` (``media.make_clip``) — never pre-rendered
  for the whole dataset. A static copy of the atlas still works: clips fall
  back to in-browser trajectory replay.
- Nearest neighbors are computed in the ORIGINAL standardized feature space
  (k-d tree above ~800 episodes) — never in 2-D screen space.

Blind-space contract: the embedding and motifs come from kinematics only.
Labels are stored alongside for the Reveal-species overlay; they play no
role in how the space is constructed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .features import trajectory_features, feature_matrix
from .hierarchy import hierarchy_summary
from .media import clip_kind
from .schema import Episode, Provenance

WINDOW_S = 3.0
STRIDE_S = 1.0
N_NEIGHBORS = 8
MAX_PATH_PTS = 64          # murmur drift path points per episode
MAX_TRAJ_PTS = 200         # per-episode meta trajectory points
MAX_SERIES_PTS = 160       # sparkline resolution
KD_TREE_MIN_N = 800

# stable feature order shared by the PCA fit and window projection
FEATURE_ORDER = sorted(trajectory_features([[0, 0], [1, 0], [0, 1]], 30.0).keys())


def _window_embedding_path(ep: Episode, mu, sd, comp) -> list[list[float]]:
    """Sliding-window kinematics projected with the episode-level PCA —
    the episode's true drift through behavioral space. Episodes whose
    temporal data cannot support it get a single point (static position):
    no fabricated motion."""
    from .zeng import velocity_features

    def project(f: dict) -> list[float]:
        row = np.array([f.get(k, 0.0) for k in FEATURE_ORDER])
        z = ((row - mu) / sd) @ comp[:2].T
        return [float(z[0]), float(z[1])]

    path: list[list[float]] = []
    vel = ep.metadata.get("velocity_series")
    if vel and not ep.centroids_px:
        v = np.asarray(vel, float)
        n_win = max(int(WINDOW_S * 100.0), 10)        # zeng frames ~100 fps
        stride = max(int(STRIDE_S * 100.0), 5)
        for s0 in range(0, max(len(v) - n_win, 0) + 1, stride):
            path.append(project(velocity_features(v[s0:s0 + n_win])))
    else:
        xy = np.asarray(ep.centroids_px, float)
        fr = np.asarray(ep.frames, int) if len(ep.frames) == len(xy) else np.arange(len(xy))
        win = max(int(WINDOW_S * ep.fps), 2)
        stride = max(int(STRIDE_S * ep.fps), 1)
        for s0 in range(0, max(len(xy) - win, 0) + 1, stride):
            seg, segfr = xy[s0:s0 + win], fr[s0:s0 + win]
            if len(seg) < 10:
                continue
            path.append(project(trajectory_features(seg.tolist(), ep.fps, ep.px_per_cm, segfr.tolist())))
    if not path:
        path = [project(ep.trajectory_features)]
    # downsample to MAX_PATH_PTS
    if len(path) > MAX_PATH_PTS:
        idx = np.unique(np.linspace(0, len(path) - 1, MAX_PATH_PTS).astype(int))
        path = [path[i] for i in idx]
    return path


def _series(ep: Episode, n: int = MAX_SERIES_PTS) -> dict:
    """Downsampled speed / turn-rate / moving time series for sparklines."""
    def rs(a):
        if len(a) == 0:
            return [0.0]
        idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
        return [round(float(x), 4) for x in a[idx]]

    vel = ep.metadata.get("velocity_series")
    if vel and not ep.centroids_px:
        v = np.asarray(vel, float)
        thresh = 0.1 * max(np.median(v), 1e-9)
        return {"speed": rs(v), "turn": [0.0] * len(rs(v)),
                "moving": rs((v > thresh).astype(float))}
    xy = np.asarray(ep.centroids_px, float)
    fr = np.asarray(ep.frames, int) if ep.frames else np.arange(len(xy))
    keep = np.isfinite(xy).all(axis=1)
    xy, fr = xy[keep], fr[keep]
    out = {"speed": [0.0], "turn": [0.0], "moving": [0.0]}
    if len(xy) > 2:
        # honor frame spacing (high-fps footage analyzed with frame skipping)
        steps = np.diff(fr)
        step = int(np.bincount(steps[steps > 0]).argmax()) if (steps > 0).any() else 1
        dt = step / max(ep.fps, 1)
        same = steps == step
        d = np.hypot(*np.diff(xy, axis=0).T)
        v = (d / dt)[same]
        heading = np.arctan2(*np.diff(xy, axis=0).T[::-1])
        valid = same[:-1] & same[1:] if len(same) > 1 else np.zeros(0, bool)
        omega = (np.abs(np.diff(heading)) / dt)[valid]
        thresh = 0.1 * np.median(v) if len(v) else 1e-9
        out = {"speed": rs(v), "turn": rs(omega),
               "moving": rs((v > thresh).astype(float))}
    return out


def _traj_pts(ep: Episode, n: int = MAX_TRAJ_PTS) -> list[list[float]]:
    vel = ep.metadata.get("velocity_series")
    if vel and not ep.centroids_px:
        v = np.asarray(vel, float)
        idx = np.unique(np.linspace(0, len(v) - 1, min(n, len(v))).astype(int))
        return [[float(i), float(v[i])] for i in idx]   # speed-space curve
    xy = np.asarray(ep.centroids_px, float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return []
    idx = np.unique(np.linspace(0, len(xy) - 1, min(n, len(xy))).astype(int))
    return [[round(float(a), 2), round(float(b), 2)] for a, b in xy[idx]]


def nearest_neighbors(X: np.ndarray, k: int = N_NEIGHBORS) -> tuple[np.ndarray, np.ndarray]:
    """Indices + distances of the k nearest episodes in standardized feature
    space. k-d tree (scipy) once N is large; brute force below that (exact
    either way — the tree is for speed, not approximation)."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = len(Z)
    k = min(k, n - 1)
    if n >= KD_TREE_MIN_N:
        from scipy.spatial import cKDTree
        tree = cKDTree(Z)
        d, idx = tree.query(Z, k=k + 1)
        return idx[:, 1:], d[:, 1:]
    d = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx, np.take_along_axis(d, idx, axis=1)


def episode_meta(ep: Episode, neighbor_ids: list[tuple[int, float]] | None = None,
                 labels: list[str] | None = None,
                 episodes: list[Episode] | None = None,
                 interaction_ctx: dict | None = None) -> dict:
    """Lazy-loaded per-episode detail (served as meta/<id>.json)."""
    hour = None
    if ep.environment.time:
        try:
            import datetime as _dt
            t = _dt.datetime.strptime(ep.environment.time.strip(), "%I:%M %p")
            hour = t.hour + t.minute / 60.0
        except ValueError:
            pass
    meta = {
        "episode_id": ep.episode_id,
        "video_id": ep.source_video_id,
        "video_path": ep.source_video_path,
        "clip_kind": clip_kind(ep),
        "label": ep.effective_label(),
        "label_source": ("human" if ep.human_label else
                         ("machine" if ep.machine_label else "unannotated")),
        "annotator": ep.annotator,
        "species_detail": ep.bio_label_detail.get("species", ""),
        "annotation_status": ep.annotation_status,
        "motif": ep.motif,
        "duration_s": ep.duration_s,
        "fps": ep.fps,
        "frames": [ep.start_frame, ep.end_frame],
        "hour": hour,
        "date": ep.environment.date,
        "sampling": {"site": ep.site_id or ep.environment.site,
                     "session": ep.session_id or ep.environment.date,
                     "video": ep.source_video_id},
        "series": _series(ep),
        "trajectory": _traj_pts(ep),
        "features": {k: round(float(v), 5) for k, v in ep.trajectory_features.items()},
        "provenance_chain": ep.provenance_chain(),
        "neighbors": [
            {"episode_id": nid, "label": (labels[ni] if labels else None),
             "dist": round(float(nd), 4),
             "similarity": round(float(np.exp(-nd / 3)), 4)}
            for nid, ni, nd in (neighbor_ids or [])
        ],
    }
    if interaction_ctx is not None:
        meta["context"] = {k: interaction_ctx.get(k) for k in
                           ("has_ant", "frac_frames_with_ant", "n_ants_mean",
                            "n_ants_max", "nearest_ant_dist_min",
                            "nearest_ant_dist_mean", "ant_activity_mean", "units",
                            "radius", "label")}
    if episodes is not None:
        scene = _scene_block(ep, episodes)
        if scene:
            meta["scene"] = scene
    return meta


def _scene_block(ep: Episode, episodes: list[Episode]) -> dict | None:
    """Full-scene data for Interaction playback: concurrent episodes'
    trajectories in this video + nearest-ant distance / count series.

    Everything here is recorded data from the same source video — the
    scene an episode actually lived in."""
    from .interaction import _xy_by_frame, _speeds_by_frame, circular_mean_deg
    same_video = [e for e in episodes
                  if e.source_video_id == ep.source_video_id
                  and e.episode_id != ep.episode_id
                  and e.metadata.get("velocity_series") is None
                  and not ep.metadata.get("velocity_series")]
    if not same_video:
        return None
    xa = _xy_by_frame(ep)
    f0, f1 = ep.start_frame, ep.end_frame
    nbrs = []
    for o in same_video:
        xo = _xy_by_frame(o)
        common = sorted(f for f in xo if f0 <= f <= f1 and f in xa)
        if len(common) < 10:
            continue
        idx = np.unique(np.linspace(0, len(common) - 1,
                                    min(160, len(common))).astype(int))
        fr = [common[i] for i in idx]
        nbrs.append({
            "episode_id": o.episode_id, "label": o.effective_label(),
            "status": o.annotation_status,
            "frames": fr,
            "xs": [round(xo[f][0], 2) for f in fr],
            "ys": [round(xo[f][1], 2) for f in fr],
        })
    if not nbrs:
        return None
    scale = ep.px_per_cm
    units = "cm" if scale else "px"
    sc = (1.0 / scale) if scale else 1.0
    # nearest-ant distance + ant count per focal frame (aligned, downsampled)
    ants = [n for n in nbrs if n["label"] == "ant"]
    frames = sorted(set(xa) & set(range(f0, f1 + 1)))
    d_ant, n_ant, head_ant, t_s = [], [], [], []
    xys = [dict(zip(n["frames"], zip(n["xs"], n["ys"]))) for n in ants]
    step = max(1, (f1 - f0) // 160)
    fps = ep.fps or 30.0
    from .interaction import _headings_by_frame
    ant_heads = [_headings_by_frame(e) for e in same_video
                 if e.effective_label() == "ant"]
    for f in frames[::max(1, step)]:
        x, y = xa[f]
        dists = [np.hypot(xy[f][0] - x, xy[f][1] - y) * sc
                 for xy in xys if f in xy]
        d_ant.append(round(float(min(dists)), 2) if dists else None)
        n_ant.append(len(dists))
        heads = [np.degrees(h[f]) for h in ant_heads if f in h]
        head_ant.append(circular_mean_deg(heads) if heads else None)
        t_s.append(round((f - f0) / fps, 3))
    return {"units": units,
            "n_neighbors": len(nbrs),
            "neighbors": nbrs,
            "t_s": t_s, "nearest_ant_dist": d_ant,
            "n_ants_within": n_ant, "mean_ant_heading_deg": head_ant}


def build_atlas(episodes: list[Episode], out_dir: str | Path,
                run_provenance: dict | None = None,
                summary: dict | None = None,
                interaction: dict | None = None,
                make_meta_files: bool = True) -> Path:
    """Build the atlas directory. Fast enough for tens of thousands of
    episodes: no video decoding, no GIF rendering, one slim JSON + small
    per-episode meta files. ``interaction`` = parsed
    interaction_summary.json (from ``motionscape interact``), enabling
    Interaction Mode context on every episode."""
    out_dir = Path(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, names = feature_matrix(episodes)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    evr = (S ** 2 / (S ** 2).sum())[:2]
    nn_idx, nn_dist = nearest_neighbors(X)

    # motifs are machine discoveries; if this episode set hasn't been through
    # analyze(), discover them here (unsupervised, label-blind as ever)
    if any(ep.motif is None for ep in episodes):
        from .motifs import find_motifs
        labels_km, _, _ = find_motifs(X, k=min(8, max(len(episodes), 1)))
        for ep, m in zip(episodes, labels_km):
            ep.motif = int(m)

    labels = [ep.effective_label() for ep in episodes]
    # biological roles (model / mimic / controls) — metadata overlay for the
    # comparative framework; never used by encoders or the embedding
    from .roles import RoleRegistry, ROLE_NAMES
    registry = RoleRegistry.load(out_dir / "roles.json")
    registry.save(out_dir / "roles.json")       # persist effective registry
    registry.apply_to_episodes(episodes)
    label_role = {}
    for lbl in sorted(set(labels)):
        rep = next(e for e in episodes if e.effective_label() == lbl)
        taxon = (rep.bio_label_detail or {}).get("species") or lbl
        label_role[lbl] = registry.role_for(taxon)
    roles_meta = {lbl: {"role": role, "name": ROLE_NAMES[role]}
                  for lbl, role in label_role.items()}
    ia = (interaction or {}).get("contexts", {})
    slim, meta_jobs = [], []
    for i, ep in enumerate(episodes):
        ctx = ia.get(ep.episode_id)
        slim.append({
            "id": ep.episode_id,
            "l": labels[i],
            "st": ep.annotation_status,
            "m": ep.motif,
            "e": [round(float(x), 4) for x in (ep.embedding or [0.0, 0.0])],
            "p": [[round(a, 4) for a in pt] for pt in
                  _window_embedding_path(ep, mu, sd, Vt)],
            "cv": round(float(ep.trajectory_features.get("speed_cv", 0.0)), 4),
            "d": round(ep.duration_s, 2),
            "fm": round(float(ep.trajectory_features.get("frac_time_moving", 0.0)), 4),
            "v": ep.source_video_id,
            "ck": clip_kind(ep),
            "nn": [[int(j), round(float(nn_dist[i, r]), 3)]
                   for r, j in enumerate(nn_idx[i])],
            "r": label_role[labels[i]],
            "ia": ({"ha": int(bool(ctx.get("has_ant"))),
                    "fw": ctx.get("frac_frames_with_ant"),
                    "nm": ctx.get("n_ants_mean"),
                    "dmin": ctx.get("nearest_ant_dist_min"),
                    "u": ctx.get("units")} if ctx else None),
        })
        meta_jobs.append((ep, [(episodes[int(j)].episode_id, int(j), nn_dist[i, r])
                               for r, j in enumerate(nn_idx[i])], ctx))

    if make_meta_files:
        for ep, nbrs, ctx in meta_jobs:
            (out_dir / "meta" / f"{ep.episode_id}.json").write_text(
                json.dumps(episode_meta(ep, nbrs, labels, episodes=episodes,
                                        interaction_ctx=ctx),
                           ensure_ascii=False),
                encoding="utf-8")

    # hero episode: a real trajectory for the intro (longest reviewed one)
    hero_src = max(episodes, key=lambda e: (e.is_reviewed(), min(e.duration_s, 20.0)))
    hero_traj = _traj_pts(hero_src, 300)

    meta = {
        "motionscape_version": __version__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 2),
        "n_episodes": len(episodes),
        "features": names,
        "embedding": {"model": "pca-v1", "blind_to_labels": True,
                      "explained_variance_ratio": [round(float(v), 4) for v in evr],
                      "note": "behavioral space was constructed from movement "
                              "features only; species labels are overlays"},
        "hierarchy": hierarchy_summary(episodes),
        "labels_summary": {l: labels.count(l) for l in set(labels)},
        "fingerprint": (summary or {}).get("mimicry_fingerprint", {}),
        "annotation": (summary or {}).get("annotation", {}),
        "interaction": _interaction_meta(interaction),
        "roles": roles_meta,
        "motif_annotations": _load_motif_annotations(out_dir),
        "provenance": run_provenance or (summary or {}).get("provenance", {}),
        "question": "How does a spider move like an ant?",
        "hero": {"episode_id": hero_src.episode_id, "trajectory": hero_traj,
                 "duration_s": hero_src.duration_s},
    }
    (out_dir / "data.json").write_text(
        json.dumps({"meta": meta, "episodes": slim}, ensure_ascii=False),
        encoding="utf-8")
    # persist the atlas projection: query particles must land at their real
    # position in THIS space (motionscape find-similar uses it)
    (out_dir / "atlas_projection.json").write_text(json.dumps({
        "feature_order": names, "mu": mu.tolist(), "sd": sd.tolist(),
        "components": Vt[:2].tolist(),
        "explained_variance_ratio": [round(float(v), 4) for v in evr],
        "blind_to_labels": True,
    }), encoding="utf-8")
    _write_index(out_dir)
    return out_dir / "index.html"


def _interaction_meta(interaction: dict | None) -> dict:
    """Interaction Layer block for data.json meta (comparison + params)."""
    if not interaction:
        return {"available": False}
    cmp_ = interaction.get("comparison", {})
    return {"available": True,
            "n_pairs": interaction.get("n_records", 0),
            "n_scene_windows": interaction.get("n_scene_windows", 0),
            "n_velocity_excluded": interaction.get("n_velocity_excluded", 0),
            "comparison": cmp_,
            "provenance": interaction.get("provenance", {}),
            "note": "proximity is not interaction; all contrasts here are "
                    "descriptive and tested against an episode-shuffle null "
                    "(predictive association, not causation)"}


def _load_motif_annotations(out_dir: Path) -> dict:
    path = out_dir / "motif_annotations.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_index(out_dir: Path) -> None:
    from .atlas_template import TEMPLATE
    (out_dir / "index.html").write_text(TEMPLATE, encoding="utf-8")
