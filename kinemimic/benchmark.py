"""Atlas scale benchmark: build an atlas from N synthetic episodes and
report timing, payload sizes, and (optionally) in-browser frame rate.

    python -m kinemimic benchmark --n 5000 10000 20000 --out benchmarks

Episodes are synthetic *velocity-style* observations (real feature
pipeline, no video). Output is written to benchmarks/benchmark_<n>.json —
numbers recorded there back the scale claims in docs/ATLAS.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .atlas import build_atlas, nearest_neighbors, FEATURE_ORDER
from .features import feature_matrix
from .schema import Episode
from .zeng import velocity_features


def synthetic_episodes(n: int, seed: int = 0,
                       frac_trajectory: float = 0.25) -> list[Episode]:
    """Two movement archetypes (steady walkers / stop-and-go) so the
    feature space is not degenerate. A fraction carry real xy trajectories
    to exercise the sliding-window path projection."""
    rng = np.random.default_rng(seed)
    eps: list[Episode] = []
    for i in range(n):
        intermittent = rng.random() < 0.5
        t = np.arange(240)
        if intermittent:
            v = rng.gamma(1.2, 12.0, len(t)) * (rng.random(len(t)) < 0.35)
        else:
            v = rng.gamma(4.0, 8.0, len(t))
        ep = Episode(episode_id=f"bench_{n}_{i}", source_video_id=f"benchvid_{i // 20}",
                     source_video_path=f"benchmark:local", start_frame=0,
                     end_frame=len(t) - 1, fps=100.0,
                     frames=t.tolist(), centroids_px=[],
                     detection_confidence=[1.0] * len(t))
        ep.metadata["velocity_series"] = [round(float(x), 3) for x in v]
        ep.metadata["forelimb_series"] = [round(float(x), 3) for x in rng.random(len(t))]
        ep.trajectory_features = velocity_features(v)
        if rng.random() < frac_trajectory:
            xy = np.cumsum(rng.normal(0, 2.0, (240, 2)), axis=0) + 200
            ep.centroids_px = xy.round(2).tolist()
            ep.trajectory_features = velocity_features(v)   # keep keys aligned
        eps.append(ep)
    return eps


def run(n_values: list[int], out_dir: str | Path = "benchmarks",
        seed: int = 0) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in n_values:
        t_gen = time.time()
        eps = synthetic_episodes(n, seed=seed)
        t_gen = time.time() - t_gen

        t_feat = time.time()
        X, _ = feature_matrix(eps)
        t_feat = time.time() - t_feat

        t_nn = time.time()
        nn_idx, nn_dist = nearest_neighbors(X)
        t_nn = time.time() - t_nn

        atlas_dir = out / f"atlas_{n}"
        t_build = time.time()
        build_atlas(eps, atlas_dir, summary=None, make_meta_files=True)
        t_build = time.time() - t_build

        data_size = (atlas_dir / "data.json").stat().st_size
        n_meta = len(list((atlas_dir / "meta").glob("*.json")))
        rec = {"n_episodes": n, "gen_s": round(t_gen, 2), "features_s": round(t_feat, 2),
               "nearest_neighbors_s": round(t_nn, 3), "atlas_build_s": round(t_build, 2),
               "data_json_mb": round(data_size / 1e6, 2), "meta_files": n_meta,
               "kb_per_episode": round(data_size / max(n, 1) / 1024, 2)}
        path = out / f"benchmark_{n}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)
        print(f"  n={n:>6}: build {rec['atlas_build_s']:>6.1f}s · nn {rec['nearest_neighbors_s']:>5.2f}s · "
              f"data.json {rec['data_json_mb']:>6.1f} MB ({rec['kb_per_episode']} KB/ep) · "
              f"{rec['meta_files']} meta files")
    return results


if __name__ == "__main__":
    run([1000, 5000, 10000, 20000])
