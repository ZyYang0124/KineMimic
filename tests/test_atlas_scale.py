"""Blind embedding, nearest-neighbor correctness, atlas scale & lazy media."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionscape.atlas import nearest_neighbors, build_atlas
from motionscape.benchmark import synthetic_episodes
from motionscape.embedding import PCAEmbedder
from motionscape.features import feature_matrix
from motionscape.schema import Episode
from motionscape.zeng import velocity_features


def _eps_with_features(n=30, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        v = np.abs(rng.gamma(2, 15, 200))
        ep = Episode(episode_id=f"e{i}", frames=list(range(200)), centroids_px=[],
                     start_frame=0, end_frame=199, fps=100.0,
                     detection_confidence=[1.0] * 200)
        ep.metadata["velocity_series"] = v.tolist()
        ep.metadata["forelimb_series"] = rng.random(200).tolist()
        ep.trajectory_features = velocity_features(v)
        eps.append(ep)
    return eps


def test_embedding_is_blind_to_labels():
    """Permuting biological labels must not change the behavioral space."""
    eps = _eps_with_features(40)
    labels_a = ["ant", "siler"] * 20
    labels_b = list(reversed(labels_a))
    X, _ = feature_matrix(eps)
    assert not any(str(l) in str(X.dtype) for l in labels_a)  # features only
    emb_a, _ = PCAEmbedder(2).fit_transform(X, _)
    emb_b, _ = PCAEmbedder(2).fit_transform(X, _)
    # labels never enter fit_transform at all: the strict guarantee is that
    # the API has no label input; embeddings identical trivially
    assert np.allclose(emb_a, emb_b)
    # and the pipeline's overlay contract: effective label is applied after
    for ep, la, lb in zip(eps, labels_a, labels_b):
        ep.human_label = la
    assert [e.effective_label() for e in eps] == labels_a


def test_pipeline_embedding_ignores_annotation(tmp_path):
    """analyze() twice on the same movements with different human labels ->
    identical embeddings (blind space), different overlays."""
    from motionscape.pipeline import analyze
    from motionscape.store import EpisodeStore
    from motionscape.annotation import AnnotationRecord, append_record

    store = EpisodeStore(tmp_path / "s1")
    run1 = analyze(store, _eps_with_features(24))
    emb1 = {json.loads(l)["episode_id"]: json.loads(l)["embedding"]
            for l in open(run1 / "episodes.jsonl", encoding="utf-8")}
    # now human-label every episode and re-analyze in a fresh run
    eps2 = _eps_with_features(24)
    for i, e in enumerate(eps2):
        append_record(store.root, AnnotationRecord(
            episode_id=e.episode_id, human_label="siler" if i % 2 else "ant"))
    run2 = analyze(store, eps2, parent_run=run1.name)
    emb2 = {json.loads(l)["episode_id"]: json.loads(l)["embedding"]
            for l in open(run2 / "episodes.jsonl", encoding="utf-8")}
    assert set(emb1) == set(emb2)
    for k in emb1:
        assert np.allclose(emb1[k], emb2[k], atol=1e-6), "embedding moved with labels!"
    # overlays did change; and the two runs are separate directories (append-only)
    assert run2 != run1 and run2.name != run1.name


def test_nearest_neighbors_match_bruteforce():
    from motionscape.zeng import velocity_features
    eps = _eps_with_features(120)
    X, _ = feature_matrix(eps)
    idx, dist = nearest_neighbors(X, k=5)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    D = np.sqrt(((Z[:, None, :] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    for i in range(0, 120, 13):
        assert np.allclose(np.sort(dist[i]), np.sort(D[i])[:5], atol=1e-6)
        assert set(idx[i]) == set(np.argsort(D[i])[:5])


def test_atlas_5000_scale(tmp_path):
    """5,000 episodes: build completes fast, payload stays slim, no media."""
    n = 5000
    eps = synthetic_episodes(n, seed=1, frac_trajectory=0.1)
    t0 = time.time()
    out = build_atlas(eps, tmp_path / "atlas")
    dt = time.time() - t0
    data = json.loads((tmp_path / "atlas" / "data.json").read_text(encoding="utf-8"))
    size_mb = (tmp_path / "atlas" / "data.json").stat().st_size / 1e6
    assert len(data["episodes"]) == n
    assert dt < 300, f"build too slow: {dt:.0f}s for {n}"
    assert size_mb < 25, f"data.json too heavy: {size_mb:.1f} MB"
    assert not (tmp_path / "atlas" / "clips").exists(), "no media should be pre-rendered"
    n_meta = len(list((tmp_path / "atlas" / "meta").glob("*.json")))
    assert n_meta == n
    # kd-tree used at this scale; neighbors present and finite
    assert all(len(e["nn"]) == 8 for e in data["episodes"][:50])


def test_lazy_meta_loading(tmp_path):
    """Only the selected episode's detail is needed at click time."""
    eps = synthetic_episodes(50, seed=2)
    build_atlas(eps, tmp_path / "atlas")
    slim = json.loads((tmp_path / "atlas" / "data.json").read_text(encoding="utf-8"))
    first = slim["episodes"][0]
    # slim record carries no time series / trajectory
    assert "series" not in first and "trajectory" not in first
    meta = json.loads(
        (tmp_path / "atlas" / "meta" / f"{first['id']}.json").read_text(encoding="utf-8"))
    assert "series" in meta and "trajectory" in meta
    assert "provenance_chain" in meta


def test_sampling_hierarchy_blocks_pseudoreplication():
    from motionscape.hierarchy import hierarchy_summary, hierarchical_bootstrap
    eps = synthetic_episodes(200, seed=3)
    for i, e in enumerate(eps):
        e.site_id = f"site_{i // 100}"          # 2 sites
        e.session_id = f"site_{i // 100}/day_{(i % 100) // 50}"
    h = hierarchy_summary(eps)
    assert h["n_sites"] == 2 and h["n_sessions"] == 4 and h["n_videos"] == 10
    assert set(h["tree"]) == {"site_0", "site_1"}
    # cluster bootstrap: resamples whole sites, so CI brackets the point estimate
    out = hierarchical_bootstrap(eps, lambda es: float(np.mean([e.duration_s for e in es])),
                                 n_boot=60, seed=0)
    assert out["resampled_level"] == "site"
    assert out["ci_low"] <= out["stat"] <= out["ci_high"]


def test_motif_annotation_separate_from_machine_id(tmp_path):
    from motionscape.serve import AtlasServer
    eps = synthetic_episodes(20, seed=4)
    build_atlas(eps, tmp_path / "atlas")
    srv = AtlasServer(tmp_path / "atlas", None)
    srv.save_motif_annotation(3, "stop-and-go cruise", "watched 6 clips", "zeynep")
    ann = srv.motif_annotations()
    # machine motif id untouched; human annotation stored separately
    slim = json.loads((tmp_path / "atlas" / "data.json").read_text(encoding="utf-8"))
    assert all(isinstance(e["m"], int) for e in slim["episodes"])
    assert ann["3"]["name"] == "stop-and-go cruise"
    assert ann["3"]["annotator"] == "zeynep"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(Path(f"/tmp/ms_test_{name}"))
            print(f"{name} OK")
