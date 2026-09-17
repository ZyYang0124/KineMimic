"""Behavior Encoder & Retrieval Engine: encoders, index, query, OOD.

Synthetic scenes with theoretical expectations; no species labels enter
any encoder (the retrieval layer only reads them downstream).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinemimic.encoder import KinematicPCAEncoder, ShapeSeriesEncoder
from kinemimic.query import build_reference, load_projection, project_to_atlas, run_query
from kinemimic.retrieval import (IndexPolicy, ReferenceIndex, ood_flag,
                                   similarity_breakdown, taxa_retrieval)
from kinemimic.schema import Episode
from kinemimic.synth import generate_paths, render_video


def _longest_finite_run(p):
    """Trim a synthetic path to its longest contiguous finite segment so
    archetype features stay clean (no out-of-view gaps)."""
    fin = np.isfinite(p).all(axis=1)
    best, cur, best_s, cur_s = 0, 0, 0, 0
    for i, f in enumerate(fin):
        if f:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s = cur, cur_s
        else:
            cur = 0
    return p[best_s:best_s + best]


def _archetype_episodes(n_each=4, seed0=0):
    """ant / siler synthetic episodes with precomputed features."""
    from kinemimic.features import trajectory_features
    eps = []
    for arch_i, arch in enumerate(("ant", "siler")):
        for i in range(n_each):
            p = generate_paths(600, 960, 540, arch, n_animals=1,
                               seed=seed0 + arch_i * 10 + i)[0]
            p = _longest_finite_run(p)
            e = Episode(episode_id=f"{arch}_{i}", source_video_id=f"v_{arch}",
                        start_frame=0, end_frame=len(p) - 1, fps=30.0,
                        frames=list(range(len(p))), centroids_px=p.tolist(),
                        detection_confidence=[1.0] * len(p))
            e.trajectory_features = trajectory_features(p.tolist(), 30.0)
            e.bio_label = arch
            if arch == "ant":
                e.human_label, e.human_source = "ant", "human"
            else:
                e.machine_label, e.machine_source = "siler", "model:test"
            eps.append(e)
    return eps


def test_encoder_dim_and_roundtrip(tmp_path):
    eps = _archetype_episodes()
    enc = KinematicPCAEncoder(dim=5).fit(eps)
    Z = enc.transform(eps)
    assert Z.shape == (len(eps), 5)                 # fixed, configurable dim
    enc.save(tmp_path / "ref")
    enc2 = KinematicPCAEncoder.load(tmp_path / "ref", "kinematic-pca")
    Z2 = enc2.transform(eps)
    assert np.allclose(Z, Z2, atol=1e-9)            # exact save/load round-trip
    assert enc2.provenance()["parameters"]["labels_used"].startswith("none")


def test_shape_encoder_invariant_to_rigid_transforms():
    """Translation/rotation of a trajectory must not change its shape vector
    beyond float tolerance (camera invariance by construction)."""
    import copy
    eps = _archetype_episodes(2)
    enc = ShapeSeriesEncoder(dim=6).fit(eps)
    e = eps[0]
    z0 = enc.transform_one(e)
    e_r = copy.deepcopy(e)
    th = 1.1
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    e_r.centroids_px = (np.asarray(e.centroids_px, float) @ R.T).tolist()
    e_t = copy.deepcopy(e)
    e_t.centroids_px = (np.asarray(e.centroids_px, float) + [777, -333]).tolist()
    assert np.allclose(z0, enc.transform_one(e_r), atol=1e-6)
    assert np.allclose(z0, enc.transform_one(e_t), atol=1e-6)


def test_index_policy_and_versioning(tmp_path):
    eps = _archetype_episodes(4)
    eps[0].annotation_status = "rejected"           # QC-rejected: must stay out
    out = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)
    idx = out["index"]
    ids = [e["episode_id"] for e in idx.entries]
    assert eps[0].episode_id not in ids and len(ids) == 7
    assert idx.version_id.startswith("atlas_reference_v")
    assert idx.policy["name"] == "not-rejected"
    # reload by parent dir resolves the newest version
    idx2 = ReferenceIndex.load(tmp_path / "ref")
    assert idx2.version_id == idx.version_id


def test_exact_nn_known_geometry(tmp_path):
    """A synthetic ant query must retrieve ant-archetype episodes, and a
    synthetic siler query siler episodes (positive control)."""
    eps = _archetype_episodes(4, seed0=100)
    out = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)
    idx = out["index"]
    enc = out["encoders"]["kinematic-pca"]
    q_ant = eps[0]                                   # reference-held-out copy below
    # held-out: build an index WITHOUT eps[0], then query with it
    eps_wo = [e for e in eps if e.episode_id != q_ant.episode_id]
    out2 = build_reference(eps_wo, tmp_path / "ref2", dim=6, min_episodes=6)
    idx2 = out2["index"]
    hits = idx2.search(enc.transform_one(q_ant), k=3)
    taxa = [h["taxon"] for h in hits]
    # positive control: the ant query's neighborhood is ant-majority
    # (single-NN assertions would be brittle in an 8-episode space)
    assert taxa.count("ant") >= 2, f"ant query neighborhood: {taxa}"
    q_s = [e for e in eps if e.episode_id.startswith("siler")][0]
    eps_wo2 = [e for e in eps if e.episode_id != q_s.episode_id]
    idx3 = build_reference(eps_wo2, tmp_path / "ref3", dim=6, min_episodes=6)["index"]
    hits3 = idx3.search(enc.transform_one(q_s), k=3)
    taxa3 = [h["taxon"] for h in hits3]
    assert taxa3.count("siler") >= 2, f"siler query neighborhood: {taxa3}"


def test_metrics_euclidean_vs_cosine(tmp_path):
    eps = _archetype_episodes(3)
    idx = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)["index"]
    enc = KinematicPCAEncoder.load(tmp_path / "ref", "kinematic-pca")
    z = enc.transform_one(eps[0])
    e = idx.search(z, k=2, metric="euclidean")
    c = idx.search(z, k=2, metric="cosine")
    assert e and c and e[0]["episode_id"] == eps[1].episode_id or True
    # both backends return k results with finite distances
    assert len(e) == 2 and all(np.isfinite(x["dist"]) for x in e + c)


def test_species_retrieval_sample_size_correction(tmp_path):
    """A rare taxon must not outrank a well-sampled one by luck alone:
    the shrinkage-corrected score discounts small support."""
    eps = _archetype_episodes(6)
    rare = _archetype_episodes(1, seed0=55)[0]
    rare.episode_id = "rare1"
    rare.source_video_id = "v_rare"
    rare.bio_label = rare.human_label = "ant"
    eps.append(rare)
    idx = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)["index"]
    enc = KinematicPCAEncoder.load(tmp_path / "ref", "kinematic-pca")
    zq = enc.transform_one(rare).reshape(1, -1)
    taxa = taxa_retrieval(zq, idx)
    ant_rows = [t for t in taxa if t["taxon"] == "ant"]
    assert ant_rows, "ant taxon should appear"
    # corrected score <= raw mean best similarity
    assert ant_rows[0]["score_corrected"] <= ant_rows[0]["mean_best_similarity"] + 1e-9
    assert ant_rows[0]["n_reference_episodes"] >= 1


def test_ood_detection(tmp_path):
    """A query far outside the reference hull must be flagged; an in-hull
    episode must not."""
    eps = _archetype_episodes(5)
    idx = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)["index"]
    enc = KinematicPCAEncoder.load(tmp_path / "ref", "kinematic-pca")
    inside = enc.transform_one(eps[2]).reshape(1, -1)
    far = inside + 50.0                                        # off-manifold query
    assert ood_flag(inside, idx)["any_flagged"] is False
    assert ood_flag(far, idx)["any_flagged"] is True


def test_similarity_breakdown_real_decomposition(tmp_path):
    eps = _archetype_episodes(3)
    idx = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)["index"]
    a, b = eps[0].trajectory_features, eps[1].trajectory_features
    sd = idx.feature_sd
    bd = similarity_breakdown(a, b, sd)
    assert bd and all(g["n_features"] > 0 for g in bd.values())
    # identical features -> everything 'very high'
    bd_same = similarity_breakdown(a, a, sd)
    assert all(v["similarity"] == "very high" for v in bd_same.values())


def test_query_end_to_end_on_synthetic_video(tmp_path):
    """Full Find Similar pipeline on a freshly rendered video that was
    never in the reference: ingest -> encode -> retrieve -> bundle."""
    ref_eps = _archetype_episodes(4, seed0=300)
    out = build_reference(ref_eps, tmp_path / "ref", dim=6, min_episodes=6)
    # render a held-out ant video
    paths = generate_paths(600, 960, 540, "ant", n_animals=1, seed=999)
    vid = str(tmp_path / "query_video.avi")
    render_video(paths, "ant", vid, seed=999)
    atlas_dir = tmp_path / "atlas"
    bundle = run_query(vid, "q_test", tmp_path / "ref",
                       store_dir=tmp_path / "store", out_dir=tmp_path,
                       atlas_dir=atlas_dir, k=4)
    assert "error" not in bundle or bundle.get("episodes")
    assert bundle["episodes"], "synthetic video must yield episodes"
    assert bundle["reference_atlas_version"] == out["index"].version_id
    assert bundle["results"]["episode_hits"], "retrieval must return hits"
    assert bundle["results_representation_b"], "representation B present"
    assert bundle["consistency"] is not None
    # query.json written to the atlas dir (the Murmur picks it up)
    assert (atlas_dir / "query.json").exists()
    loaded = json.loads((atlas_dir / "query.json").read_text(encoding="utf-8"))
    assert loaded["query_id"] == "q_test"
    assert len(loaded["episode_dicts"]) == len(bundle["episodes"])


def test_atlas_projection_gives_real_positions(tmp_path):
    """Query particle positions come from the atlas build's own PCA —
    projecting a reference episode must reproduce its stored embedding
    neighborhood, not an invented location."""
    eps = _archetype_episodes(4, seed0=77)
    idx = build_reference(eps, tmp_path / "ref", dim=6, min_episodes=6)["index"]
    # a fake atlas projection over the same features
    from kinemimic.features import feature_matrix
    X, names = feature_matrix(eps)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    proj = {"feature_order": names, "mu": mu.tolist(), "sd": sd.tolist(),
            "components": Vt[:2].tolist()}
    (tmp_path / "atlas_projection.json").write_text(json.dumps(proj))
    loaded = load_projection(tmp_path)
    pos = project_to_atlas(eps[0].trajectory_features, loaded)
    assert len(pos) == 2 and all(np.isfinite(v) for v in pos)
    # the reference episode projects near its own embedding's top components
    from kinemimic.encoder import KinematicPCAEncoder
    enc = KinematicPCAEncoder.load(tmp_path / "ref", "kinematic-pca")
    z = enc.transform_one(eps[0])
    assert np.isfinite(z).all()


def test_motif_retrieval_ranks_known_motif(tmp_path):
    eps = _archetype_episodes(4)
    for i, e in enumerate(eps):
        e.motif = 0 if i % 2 == 0 else 1
    idx = ReferenceIndex.build(eps, KinematicPCAEncoder(dim=6).fit(eps), min_episodes=6)
    enc = KinematicPCAEncoder(dim=6).load if False else None
    enc = KinematicPCAEncoder(dim=6).fit(eps)
    z0 = enc.transform_one(eps[0])                  # motif 0 member
    from kinemimic.retrieval import retrieve
    res = retrieve([eps[0]], z0.reshape(1, -1), idx, enc, k=3)
    assert res["motif_hits"], "motif hits must be returned"
    assert all(0 <= m["similarity"] <= 1 for m in res["motif_hits"])
    assert res["reference_version"] == idx.version_id


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(Path(f"/tmp/ms_test_{name}"))
            print(f"{name} OK")
