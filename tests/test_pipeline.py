import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionscape.schema import Episode
from motionscape.features import trajectory_features, feature_matrix
from motionscape.synth import generate_paths
from motionscape.track import tracks_to_episodes, Track
from motionscape.classify import classify_episode, mimicry_fingerprint
from motionscape.embedding import PCAEmbedder
from motionscape.motifs import find_motifs


def test_features_separate_archetypes():
    ant = generate_paths(900, 960, 540, "ant", n_animals=1, seed=1)[0]
    siler = generate_paths(900, 960, 540, "siler", n_animals=1, seed=2)[0]
    fa = trajectory_features(ant.tolist(), 30.0)
    fs = trajectory_features(siler.tolist(), 30.0)
    # ants: steadier (lower speed_cv), more time moving
    assert fa["speed_cv"] < fs["speed_cv"]
    assert fa["frac_time_moving"] > fs["frac_time_moving"]


def test_classifier():
    def ep_from(path, aspect):
        return Episode(frames=list(range(len(path))), centroids_px=path.tolist(),
                       bbox_sizes_px=[[aspect * 9, 9]] * len(path),
                       detection_confidence=[0.9] * len(path),
                       start_frame=0, end_frame=len(path), fps=30.0)
    ant = generate_paths(900, 960, 540, "ant", n_animals=1, seed=3)[0]
    siler = generate_paths(900, 960, 540, "siler", n_animals=1, seed=4)[0]
    assert classify_episode(ep_from(ant, 3.0))[0] == "ant"
    assert classify_episode(ep_from(siler, 1.3))[0] == "siler"


def test_embedding_and_motifs_roundtrip():
    eps = []
    for s, arch in [(10, "ant"), (11, "siler")]:
        for p in generate_paths(600, 960, 540, arch, n_animals=2, seed=s):
            eps.append(Episode(frames=list(range(len(p))), centroids_px=p.tolist(),
                               start_frame=0, end_frame=len(p), fps=30.0,
                               embedding=[0, 0]))
            eps[-1].trajectory_features = trajectory_features(p.tolist(), 30.0)
    X, names = feature_matrix(eps)
    emb, prov = PCAEmbedder(2).fit_transform(X, names)
    assert emb.shape == (len(eps), 2) and prov.parameters["n_components"] == 2
    labels, centers, _ = find_motifs(X, k=3)
    assert set(labels) <= {0, 1, 2}


def test_mimicry_fingerprint_shape():
    ant = generate_paths(600, 960, 540, "ant", n_animals=3, seed=5)
    siler = generate_paths(600, 960, 540, "siler", n_animals=3, seed=6)
    mk = lambda ps, lbl: [Episode(centroids_px=p.tolist(), start_frame=0,
                                  end_frame=len(p), fps=30.0, bio_label=lbl,
                                  embedding=[float(np.random.randn()), float(np.random.randn())])
                          for p in ps]
    out = mimicry_fingerprint(mk(siler, "siler"), mk(ant, "ant"))
    assert out["fingerprint"] and all(0 <= v <= 1.2 for v in out["fingerprint"].values())
    assert "provenance" in out


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"{name} OK")
