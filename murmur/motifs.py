"""Unsupervised movement motifs.

K-means over the standardized feature/embedding space discovers
recurring movement patterns M1..Mk. The comparative read-out follows in
``compare.py``: motifs frequent in ants AND Siler but rare in
non-mimetic spiders are candidate dynamic-mimicry behaviors.
"""

from __future__ import annotations

import numpy as np

from . import __version__
from .schema import Provenance

MODEL_NAME = "kmeans-motifs-v1"


def kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        labels = d.argmin(1)
        for j in range(len(C)):
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
    inertia = float(((X - C[labels]) ** 2).sum())
    return labels, C, inertia


def find_motifs(X: np.ndarray, k: int = 8, seed: int = 0):
    labels, centers, inertia = kmeans(X, k, seed=seed)
    prov = Provenance(software_version=__version__, model_name=MODEL_NAME,
                      model_version="1", parameters=dict(k=k, seed=seed, inertia=inertia))
    return labels, centers, prov


def motif_profile(labels: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-group motif occupancy (rows sum to 1). groups: name -> boolean mask."""
    k = int(labels.max()) + 1
    out = {}
    for name, mask in groups.items():
        counts = np.bincount(labels[mask], minlength=k).astype(float)
        out[name] = counts / max(mask.sum(), 1)
    return out
