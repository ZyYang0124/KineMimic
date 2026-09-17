"""Behavioral space embedding.

The machine-discovery route: episodes -> standardized kinematics ->
PCA (numpy SVD; no heavyweight deps). Every episode's embedding is
written back with model provenance so the space is reproducible.
UMAP/autoencoder variants can later be added as alternative models --
old runs are never invalidated.
"""

from __future__ import annotations

import numpy as np

from . import __version__
from .schema import Provenance

MODEL_NAME = "pca-v1"


class PCAEmbedder:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray, feature_names: list[str]):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        emb = U[:, :k] * S[:k]
        self.components_ = Vt[:k]
        self.feature_names_ = feature_names
        self.mean_, self.sd_ = mu, sd
        self.explained_variance_ratio_ = (S ** 2 / (S ** 2).sum())[:k]
        prov = Provenance(software_version=__version__, model_name=MODEL_NAME,
                          model_version="1", parameters=dict(
                              n_components=k, feature_names=feature_names,
                              explained_variance_ratio=self.explained_variance_ratio_.tolist()))
        return emb, prov

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) / self.sd_
        return Z @ self.components_.T
