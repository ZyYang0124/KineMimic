"""MOTIONSCAPE encoders: movement episodes -> fixed-dim behavior vectors.

A behavioral retrieval system, not a species classifier: encoders map
trajectories to vectors z ∈ R^d such that similar MOVEMENTS land close
together. Biological labels never enter fitting or transform (they are
used downstream for evaluation and aggregation only).

Two representations are kept side by side (docs/RETRIEVAL.md):

- Representation A — ``KinematicPCAEncoder``: the interpretable 16-D
  kinematic feature vector, standardized, projected by PCA. Strong,
  transparent baseline.
- Representation B — ``ShapeSeriesEncoder``: resampled speed / turning /
  pause-state time series (shape-normalized: translation/rotation
  invariant by construction, camera-invariant by design), PCA-projected.
  Captures temporal form that scalar features average away.

Contract (docs/MODEL_CARD.md):
- fixed, configurable output dimension ``dim``
- full provenance: encoder name/version, feature version, preprocessing
  version, weights checksum; saving/loading is exact
- ``fit`` sees only reference episodes; ``transform`` is deterministic
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .features import feature_matrix, trajectory_features
from .schema import Episode, Provenance

FEATURE_VERSION = "kinematics-v1"
PREPROCESSING_VERSION = "resample-v1"


def _checksum(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _pca_fit(X: np.ndarray, dim: int):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    k = max(1, min(dim, Vt.shape[0]))
    return mu, sd, Vt[:k], (S ** 2 / (S ** 2).sum())[:k]


def _pca_transform(x: np.ndarray, mu, sd, comp) -> np.ndarray:
    return ((x - mu) / sd) @ comp.T


class BaseEncoder:
    """BehaviorVector interface: fit(episodes) -> transform(episode) -> R^dim.

    Any backend (future learned/contrastive encoders, FAISS-adjacent
    pipelines) implements the same surface so retrieval never changes.
    """

    name = "base"
    version = "0"

    def __init__(self, dim: int = 12):
        self.dim = int(dim)
        self.fitted = False

    # -- subclasses implement --
    def _fit(self, episodes: list[Episode]):  # pragma: no cover
        raise NotImplementedError

    def _transform_one(self, ep: Episode) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # -- public surface --
    def fit(self, episodes: list[Episode]) -> "BaseEncoder":
        self._fit(episodes)
        self.fitted = True
        return self

    def transform(self, episodes: list[Episode]) -> np.ndarray:
        assert self.fitted, "encoder not fitted"
        return np.vstack([self._transform_one(e) for e in episodes])

    def transform_one(self, ep: Episode) -> np.ndarray:
        return self.transform([ep])[0]

    @property
    def weights_checksum(self) -> str:
        return _checksum(*[v for k, v in sorted(vars(self).items())
                           if isinstance(v, np.ndarray)]) if self.fitted else ""

    def provenance(self) -> dict:
        return Provenance(
            software_version=__version__, model_name=self.name,
            model_version=self.version,
            parameters=dict(dim=self.dim, feature_version=FEATURE_VERSION,
                            preprocessing_version=PREPROCESSING_VERSION,
                            weights_checksum=self.weights_checksum,
                            labels_used="none — encoders are label-blind",
                            trained_on="reference episodes only")).to_dict()

    # -- persistence --
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        plain = {k: v for k, v in sorted(vars(self).items())
                 if isinstance(v, (str, int, float, bool, list)) and k != "name"}
        state = {"name": self.name, "version": self.version, "dim": self.dim,
                 "plain": plain,
                 "provenance": self.provenance(),
                 "arrays": {k: v.tolist() for k, v in sorted(vars(self).items())
                            if isinstance(v, np.ndarray)}}
        (d / f"encoder_{self.name}.json").write_text(
            json.dumps(state, indent=1), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path, name: str) -> "BaseEncoder":
        d = Path(directory)
        state = json.loads((d / f"encoder_{name}.json").read_text(encoding="utf-8"))
        enc = cls(dim=state["dim"])
        for k, v in state["arrays"].items():
            setattr(enc, k, np.asarray(v, float))
        for k, v in state.get("plain", {}).items():
            setattr(enc, k, v)
        enc.fitted = True
        return enc


class KinematicPCAEncoder(BaseEncoder):
    """Representation A: interpretable kinematics -> standardized PCA."""

    name = "kinematic-pca"
    version = "1"

    def _fit(self, episodes):
        X, self.feature_names = feature_matrix(episodes)
        self.mu, self.sd, self.comp, self.evr = _pca_fit(X, self.dim)

    def _transform_one(self, ep):
        f = ep.trajectory_features or trajectory_features(
            ep.centroids_px, ep.fps, ep.px_per_cm, ep.frames)
        x = np.array([f.get(k, 0.0) for k in self.feature_names], float)
        return _pca_transform(x, self.mu, self.sd, self.comp)


def _resampled_series(ep: Episode, length: int,
                      normalize: str = "none") -> np.ndarray:
    """speed / |turn rate| / moving state, resampled to fixed length.

    Translation- and rotation-invariant (magnitudes only). With
    ``normalize="per_episode"`` speed and turning are z-scored within the
    episode (shape-only representation, robust to dataset unit and camera
    scale differences). Non-finite samples are dropped — dirty series
    must never poison the fit."""
    vel = ep.metadata.get("velocity_series")
    if vel and not ep.centroids_px:
        v = np.asarray(vel, float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return np.zeros(length * 3, float)
        turn = np.abs(np.diff(v, prepend=v[0]))
        moving = (v > 0.1 * max(np.median(v), 1e-9)).astype(float)
    else:
        xy = np.asarray([c for c in ep.centroids_px if c], float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 3:
            return np.zeros(length * 3, float)
        fr = np.asarray(ep.frames, int)[:len(xy)]
        steps = np.diff(fr)
        step = int(np.bincount(steps[steps > 0]).argmax()) if (steps > 0).any() else 1
        dt = step / max(ep.fps, 1)
        scale = (1.0 / ep.px_per_cm) if ep.px_per_cm else 1.0
        same = steps == step
        v = (np.hypot(*np.diff(xy, axis=0).T) * scale / dt)[same]
        heading = np.arctan2(*np.diff(xy, axis=0).T[::-1])
        valid = same[:-1] & same[1:] if len(same) > 1 else np.zeros(0, bool)
        # wrapped angle difference: rotation-invariant across the +-pi seam
        dh = np.diff(heading)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        turn = (np.abs(dh) / dt)[valid]
        moving = (v > 0.1 * max(np.median(v) if len(v) else 0, 1e-9)).astype(float)
        if len(v) < 2:
            return np.zeros(length * 3, float)

    if normalize == "per_episode":
        v = (v - v.mean()) / max(v.std(), 1e-9)
        turn = (turn - turn.mean()) / max(turn.std(), 1e-9)

    def rs(a):
        if len(a) == 0:
            return np.zeros(length)
        idx = np.unique(np.linspace(0, len(a) - 1, length).astype(int))
        return np.nan_to_num(np.asarray(a, float)[idx], nan=0.0,
                             posinf=0.0, neginf=0.0)

    return np.concatenate([rs(v), rs(turn), rs(moving)])


class ShapeSeriesEncoder(BaseEncoder):
    """Representation B: shape-normalized movement series -> PCA.

    Per-episode z-scoring of speed and turning removes absolute physical
    scale (px/s vs cm/s vs mm/s across datasets — docs/RETRIEVAL.md
    §Scale) so the space encodes the SHAPE of movement: burst structure,
    turn phrasing, pause rhythm. Physical magnitudes remain in
    Representation A. ``moving`` stays binary.
    """

    name = "shape-series"
    version = "2"

    def __init__(self, dim: int = 12, series_length: int = 64,
                 normalize: str = "per_episode"):
        self.series_length = int(series_length)
        self.normalize = normalize
        super().__init__(dim)

    def _fit(self, episodes):
        X = np.vstack([_resampled_series(e, self.series_length,
                                         normalize=self.normalize)
                       for e in episodes])
        self.mu, self.sd, self.comp, self.evr = _pca_fit(X, self.dim)

    def _transform_one(self, ep):
        x = _resampled_series(ep, self.series_length, normalize=self.normalize)
        return _pca_transform(x, self.mu, self.sd, self.comp)

    def provenance(self) -> dict:
        p = super().provenance()
        p["parameters"]["series_length"] = self.series_length
        p["parameters"]["normalize"] = self.normalize
        return p


def load_any(directory: str | Path, name: str) -> BaseEncoder:
    for cls in (KinematicPCAEncoder, ShapeSeriesEncoder):
        if cls.name == name:
            return cls.load(directory, name)
    raise ValueError(f"unknown encoder {name!r}")
