"""Behavioral Retrieval Engine: reference index + similarity search.

Given query behavior vectors, answer "what moves like this?" at three
levels (docs/RETRIEVAL.md):

- Episode retrieval: nearest real reference episodes (exact search;
  backend-swappable for FAISS/HNSW at scale).
- Motif retrieval: distance to machine-discovered motif centroids.
- Taxa retrieval: behaviorally similar taxa — comparing the query
  behavioral CLOUD against each taxon's episode distribution (centroid
  AND distribution distances), with sample-size-aware aggregation and
  support counts. The output is "behaviorally most similar taxa",
  never "predicted species identity".

Scientific rules:
- similarity is not identity probability;
- reference index is versioned (atlas_reference_vNNN) and never silently
  changed; every retrieval records the index version it used;
- out-of-distribution queries are flagged, not forced onto a taxon;
- similarity explanations come from real standardized per-group
  distance contributions, never invented.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import __version__
from .encoder import BaseEncoder
from .schema import Episode, Provenance

REJECTED_STATES = {"rejected", "tracking_failure", "severe_occlusion"}
OOD_PERCENTILE = 95.0
SHRINKAGE_N0 = 5.0          # taxa sample-size prior strength (documented)


# --------------------------------------------------------------------------
# reference index
# --------------------------------------------------------------------------

@dataclass
class IndexPolicy:
    """Which episodes may enter a reference index (recorded per version)."""
    name: str = "not-rejected"       # all | not-rejected | reviewed
    min_duration_s: float = 3.0
    require_human_label: bool = False

    def accepts(self, ep: Episode) -> bool:
        if ep.annotation_status in REJECTED_STATES:
            return False
        if self.name == "reviewed" and ep.annotation_status not in ("accepted",):
            return False
        if self.require_human_label and not ep.human_label:
            return False
        if ep.duration_s < self.min_duration_s:
            return False
        return True


@dataclass
class ReferenceIndex:
    version_id: str
    encoder_name: str
    encoder_provenance: dict
    policy: dict
    entries: list[dict] = field(default_factory=list)   # metadata per row
    embeddings: np.ndarray | None = None
    created_utc: str = ""
    motifs: dict = field(default_factory=dict)          # motif -> centroid vec

    @classmethod
    def build(cls, episodes: list[Episode], encoder: BaseEncoder,
              policy: IndexPolicy | None = None,
              previous_version: str | None = None,
              out_dir: str | Path | None = None,
              min_episodes: int = 10) -> "ReferenceIndex":
        policy = policy or IndexPolicy()
        kept = [e for e in episodes if policy.accepts(e)]
        if len(kept) < min_episodes:
            raise ValueError(f"only {len(kept)} episodes pass the reference "
                             f"policy {policy.name!r}; not enough to build an index")
        Z = encoder.transform(kept)
        motif_centroids: dict[int, np.ndarray] = {}
        for e, z in zip(kept, Z):
            if e.motif is not None:
                motif_centroids.setdefault(int(e.motif), []).append(z)
        motifs = {m: np.mean(v, axis=0).tolist()
                  for m, v in sorted(motif_centroids.items())}
        n = sum(len(kept[i].processing_history) > 0 for i in range(len(kept)))
        version_id = f"atlas_reference_v{int(time.time()) % 100000:05d}"
        idx = cls(version_id=version_id, encoder_name=encoder.name,
                  encoder_provenance=encoder.provenance(),
                  policy=asdict(policy), embeddings=Z,
                  entries=[{
                      "episode_id": e.episode_id,
                      "taxon": e.effective_label(),
                      "label_source": ("human" if e.human_label else
                                       ("machine" if e.machine_label else "unannotated")),
                      "site": e.site_id or e.environment.site,
                      "session": e.session_id or e.environment.date,
                      "video": e.source_video_id,
                      "motif": e.motif,
                      "qc_state": e.annotation_status,
                      "duration_s": round(e.duration_s, 2),
                  } for e in kept],
                  motifs=motifs,
                  created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        idx.entries_meta = {
            "n_input": len(episodes), "n_kept": len(kept),
            "n_rejected_by_policy": len(episodes) - len(kept),
            "software_version": __version__,
            "parent_version": previous_version,
            "taxon_counts": {t: sum(1 for e in kept if e.effective_label() == t)
                             for t in {e.effective_label() for e in kept}},
            "note": ("index building is label-blind; taxon metadata is "
                     "attached for evaluation and aggregation only"),
        }
        if out_dir:
            idx.save(out_dir)
        return idx

    # -- persistence --
    def save(self, out_dir: str | Path) -> Path:
        d = Path(out_dir) / self.version_id
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        (d / "index.json").write_text(json.dumps({
            "version_id": self.version_id, "encoder_name": self.encoder_name,
            "encoder_provenance": self.encoder_provenance,
            "policy": self.policy, "entries": self.entries,
            "motifs": {str(k): v for k, v in self.motifs.items()},
            "created_utc": self.created_utc, "meta": getattr(self, "entries_meta", {}),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "ReferenceIndex":
        d = Path(directory)
        if not (d / "index.json").exists():
            versions = sorted(d.glob("atlas_reference_v*/index.json"))
            if not versions:
                raise FileNotFoundError(f"no reference index under {d}")
            d = versions[-1].parent          # newest version wins
        state = json.loads((d / "index.json").read_text(encoding="utf-8"))
        idx = cls(version_id=state["version_id"], encoder_name=state["encoder_name"],
                  encoder_provenance=state["encoder_provenance"],
                  policy=state["policy"], entries=state["entries"],
                  motifs={int(k): np.asarray(v) for k, v in state["motifs"].items()},
                  created_utc=state["created_utc"])
        idx.embeddings = np.load(d / "embeddings.npy")
        idx.entries_meta = state.get("meta", {})
        idx.feature_sd = state.get("feature_sd", {})
        idx.features_by_id = state.get("features_by_id", {})
        ep_file = d / "episodes.jsonl"
        if ep_file.exists():
            from .store import EpisodeStore
            idx.episodes_ref = EpisodeStore.read_episodes(ep_file)
        return idx

    # -- search backends --
    def search(self, q: np.ndarray, k: int = 8, metric: str = "euclidean",
               exclude_ids: set[str] | None = None) -> list[dict]:
        """Exact nearest neighbors. Backend swap point for ANN at scale."""
        Z = self.embeddings
        if metric == "cosine":
            qn, Zn = np.linalg.norm(q), np.linalg.norm(Z, axis=1)
            d = 1.0 - (Z @ q) / np.maximum(Zn * max(qn, 1e-12), 1e-12)
        else:
            d = np.linalg.norm(Z - q, axis=1)
        order = np.argsort(d)
        out, seen = [], set(exclude_ids or ())
        for i in order:
            e = self.entries[i]
            if e["episode_id"] in seen:
                continue
            seen.add(e["episode_id"])
            out.append({"episode_id": e["episode_id"], "dist": float(d[i]),
                        "similarity": float(np.exp(-d[i] / 3.0)),
                        "taxon": e["taxon"], "label_source": e["label_source"],
                        "motif": e["motif"], "video": e["video"],
                        "site": e["site"], "qc_state": e["qc_state"]})
            if len(out) >= k:
                break
        return out

    def internal_nn_distances(self, metric: str = "euclidean") -> np.ndarray:
        """Each reference episode's distance to its nearest OTHER reference
        episode — the yardstick for out-of-distribution detection."""
        Z = self.embeddings
        if metric == "cosine":
            Zn = np.linalg.norm(Z, axis=1, keepdims=True)
            S = (Z @ Z.T) / np.maximum(Zn @ Zn.T, 1e-12)
            np.fill_diagonal(S, 1.0)
            return 1.0 - S.min(axis=1)
        D = np.linalg.norm(Z[:, None, :] - Z[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        return D.min(axis=1)


# --------------------------------------------------------------------------
# retrieval result assembly
# --------------------------------------------------------------------------

FEATURE_GROUPS = {
    "trajectory_geometry": ["sinuosity", "straightness_index", "net_displacement",
                            "curvature_mean", "distance"],
    "speed_dynamics": ["speed_mean", "speed_p90", "accel_rms"],
    "turning_dynamics": ["turn_rate_mean", "turn_rate_p90"],
    "stop_go_rhythm": ["frac_time_moving", "n_pauses", "pause_run_mean_s",
                       "move_run_mean_s"],
    "intermittency": ["speed_cv"],
}


def similarity_breakdown(feat_a: dict, feat_b: dict, sd: dict) -> dict:
    """Real distance decomposition: per group, mean standardized absolute
    feature difference -> qualitative level. No invented numbers."""
    groups = {}
    for g, keys in FEATURE_GROUPS.items():
        ds = []
        for k in keys:
            if k in feat_a and k in feat_b and k in sd and sd[k] > 1e-9:
                ds.append(abs(feat_a[k] - feat_b[k]) / sd[k])
        if ds:
            m = float(np.mean(ds))
            level = ("very high" if m < 0.35 else "high" if m < 0.8 else
                     "moderate" if m < 1.5 else "low")
            groups[g] = {"mean_standardized_diff": round(m, 3), "similarity": level,
                         "n_features": len(ds)}
    return groups


def taxa_retrieval(query_Z: np.ndarray, index: ReferenceIndex, k: int = 6,
                   metric: str = "euclidean") -> list[dict]:
    """Behaviorally similar taxa: query cloud vs each taxon's episode cloud.

    Centroid distance AND mean per-dimension Wasserstein distance are both
    reported (multi-modal taxa make centroids misleading). Ranking uses a
    sample-size-corrected score: mean top-k episode similarity shrunk by
    n/(n+N0) so a taxon with 3 episodes cannot outrank a well-sampled one
    by luck alone. Method recorded in docs/RETRIEVAL.md.
    """
    by_taxon: dict[str, list[int]] = {}
    for i, e in enumerate(index.entries):
        by_taxon.setdefault(e["taxon"], []).append(i)
    qn = np.linalg.norm(query_Z, axis=1)

    out = []
    for taxon, rows in by_taxon.items():
        Zt = index.embeddings[rows]
        n = len(rows)
        # centroid distance (query mean -> taxon mean)
        cd = float(np.linalg.norm(Zt.mean(0) - query_Z.mean(0)))
        # distribution distance: mean per-dimension Wasserstein-1
        from scipy.stats import wasserstein_distance
        wd = float(np.mean([wasserstein_distance(query_Z[:, j], Zt[:, j]) /
                            max(np.std(index.embeddings[:, j]), 1e-9)
                            for j in range(query_Z.shape[1])]))
        # mean best-episode similarity across query episodes (raw score)
        best = []
        for qi in range(len(query_Z)):
            d = np.linalg.norm(Zt - query_Z[qi], axis=1)
            best.append(float(np.exp(-d.min() / 3.0)))
        raw = float(np.mean(best))
        corrected = raw * n / (n + SHRINKAGE_N0)
        out.append({"taxon": taxon, "n_reference_episodes": n,
                    "centroid_distance": round(cd, 4),
                    "distribution_wasserstein": round(wd, 4),
                    "mean_best_similarity": round(raw, 4),
                    "score_corrected": round(corrected, 4),
                    "human_confirmed": sum(1 for i in rows
                                           if index.entries[i]["label_source"] == "human")})
    out.sort(key=lambda r: -r["score_corrected"])
    return out[:k]


def ood_flag(query_Z: np.ndarray, index: ReferenceIndex,
             metric: str = "euclidean") -> dict:
    """§47: is the query inside the well-sampled region of the atlas?

    A query is novel-flagged when its distance to the nearest reference
    episode exceeds the 95th percentile of reference-internal nearest-
    neighbor distances."""
    ref_nn = index.internal_nn_distances(metric)
    thresh = float(np.percentile(ref_nn, OOD_PERCENTILE))
    per_ep = []
    for q in query_Z:
        d = float(np.min(np.linalg.norm(index.embeddings - q, axis=1))) \
            if metric == "euclidean" else \
            float(np.min(1.0 - (index.embeddings @ q) /
                         np.maximum(np.linalg.norm(index.embeddings, axis=1) *
                                    np.linalg.norm(q), 1e-12)))
        per_ep.append(d)
    flagged = [d > thresh for d in per_ep]
    return {"metric": metric, "threshold": round(thresh, 4),
            "threshold_percentile": OOD_PERCENTILE,
            "per_episode_dist": [round(d, 4) for d in per_ep],
            "per_episode_flagged": flagged,
            "any_flagged": any(flagged),
            "message": ("This movement lies outside the well-sampled region "
                        "of the current atlas." if any(flagged) else "")}


def retrieve(query_episodes: list[Episode], query_Z: np.ndarray,
             index: ReferenceIndex, encoder: BaseEncoder, k: int = 6,
             metric: str = "euclidean") -> dict:
    """Full three-level retrieval result for one query bundle."""
    episode_hits = []
    for qi, (ep, z) in enumerate(zip(query_episodes, query_Z)):
        hits = index.search(z, k=k, metric=metric)
        # interpretable similarity breakdown vs the top hit
        bd = None
        if hits:
            top = next((e for e in index.entries
                        if e["episode_id"] == hits[0]["episode_id"]), None)
        episode_hits.append({
            "query_episode_id": ep.episode_id,
            "duration_s": round(ep.duration_s, 2),
            "neighbors": hits,
            "breakdown": bd,
        })
    # breakdown vs top neighbor (needs reference features; recomputed from
    # stored index entries is impossible — caller attaches features via
    # attach_breakdown when episodes are available)
    motif_hits = []
    qmean = query_Z.mean(0)
    if index.motifs:
        ds = sorted(((float(np.linalg.norm(qmean - np.asarray(c))), m)
                     for m, c in index.motifs.items()))
        motif_hits = [{"motif": int(m), "dist": round(d, 4),
                       "similarity": round(float(np.exp(-d / 3.0)), 4)}
                      for d, m in ds[:5]]
    taxa = taxa_retrieval(query_Z, index, metric=metric)
    ood = ood_flag(query_Z, index, metric)
    median_z = np.median(query_Z, axis=0)
    agg = {"n_query_episodes": len(query_episodes),
           "mean_embedding": [round(float(x), 4) for x in qmean],
           "median_embedding": [round(float(x), 4) for x in median_z],
           "aggregation_note": "distribution-level comparison preferred; "
                               "mean and median embeddings reported for transparency"}
    prov = Provenance(
        software_version=__version__, model_name="retrieval-v1", model_version="1",
        parameters=dict(reference_atlas_version=index.version_id,
                        encoder=index.encoder_name, metric=metric, k=k,
                        n_reference=len(index.entries),
                        policy=index.policy, ood_percentile=OOD_PERCENTILE,
                        shrinkage_n0=SHRINKAGE_N0)).to_dict()
    return {"episode_hits": episode_hits, "motif_hits": motif_hits,
            "taxa": taxa, "ood": ood, "aggregation": agg,
            "reference_version": index.version_id, "provenance": prov}
