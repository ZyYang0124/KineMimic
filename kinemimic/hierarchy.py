"""Sampling hierarchy: Site -> Session -> Video -> Episode.

An episode is the unit of *visualization and movement analysis*, but it is
NOT an independent biological replicate: hundreds of episodes may come from
one video on one afternoon at one site. Comparative statistics must respect
the hierarchy (hierarchical bootstrap / mixed models / blocked permutation)
instead of reporting n = 5,000 independent replicates.

This module builds the explicit tree and provides a cluster (hierarchical)
bootstrap that resamples whole sites, then sessions within sites, then
episodes within sessions.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np

from .schema import Episode


def key_of(ep: Episode, level: str) -> str:
    if level == "site":
        return ep.site_id or ep.environment.site or "unspecified_site"
    if level == "session":
        return ep.session_id or f"{key_of(ep, 'site')}/{ep.environment.date or 'unknown_date'}"
    if level == "video":
        return ep.source_video_id or "unknown_video"
    return ep.episode_id


def hierarchy_tree(episodes: list[Episode]) -> dict:
    """Nested counts: {site: {session: {video: n_episodes}}}."""
    tree: dict[str, dict[str, Counter]] = {}
    for ep in episodes:
        site, session, video = (key_of(ep, l) for l in ("site", "session", "video"))
        tree.setdefault(site, {}).setdefault(session, Counter())[video] += 1
    return {s: {ss: dict(c) for ss, c in sess.items()} for s, sess in tree.items()}


def hierarchy_summary(episodes: list[Episode]) -> dict:
    sites = {key_of(e, "site") for e in episodes}
    sessions = {key_of(e, "session") for e in episodes}
    videos = {key_of(e, "video") for e in episodes}
    return {"n_episodes": len(episodes), "n_videos": len(videos),
            "n_sessions": len(sessions), "n_sites": len(sites),
            "tree": hierarchy_tree(episodes)}


def _cluster_keys(episodes: list[Episode], level: str) -> list[str]:
    return sorted({key_of(e, level) for e in episodes})


def hierarchical_bootstrap(episodes: list[Episode], stat: Callable[[list[Episode]], float],
                           n_boot: int = 500, seed: int = 0,
                           level: str = "auto") -> dict:
    """Cluster bootstrap of ``stat`` respecting the sampling hierarchy.

    Resampling units are sites when >= 2 sites exist, otherwise sessions,
    otherwise videos (reported in ``resampled_level``). Within each resample,
    all episodes of a drawn cluster are included (whole-cluster resampling),
    which is what keeps pseudo-replication out of the confidence interval.
    """
    if not episodes:
        return {"stat": None, "ci_low": None, "ci_high": None, "n_boot": 0,
                "resampled_level": None}
    if level == "auto":
        level = ("site" if len(_cluster_keys(episodes, "site")) >= 2 else
                 "session" if len(_cluster_keys(episodes, "session")) >= 2 else "video")
    keys_of_eps = [key_of(e, level) for e in episodes]
    clusters = sorted(set(keys_of_eps))
    by_cluster: dict[str, list[Episode]] = {c: [] for c in clusters}
    for ep, c in zip(episodes, keys_of_eps):
        by_cluster[c].append(ep)

    rng = np.random.default_rng(seed)
    point = float(stat(episodes))
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(len(clusters), size=len(clusters), replace=True)
        sample: list[Episode] = []
        for ci in draw:
            sample.extend(by_cluster[clusters[int(ci)]])
        if not sample:
            continue
        stats.append(float(stat(sample)))
    if not stats:
        return {"stat": point, "ci_low": point, "ci_high": point,
                "n_boot": 0, "resampled_level": level}
    return {"stat": point,
            "ci_low": float(np.percentile(stats, 2.5)),
            "ci_high": float(np.percentile(stats, 97.5)),
            "n_boot": len(stats), "resampled_level": level}
