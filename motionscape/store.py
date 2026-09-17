"""Append-only episode storage.

Layout of an analysis directory::

    runs/<run_id>/
        manifest.json        # provenance of the run itself
        episodes.jsonl       # one Episode per line (snapshot at run time)
        annotations.jsonl    # human/model label refinements (separate!)
        embeddings.npz       # optional dense arrays
        viz/index.html       # exported visualization

Analyses never overwrite each other: each run gets a fresh ``run_id``
directory. Later refinement of biological labels is written as new
annotation records that supersede by (episode_id, timestamp) ordering.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import __version__
from .schema import Episode, Provenance, new_id


class EpisodeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_run(self, model_name: str, model_version: str,
                parameters: dict, parent_run: str | None = None) -> Path:
        run_id = new_id("run")
        run_dir = self.root / "runs" / run_id
        (run_dir).mkdir(parents=True, exist_ok=True)
        prov = Provenance(software_version=__version__, model_name=model_name,
                          model_version=model_version, parameters=parameters,
                          parent_ids=[parent_run] if parent_run else [])
        (run_dir / "manifest.json").write_text(json.dumps(prov.to_dict(), indent=2))
        return run_dir

    def write_episodes(self, run_dir: Path, episodes: list[Episode]) -> Path:
        path = run_dir / "episodes.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep.to_dict()) + "\n")
        return path

    @staticmethod
    def read_episodes(path: str | Path) -> list[Episode]:
        episodes = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(Episode.from_dict(json.loads(line)))
        return episodes

    @staticmethod
    def add_annotation(root: str | Path, episode_id: str, bio_label: str,
                       confidence: float, source: str = "human",
                       detail: dict | None = None) -> Path:
        """Refine a biological label without touching observation data."""
        import time
        ann = {"episode_id": episode_id, "bio_label": bio_label,
               "bio_label_confidence": confidence, "source": source,
               "detail": detail or {}, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        ann_dir = Path(root) / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        path = ann_dir / "annotations.jsonl"
        with open(path, "a", encoding="utf-8") as f:  # append: never overwrite
            f.write(json.dumps(ann) + "\n")
        return path
