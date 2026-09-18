"""KineMimic Tracking Benchmark (§28-35, §51-53, §76).

Not COCO mAP, not a MOT leaderboard: the metrics that matter for
behavioral phenotyping, computed on controlled synthetic scenes with
exact ground truth.

Scenarios (docs/VISION_BENCHMARK.md): crossing, collision-merge,
stationary pause, occlusion gap, fast animal, single animal.

KineMimic metrics:

- Episode Purity          — fraction of an episode's frames belonging to
                            one true animal (identity contamination);
- Usable Episode Recall   — GT movement time covered by pure episodes;
- False Merge Rate        — episodes mixing >=2 true animals (the worst
                            error; heavily penalized);
- Fragmentation Rate      — extra episodes per GT animal;
- Trajectory Error        — px distance between episode points and GT.

Cost = 5*false_merge_rate + 0.5*fragmentation + (1-usable_recall).
False merges dominate by design (§33-34): never tune fragmentation down
at the price of purity.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from pathlib import Path

from ..schema import Episode
from .pipeline import VisionConfig, run_vision_frontend
from .tracker import TrackerConfig


# --------------------------------------------------------------------------
# synthetic scenes with exact ground truth
# --------------------------------------------------------------------------

@dataclass
class GTAnimal:
    name: str
    waypoints: list[tuple[float, float, float]]      # (t_s, x_px, y_px)
    radius: float = 9.0
    hidden: list[tuple[float, float]] = field(default_factory=list)  # occlusion windows (s)


@dataclass
class Scene:
    name: str
    duration_s: float
    fps: int = 30
    w: int = 960
    h: int = 540
    animals: list[GTAnimal] = field(default_factory=list)
    occluder: tuple[int, int, int, int] | None = None  # x, y, w, h

    def position(self, name: str, t: float) -> tuple[float, float] | None:
        a = next(a for a in self.animals if a.name == name)
        for (t0, x0, y0), (t1, x1, y1) in zip(a.waypoints, a.waypoints[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-9)
                return (x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)
        return a.waypoints[-1][1:] if t > a.waypoints[-1][0] else a.waypoints[0][1:]

    def is_hidden(self, name: str, t: float) -> bool:
        a = next(x for x in self.animals if x.name == name)
        return any(w0 <= t <= w1 for w0, w1 in a.hidden)

    def gt_points(self) -> dict[str, list[tuple[int, float, float]]]:
        """Ground-truth positions per analyzed frame (skipping hidden)."""
        out = {a.name: [] for a in self.animals}
        n = int(self.duration_s * self.fps)
        for f in range(n):
            t = f / self.fps
            for a in self.animals:
                if self.is_hidden(a.name, t):
                    continue
                pos = self.position(a.name, t)
                out[a.name].append((f, pos[0], pos[1]))
        return out


def _waypoints(*pts) -> list[tuple[float, float, float]]:
    return [(float(t), float(x), float(y)) for t, x, y in pts]


def default_scenes() -> list[Scene]:
    scenes = []

    # crossing: A left->right, B right->left, meet in the middle
    scenes.append(Scene(
        name="crossing", duration_s=8, animals=[
            GTAnimal("A", _waypoints((0, 100, 270), (8, 860, 270))),
            GTAnimal("B", _waypoints((0, 860, 270), (8, 100, 270)))]))
    # collision-merge: two animals approach, overlap into one blob, separate
    scenes.append(Scene(
        name="collision", duration_s=8, animals=[
            GTAnimal("A", _waypoints((0, 100, 270), (4, 470, 270), (8, 860, 270))),
            GTAnimal("B", _waypoints((0, 840, 270), (4, 490, 270), (8, 100, 270)))]))
    # stationary pause: walk, stop 2s, walk
    scenes.append(Scene(
        name="stationary", duration_s=8, animals=[
            GTAnimal("A", _waypoints((0, 100, 270), (3, 400, 270),
                                     (5, 400, 270), (8, 800, 270)))]))
    # occlusion: animal passes behind a leaf, gap in the middle
    occ = Scene(name="occlusion", duration_s=8, w=960, h=540,
                occluder=(420, 200, 120, 140), animals=[
                    GTAnimal("A", _waypoints((0, 100, 270), (8, 860, 270)),
                             hidden=[(3.2, 4.6)])])
    occ.animals[0].hidden = [(3.2, 4.6)]
    scenes.append(occ)
    # fast animal
    scenes.append(Scene(
        name="fast", duration_s=4, animals=[
            GTAnimal("A", _waypoints((0, 80, 270), (2, 880, 270),
                                     (2.6, 800, 270), (4, 100, 270)))]))
    return scenes


def render_frames(scene: Scene, seed: int = 0):
    """Render the scene to grayscale frames with ground truth positions
    (one blob per visible animal; overlapping animals merge into one blob —
    exactly what a real camera shows)."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(
        (rng.normal(0.5, 0.12, (scene.h, scene.w)).clip(0, 1) * 255
         ).astype(np.uint8), (0, 0), 3)
    n = int(scene.duration_s * scene.fps)
    for f in range(n):
        t = f / scene.fps
        frame = bg.copy()
        blobs = []
        for a in scene.animals:
            if scene.is_hidden(a.name, t):
                continue
            pos = scene.position(a.name, t)
            blobs.append((round(pos[0]), round(pos[1]), a.radius))
        # merge overlapping blobs into one filled silhouette
        layer = np.zeros((scene.h, scene.w), np.uint8)
        for x, y, r in blobs:
            cv2.circle(layer, (int(x), int(y)), int(r), 255, -1)
        frame[layer > 0] = (frame[layer > 0] * 0.35).astype(np.uint8)
        frame = np.clip(frame.astype(np.float32) +
                        rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
        yield f, frame, blobs, layer


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _match_gt(episode_frames: list[tuple[int, float, float]],
              gt: dict[str, list[tuple[int, float, float]]]
              ) -> tuple[str | None, float, float]:
    """Nearest-GT matching for one episode. Returns (animal, purity,
    mean position error)."""
    index = {}
    for name, pts in gt.items():
        for f, x, y in pts:
            index.setdefault(f, []).append((name, x, y))
    votes: dict[str, list[float]] = {}
    for f, x, y in episode_frames:
        cands = index.get(f)
        if not cands:
            continue
        name, gx, gy = min(cands, key=lambda c: (c[1] - x) ** 2 + (c[2] - y) ** 2)
        votes.setdefault(name, []).append(float(np.hypot(gx - x, gy - y)))
    if not votes:
        return None, 0.0, float("inf")
    best = max(votes, key=lambda k: len(votes[k]))
    purity = len(votes[best]) / sum(len(v) for v in votes.values())
    err = float(np.mean(votes[best]))
    return best, purity, err


def evaluate_scenes(scenes: list[Scene], cfg: VisionConfig,
                    video_dir=None, purity_min: float = 0.95) -> dict:
    """Run the vision frontend over the scenes and compute KineMimic
    metrics against exact GT."""
    total_eps = 0
    n_contaminated = 0
    frag_extra = 0
    errs = []
    covered_s = 0.0
    gt_movement_s = 0.0
    per_scene = []
    tmp_root = Path(video_dir or (Path(tempfile_dir()) / "km_bench"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    for si, scene in enumerate(scenes):
        # render to a temp video (the frontend consumes real videos)
        vpath = tmp_root / f"scene_{scene.name}.avi"
        vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"MJPG"),
                             scene.fps, (scene.w, scene.h), isColor=False)
        for f, frame, blobs, layer in render_frames(scene):
            vw.write(frame)
        vw.release()

        run = run_vision_frontend(str(vpath), f"bench_{scene.name}", cfg)
        gt = scene.gt_points()
        gt_movement_s += sum(len(p) for p in gt.values()) / scene.fps

        scene_purities, scene_errs = [], []
        n_animals_hit = set()
        for ep in run["episodes"]:
            pts = list(zip(ep.frames,
                           [c[0] for c in ep.centroids_px],
                           [c[1] for c in ep.centroids_px]))
            animal, purity, err = _match_gt(pts, gt)
            total_eps += 1
            scene_purities.append(purity)
            if animal:
                errs.append(err)
            if purity < 0.9:
                n_contaminated += 1
            if purity >= purity_min:
                covered_s += ep.duration_s
                n_animals_hit.add(animal)
        frag_extra += max(0, len(run["episodes"]) - len(scene.animals))
        gt_movement_s -= 0  # already accumulated above
        per_scene.append({"scene": scene.name, "n_episodes": len(run["episodes"]),
                          "mean_purity": round(float(np.mean(scene_purities)), 3)
                          if scene_purities else 0.0})

    usable_recall = covered_s / max(gt_movement_s, 1e-9)
    false_merge_rate = n_contaminated / max(total_eps, 1)
    fragmentation = frag_extra / max(len(scenes), 1)
    metrics = {
        "n_episodes": total_eps,
        "episode_purity_mean": round(float(np.mean(
            [s["mean_purity"] for s in per_scene])) if per_scene else 0.0, 3),
        "usable_episode_recall": round(usable_recall, 3),
        "false_merge_rate": round(false_merge_rate, 3),
        "fragmentation_extra_per_scene": round(fragmentation, 2),
        "trajectory_error_px_mean": round(float(np.mean(errs)), 2) if errs else None,
        "cost_weighted": round(5 * false_merge_rate + 0.5 * fragmentation
                               + (1 - usable_recall), 3),
        "cost_note": "5x false-merge penalty: a merged pair is far worse "
                     "than a split episode (docs/TRACKING.md)",
        "per_scene": per_scene,
    }
    return metrics


def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()


def run_benchmark(out_path: str | Path | None = None,
                  mode: str = "fast", detector: str = "legacy",
                  purity_min: float = 0.95) -> dict:
    """Compare complete pipelines on the same scenes (§51)."""
    scenes = default_scenes()
    results = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "mode": mode, "detector": detector, "pipelines": []}
    pipelines = {
        "A_legacy_bgdiff+legacy-greedy": None,   # handled via old ingest below
        "B_legacy_bgdiff+twostage": VisionConfig(
            mode=mode, detector="legacy",
            tracker_cfg=TrackerConfig(source_backend="twostage")),
    }
    for name, cfg in pipelines.items():
        if cfg is None:
            from ..pipeline import ingest_video
            from ..store import EpisodeStore
            t0 = time.time()
            per = []
            store = EpisodeStore(Path(tempfile_dir()) / f"km_bench_{abs(hash(name)) % 99999}")
            for scene in scenes:
                vpath = Path(tempfile_dir()) / f"km_bench_scene_{scene.name}.avi"
                vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"MJPG"),
                                     scene.fps, (scene.w, scene.h), isColor=False)
                if not vw.isOpened():
                    raise RuntimeError(f"cannot write {vpath}")
                for f, frame, blobs, layer in render_frames(scene):
                    vw.write(frame)
                vw.release()
                eps = ingest_video(str(vpath), store, f"bench_{scene.name}",
                                   min_duration_s=2.0)
                gt = scene.gt_points()
                purities = []
                for ep in eps:
                    pts = list(zip(ep.frames, [c[0] for c in ep.centroids_px],
                                   [c[1] for c in ep.centroids_px]))
                    animal, purity, err = _match_gt(pts, gt)
                    purities.append(purity)
                per.append({"scene": scene.name, "n_episodes": len(eps),
                            "mean_purity": round(float(np.mean(purities)), 3)
                            if purities else 0.0})
            # same metric battery as pipeline B (apples to apples)
            all_eps = []
            for scene in scenes:
                eps = ingest_video(str(Path(tempfile_dir()) /
                                       f"km_bench_scene_{scene.name}.avi"),
                                   store, f"bench_{scene.name}", min_duration_s=2.0)
                for ep in eps:
                    pts = list(zip(ep.frames, [c[0] for c in ep.centroids_px],
                                   [c[1] for c in ep.centroids_px]))
                    animal, purity, err = _match_gt(pts, scene.gt_points())
                    all_eps.append((ep, animal, purity, err))
            n_cont = sum(1 for _, _, pu, _ in all_eps if pu < 0.9)
            pure_s = sum(ep.duration_s for ep, _, pu, _ in all_eps if pu >= purity_min)
            frag = max(0, len(all_eps) - sum(len(s.animals) for s in scenes))
            errs = [e for _, _, _, e in all_eps if np.isfinite(e)]
            covered = pure_s
            gt_s = sum(sum(len(pts) for pts in scene.gt_points().values()) / scene.fps
                       for scene in scenes)
            fm_rate = n_cont / max(len(all_eps), 1)
            results["pipelines"].append({
                "pipeline": name,
                "note": "legacy ingest path (background subtraction + greedy)",
                "per_scene": per,
                "n_episodes": len(all_eps),
                "episode_purity_mean": round(float(np.mean([pu for _, _, pu, _ in all_eps])), 3)
                                       if all_eps else 0.0,
                "usable_episode_recall": round(covered / max(gt_s, 1e-9), 3),
                "false_merge_rate": round(fm_rate, 3),
                "fragmentation_extra_per_scene": round(frag / max(len(scenes), 1), 2),
                "trajectory_error_px_mean": round(float(np.mean(errs)), 2) if errs else None,
                "cost_weighted": round(5 * fm_rate + 0.5 * (frag / max(len(scenes), 1))
                                       + (1 - covered / max(gt_s, 1e-9)), 3),
                "runtime_s": round(time.time() - t0, 1)})
            continue
        t0 = time.time()
        m = evaluate_scenes(scenes, cfg)
        m["pipeline"] = name
        m["runtime_s"] = round(time.time() - t0, 1)
        results["pipelines"].append(m)
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results
