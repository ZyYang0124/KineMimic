"""Murmur command-line interface.

    python -m murmur demo --out runs_demo        # full tour on synthetic videos
    python -m murmur ingest VIDEO [VIDEO...] --id ID... --store runs
    python -m murmur viz runs/run_xxx/episodes.jsonl --out viz.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .atlas import build_atlas
from .pipeline import ingest_video, analyze
from .store import EpisodeStore
from .synth import generate_paths, render_video
from .viz import export_murmuration


def _print_summary(run_dir: Path):
    import json
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"run: {run_dir}")
    print(f"  episodes: {s['n_episodes']}  videos: {s['n_videos']}")
    print(f"  labels:   {s['labels']}")
    fp = s["mimicry_fingerprint"]["fingerprint"]
    for dim, v in fp.items():
        print(f"  mimicry[{dim}] Siler↔Ant overlap = {v:.2f}")


def cmd_demo(args):
    store = EpisodeStore(args.out)
    all_eps = []
    for i, arch in enumerate(("ant", "ant", "siler", "siler")):
        t0 = time.time()
        n = int(args.seconds * 30)
        paths = generate_paths(n, 960, 540, arch, n_animals=2, seed=100 + i)
        vid = str(Path(args.out) / "videos" / f"demo_{arch}_{i}.avi")
        Path(vid).parent.mkdir(parents=True, exist_ok=True)
        render_video(paths, arch, vid, seed=100 + i)
        eps = ingest_video(vid, store, f"demo_{arch}_{i}",
                           min_duration_s=args.min_episode_s,
                           environment=dict(site="synthetic", field_or_lab="sim",
                                            substrate="bark", fps=30))
        print(f"  {vid}: {len(eps)} episodes ({time.time()-t0:.0f}s)")
        all_eps += eps
    run_dir = analyze(store, all_eps, n_motifs=args.n_motifs)
    atlas = build_atlas(all_eps, run_dir / "atlas")
    _print_summary(run_dir)
    print(f"  atlas:          {atlas}  <- open in browser")


def cmd_ingest(args):
    store = EpisodeStore(args.store)
    all_eps = []
    for vid, vid_id in zip(args.video, args.id):
        all_eps += ingest_video(vid, store, vid_id, min_duration_s=args.min_episode_s)
    run_dir = analyze(store, all_eps, n_motifs=args.n_motifs)
    atlas = build_atlas(all_eps, run_dir / "atlas")
    _print_summary(run_dir)
    print(f"  atlas:          {atlas}  <- open in browser")


def cmd_atlas(args):
    from .store import EpisodeStore as ES
    eps = ES.read_episodes(args.episodes)
    out = build_atlas(eps, args.out)
    print(f"atlas: {out} ({len(eps)} episodes)")


def cmd_viz(args):
    from .store import EpisodeStore as ES
    eps = ES.read_episodes(args.episodes)
    out = export_murmuration(eps, args.out)
    print(f"wrote {out} ({len(eps)} episodes)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="murmur")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic end-to-end tour")
    d.add_argument("--out", default="murmur_runs")
    d.add_argument("--seconds", type=float, default=30, help="per video")
    d.add_argument("--min-episode-s", type=float, default=3.0)
    d.add_argument("--n-motifs", type=int, default=6)
    d.set_defaults(fn=cmd_demo)

    g = sub.add_parser("ingest", help="process real videos")
    g.add_argument("video", nargs="+")
    g.add_argument("--id", nargs="+", required=True)
    g.add_argument("--store", default="murmur_runs")
    g.add_argument("--min-episode-s", type=float, default=3.0)
    g.add_argument("--n-motifs", type=int, default=8)
    g.set_defaults(fn=cmd_ingest)

    a = sub.add_parser("atlas", help="build explorable atlas from episodes.jsonl")
    a.add_argument("episodes")
    a.add_argument("--out", default="atlas")
    a.set_defaults(fn=cmd_atlas)

    v = sub.add_parser("viz", help="export murmuration HTML from episodes.jsonl")
    v.add_argument("episodes")
    v.add_argument("--out", default="murmuration.html")
    v.set_defaults(fn=cmd_viz)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
