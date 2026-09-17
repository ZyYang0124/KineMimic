"""MOTIONSCAPE command-line interface.

    python -m motionscape demo                          # synthetic end-to-end tour
    python -m motionscape ingest VIDEO --id ID          # real videos -> run + atlas
    python -m motionscape annotate episodes.jsonl       # human QC + biological labels
    python -m motionscape atlas  episodes.jsonl         # (re)build the atlas
    python -m motionscape serve   <atlas_dir>           # atlas + on-demand media
    python -m motionscape benchmark --n 5000 10000      # scale benchmark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .atlas import build_atlas
from .shamble import load_shamble_episodes
from .zeng import load_zeng_episodes
from .pipeline import ingest_video, analyze
from .store import EpisodeStore
from .synth import generate_paths, render_video


def _print_summary(run_dir: Path):
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"run: {run_dir}")
    print(f"  episodes: {s['n_episodes']}  videos: {s['n_videos']}")
    ann = s.get("annotation", {})
    print(f"  reviewed: {ann.get('reviewed', 0)}/{ann.get('total', s['n_episodes'])}")
    print(f"  labels:   {s['labels']}")
    fp = s["mimicry_fingerprint"]["fingerprint"]
    for dim, v in fp.items():
        print(f"  mimicry[{dim}] Siler↔Ant overlap = {v:.2f}")
    return s


def _atlas_meta(run_dir: Path, s: dict) -> dict:
    return {"mimicry_fingerprint": s.get("mimicry_fingerprint", {}),
            "annotation": s.get("annotation", {}),
            "provenance": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))}


def _finish(store: EpisodeStore, all_eps, run_dir: Path, n_motifs: int):
    atlas = build_atlas(all_eps, run_dir / "atlas",
                        summary=json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
                        run_provenance=json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")))
    _print_summary(run_dir)
    print(f"  next:     python -m motionscape annotate {run_dir / 'episodes.jsonl'}")
    print(f"  atlas:    {atlas}")
    print(f"  open:     python -m motionscape serve {run_dir / 'atlas'}")


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
    _finish(store, all_eps, run_dir, args.n_motifs)


def cmd_shamble(args):
    """Real-data showcase: Shamble 2017 trajectories -> episodes -> atlas."""
    eps = load_shamble_episodes(args.mat, video_dir=args.videos)
    store = EpisodeStore(args.store)
    run_dir = analyze(store, eps, n_motifs=args.n_motifs)
    _finish(store, eps, run_dir, args.n_motifs)


def cmd_zeng(args):
    """Gait universe: Zeng 2023 Siler collingwoodi velocity/pose episodes."""
    eps = load_zeng_episodes(args.xlsx)
    store = EpisodeStore(args.store)
    run_dir = analyze(store, eps, n_motifs=args.n_motifs)
    _finish(store, eps, run_dir, args.n_motifs)


def cmd_ingest(args):
    store = EpisodeStore(args.store)
    all_eps = []
    for vid, vid_id in zip(args.video, args.id):
        eps = ingest_video(vid, store, vid_id, min_duration_s=args.min_episode_s,
                           target_fps=args.target_fps,
                           environment=dict(site=args.site) if args.site else None)
        for ep in eps:
            ep.site_id = args.site or ep.site_id
            ep.session_id = args.session or ep.session_id
        all_eps += eps
    run_dir = analyze(store, all_eps, n_motifs=args.n_motifs)
    _finish(store, all_eps, run_dir, args.n_motifs)


def cmd_annotate(args):
    from .workbench import run_workbench
    run_workbench(args.episodes, store_root=args.store, port=args.port)


def cmd_serve(args):
    from .serve import serve_atlas
    serve_atlas(args.atlas, episodes_path=args.episodes, port=args.port)


def cmd_atlas(args):
    from .store import EpisodeStore as ES
    eps = ES.read_episodes(args.episodes)
    sfile = Path(args.episodes).parent / "summary.json"
    s = json.loads(sfile.read_text(encoding="utf-8")) if sfile.exists() else {}
    out = build_atlas(eps, args.out, summary=s or None)
    print(f"atlas: {out} ({len(eps)} episodes)")
    print(f"open:  python -m motionscape serve {args.out} --episodes {args.episodes}")


def cmd_viz(args):
    from .store import EpisodeStore as ES
    from .viz import export_murmuration
    eps = ES.read_episodes(args.episodes)
    out = export_murmuration(eps, args.out)
    print(f"wrote {out} ({len(eps)} episodes)")


def cmd_benchmark(args):
    from .benchmark import run as run_bench
    print("MOTIONSCAPE atlas benchmark (synthetic velocity-style episodes)")
    run_bench(args.n, out_dir=args.out, seed=args.seed)


def main(argv=None):
    p = argparse.ArgumentParser(prog="motionscape")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic end-to-end tour")
    d.add_argument("--out", default="motionscape_runs")
    d.add_argument("--seconds", type=float, default=30, help="per video")
    d.add_argument("--min-episode-s", type=float, default=3.0)
    d.add_argument("--n-motifs", type=int, default=6)
    d.set_defaults(fn=cmd_demo)

    sh = sub.add_parser("shamble", help="build atlas from Shamble 2017 Dryad data")
    sh.add_argument("--mat", default="data/external/shamble2017/OverallMovement/OverallMovement/data/data_folders_5_to_18_v3.mat")
    sh.add_argument("--videos", default="data/external/shamble2017/OverallMovement/OverallMovement/example videos")
    sh.add_argument("--store", default="motionscape_runs")
    sh.add_argument("--n-motifs", type=int, default=8)
    sh.set_defaults(fn=cmd_shamble)

    z = sub.add_parser("zeng", help="gait universe from Zeng 2023 raw gait data")
    z.add_argument("--xlsx", default="data/external/Zeng_et_al_2023_Gait analysis_raw data.xlsx")
    z.add_argument("--store", default="motionscape_runs/zeng")
    z.add_argument("--n-motifs", type=int, default=8)
    z.set_defaults(fn=cmd_zeng)

    g = sub.add_parser("ingest", help="process real videos")
    g.add_argument("video", nargs="+")
    g.add_argument("--id", nargs="+", required=True)
    g.add_argument("--store", default="motionscape_runs")
    g.add_argument("--site", default=None, help="sampling-hierarchy site id")
    g.add_argument("--session", default=None, help="sampling-hierarchy session id")
    g.add_argument("--min-episode-s", type=float, default=3.0)
    g.add_argument("--n-motifs", type=int, default=8)
    g.add_argument("--target-fps", type=float, default=30.0,
                   help="analyze above this fps by frame skipping (faster)")
    g.set_defaults(fn=cmd_ingest)

    a = sub.add_parser("annotate", help="human QC + biological annotation workbench")
    a.add_argument("episodes", help="path to episodes.jsonl (a run directory)")
    a.add_argument("--store", default=None, help="store root (auto-detected)")
    a.add_argument("--port", type=int, default=8692)
    a.set_defaults(fn=cmd_annotate)

    at = sub.add_parser("atlas", help="build explorable atlas from episodes.jsonl")
    at.add_argument("episodes")
    at.add_argument("--out", default="atlas")
    at.set_defaults(fn=cmd_atlas)

    sv = sub.add_parser("serve", help="serve the atlas with on-demand media")
    sv.add_argument("atlas", help="atlas directory (contains index.html + data.json)")
    sv.add_argument("--episodes", default=None, help="episodes.jsonl for clip generation")
    sv.add_argument("--port", type=int, default=8694)
    sv.set_defaults(fn=cmd_serve)

    v = sub.add_parser("viz", help="export murmuration HTML from episodes.jsonl")
    v.add_argument("episodes")
    v.add_argument("--out", default="murmuration.html")
    v.set_defaults(fn=cmd_viz)

    b = sub.add_parser("benchmark", help="atlas scale benchmark")
    b.add_argument("--n", type=int, nargs="+", default=[1000, 5000, 10000, 20000])
    b.add_argument("--out", default="benchmarks")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_benchmark)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
