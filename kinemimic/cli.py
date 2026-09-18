"""KineMimic command-line interface.

    python -m kinemimic demo                          # synthetic end-to-end tour
    python -m kinemimic ingest VIDEO --id ID          # real videos -> run + atlas
    python -m kinemimic annotate episodes.jsonl       # human QC + biological labels
    python -m kinemimic atlas  episodes.jsonl         # (re)build the atlas
    python -m kinemimic serve   <atlas_dir>           # atlas + on-demand media
    python -m kinemimic benchmark --n 5000 10000      # scale benchmark
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
    print(f"  next:     python -m kinemimic annotate {run_dir / 'episodes.jsonl'}")
    print(f"  atlas:    {atlas}")
    print(f"  open:     python -m kinemimic serve {run_dir / 'atlas'}")


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
    serve_atlas(args.atlas, episodes_path=args.episodes, reference_dir=args.reference,
                port=args.port)


def cmd_atlas(args):
    from .store import EpisodeStore as ES
    from .interaction import load_summary
    eps = ES.read_episodes(args.episodes)
    sfile = Path(args.episodes).parent / "summary.json"
    s = json.loads(sfile.read_text(encoding="utf-8")) if sfile.exists() else {}
    ia = load_summary(Path(args.episodes).parent)
    out = build_atlas(eps, args.out, summary=s or None, interaction=ia)
    print(f"atlas: {out} ({len(eps)} episodes"
          + (f", interaction: {ia['n_records']} pairs)" if ia else ")"))
    print(f"open:  python -m kinemimic serve {args.out} --episodes {args.episodes}")


def cmd_viz(args):
    from .store import EpisodeStore as ES
    from .viz import export_murmuration
    eps = ES.read_episodes(args.episodes)
    out = export_murmuration(eps, args.out)
    print(f"wrote {out} ({len(eps)} episodes)")


def cmd_interact(args):
    """Interaction Layer V0: scene windows, pairwise geometry, nearest-ant
    context, Siler with/without-ant comparison (with shuffle null)."""
    from pathlib import Path as _P
    import json as _json
    from .interaction import analyze_run, write_outputs
    from .store import EpisodeStore as ES
    p = _P(args.episodes)
    run_dir = p.parent if p.suffix == ".jsonl" else p
    eps = ES.read_episodes(p if p.suffix == ".jsonl" else p / "episodes.jsonl")
    result = analyze_run(eps, window_s=args.window_s, stride_s=args.stride_s,
                         radius=args.radius,
                         require_human_labels=args.require_human,
                         n_shuffle=args.n_shuffle)
    out = write_outputs(run_dir, result)
    print(f"interaction analysis: {run_dir}")
    print(f"  concurrent pairs:    {len(result['records'])}")
    print(f"  scene windows:       {len(result['windows'])}")
    print(f"  episodes with scene: {len(result['contexts'])}"
          f"  (velocity-only excluded: {result['n_velocity_excluded']})")
    print(f"  videos w/ concurrency: {result['n_videos_with_concurrency']}")
    cmp_ = result["comparison"]
    if "speed" in cmp_:
        s = cmp_["speed"]
        print(f"  Siler speed median  with ants: {s['median_with_ant']}  "
              f"without: {s['median_without_ant']}  (p_perm={s.get('p_perm')})")
    if "d_ant" in cmp_ and "median_with_ant" in cmp_["d_ant"]:
        d = cmp_["d_ant"]
        print(f"  D_ant median        with ants: {d['median_with_ant']}  "
              f"without: {d['median_without_ant']}  (p_perm={d.get('p_perm')})")
    for row in cmp_.get("response", []):
        parts = [f"radius {row['radius']}: n {row['n_with']}/{row['n_without']}"]
        for m in ("speed", "turning", "stop_go", "d_ant"):
            if m in row:
                e = row[m]
                parts.append(f"{m} Δ {e['observed_median_diff']} (p={e['p_perm']})")
        print("  response | " + " · ".join(parts))
    if "speed" not in cmp_ and "response" not in cmp_:
        print("  Siler x ant comparison: not enough annotated episodes in both contexts")
    print(f"  note: {result['provenance']['parameters']['labels_note']}")
    print(f"  next: python -m kinemimic atlas {run_dir / 'episodes.jsonl'} "
          f"--out {run_dir / 'atlas_v2'}  (Interaction Mode appears automatically)")


def cmd_build_reference(args):
    """Build a versioned Reference Atlas Index from one or more runs."""
    from .query import build_reference
    from .retrieval import IndexPolicy
    from .store import EpisodeStore as ES
    eps = []
    for p in args.episodes:
        eps += ES.read_episodes(p)
    policy = IndexPolicy(name=args.policy, min_duration_s=args.min_duration_s,
                         require_human_label=args.require_human)
    out = build_reference(eps, args.out, dim=args.dim, policy=policy,
                          previous_version=args.parent)
    idx = out["index"]
    print(f"reference index: {idx.version_id}")
    print(f"  episodes: {out['n_kept']} kept / {out['n_input']} input "
          f"(policy: {policy.name}, min {args.min_duration_s}s)")
    print(f"  encoders: {', '.join(out['encoders'])} (dim={args.dim})")
    print(f"  taxa: {getattr(idx, 'entries_meta', {}).get('taxon_counts', {})}")
    print(f"  dir: {Path(args.out) / idx.version_id}")


def cmd_find_similar(args):
    """Drop in a video. See what moves like it."""
    from .interaction import load_summary  # noqa: F401 (parity import guard)
    from .query import run_query
    detector_params = {}
    if args.min_area is not None:
        detector_params["min_area"] = args.min_area
    if args.threshold is not None:
        detector_params["threshold"] = args.threshold
    if args.no_morph_open:
        detector_params["morph_open_k"] = 0
    bundle = run_query(args.video, args.query_id, args.reference,
                       store_dir=args.store, out_dir=args.out,
                       atlas_dir=args.atlas, k=args.k, metric=args.metric,
                       target_fps=args.target_fps,
                       min_duration_s=args.min_duration_s,
                       detector_params=detector_params or None,
                       tracker_max_gap=args.max_gap,
                       vision_mode=args.vision_mode,
                       vision_detector=args.detector)
    if bundle.get("error"):
        print(f"query failed: {bundle['error']}")
        return
    print(f"find-similar: {args.query_id} ({bundle['build_seconds']}s)")
    print(f"  reference: {bundle['reference_atlas_version']} "
          f"(metric={args.metric}, k={args.k})")
    print(f"  query episodes: {len(bundle['episodes'])} "
          f"({sum(1 for q in bundle['qc'] if q['usable'])} usable)")
    r = bundle["results"]
    for h in r["episode_hits"][:3]:
        top = h["neighbors"][0] if h["neighbors"] else None
        if top:
            print(f"  {h['query_episode_id'][:18]} -> {top['episode_id'][:18]} "
                  f"[{top['taxon']}] sim {top['similarity']:.2f}")
    for m in r["motif_hits"][:3]:
        print(f"  motif M{m['motif']} sim {m['similarity']:.2f}")
    for t in r["taxa"][:3]:
        print(f"  taxa {t['taxon']} (n={t['n_reference_episodes']}) "
              f"score {t['score_corrected']:.2f} "
              f"[centroid d {t['centroid_distance']:.2f} · wasserstein {t['distribution_wasserstein']:.3f}]")
    if r["ood"]["any_flagged"]:
        print(f"  ⚠ OOD (physical space): {r['ood']['message']}")
    rb = bundle.get("results_representation_b")
    if rb:
        for h in rb["episode_hits"][:3]:
            if h["neighbors"]:
                n0 = h["neighbors"][0]
                print(f"  [shape] {h['query_episode_id'][:18]} -> "
                      f"{n0['episode_id'][:18]} [{n0['taxon']}] "
                      f"sim {n0['similarity']:.2f}")
        for t2 in rb.get("taxa", [])[:3]:
            print(f"  [shape] taxa {t2['taxon']} (n={t2['n_reference_episodes']}) "
                  f"score {t2['score_corrected']:.2f}")
        if rb.get("ood", {}).get("any_flagged"):
            print("  ⚠ OOD (shape space): query outside well-sampled region")
    if bundle.get("consistency") and bundle["consistency"].get("mean_jaccard") is not None:
        print(f"  representation agreement (A∩B Jaccard@k): "
              f"{bundle['consistency']['mean_jaccard']} "
              f"over {bundle['consistency']['n_compared']} episode(s)")
    for w in bundle.get("written", []):
        print(f"  wrote: {w}")


def cmd_eval_retrieval(args):
    """Retrieval benchmark: leave-video-out, confound + positive controls."""
    from .eval_retrieval import run_eval
    run_eval(args.reference, n_query=args.n_query, k=args.k, seed=args.seed)


def cmd_vision_doctor(args):
    """Vision environment probe: GPU, backends, recommendations."""
    from .vision.doctor import run_doctor
    run_doctor()


def cmd_vision(args):
    """Modern multi-animal vision frontend: detect -> track -> episodes."""
    import json as _json
    from .vision.pipeline import VisionConfig, run_vision_frontend, VisionQCFailed
    detector_params = {}
    if args.threshold is not None:
        detector_params["threshold"] = args.threshold
    if args.min_area is not None:
        detector_params["min_area"] = args.min_area
    if args.no_morph_open:
        detector_params["morph_open_k"] = 0
    cfg = VisionConfig(
        mode=args.mode, detector=args.detector, detector_params=detector_params,
        tracker_cfg=__import__("kinemimic.vision.tracker", fromlist=["TrackerConfig"]).TrackerConfig(
            max_gap_frames=args.max_gap),
        tile_w=args.tile, tile_h=args.tile, tile_overlap=args.overlap,
        inference_stride=args.stride, camera_stabilization=args.stabilize,
        min_duration_s=args.min_episode_s,
        roi=tuple(int(x) for x in args.roi.split(",")) if args.roi else None)
    try:
        run = run_vision_frontend(args.video, args.id, cfg)
    except VisionQCFailed as e:
        print(f"VISION QC FAILED: {e}")
        raise SystemExit(2)
    from .store import EpisodeStore
    store = EpisodeStore(args.store)
    from .pipeline import analyze
    run_dir = analyze(store, run["episodes"], n_motifs=args.n_motifs)
    s = _json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    atlas = build_atlas(run["episodes"], run_dir / "atlas",
                        summary=s,
                        run_provenance=_json.loads(
                            (run_dir / "manifest.json").read_text(encoding="utf-8")))
    print(f"vision frontend: {args.video}")
    st = run["stats"]
    print(f"  mode {args.mode} · detector {args.detector} · {st['n_frames_analyzed']} frames "
          f"({st['runtime_s']}s) · fps {st['fps']}")
    print(f"  tracklets: {st['n_tracklets']} -> episodes: {st['n_episodes']}")
    print(f"  fragmentation: median tracklet {st['median_tracklet_seconds']}s "
          f"(need >= {args.min_episode_s}s) · {st['detections_per_frame']} dets/frame")
    n_amb = sum(len(e.metadata.get("ambiguity_events", [])) for e in run["episodes"])
    print(f"  ambiguity events: {n_amb} (episodes carry them for QC)")
    print(f"  run: {run_dir}")
    print(f"  atlas: {atlas}")
    print(f"  next: python -m kinemimic annotate {run_dir / 'episodes.jsonl'}")


def cmd_vision_benchmark(args):
    from .vision.benchmark import run_benchmark
    r = run_benchmark(out_path=args.out, mode=args.mode, detector=args.detector)
    for p in r["pipelines"]:
        print(f"{p['pipeline']}: purity {p.get('episode_purity_mean')} · "
              f"recall {p.get('usable_episode_recall')} · false merge {p.get('false_merge_rate')} · "
              f"COST {p.get('cost_weighted')}")


def cmd_benchmark(args):
    from .benchmark import run as run_bench
    print("KineMimic atlas benchmark (synthetic velocity-style episodes)")
    run_bench(args.n, out_dir=args.out, seed=args.seed)


def main(argv=None):
    p = argparse.ArgumentParser(prog="kinemimic")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic end-to-end tour")
    d.add_argument("--out", default="kinemimic_runs")
    d.add_argument("--seconds", type=float, default=30, help="per video")
    d.add_argument("--min-episode-s", type=float, default=3.0)
    d.add_argument("--n-motifs", type=int, default=6)
    d.set_defaults(fn=cmd_demo)

    sh = sub.add_parser("shamble", help="build atlas from Shamble 2017 Dryad data")
    sh.add_argument("--mat", default="data/external/shamble2017/OverallMovement/OverallMovement/data/data_folders_5_to_18_v3.mat")
    sh.add_argument("--videos", default="data/external/shamble2017/OverallMovement/OverallMovement/example videos")
    sh.add_argument("--store", default="kinemimic_runs")
    sh.add_argument("--n-motifs", type=int, default=8)
    sh.set_defaults(fn=cmd_shamble)

    z = sub.add_parser("zeng", help="gait universe from Zeng 2023 raw gait data")
    z.add_argument("--xlsx", default="data/external/Zeng_et_al_2023_Gait analysis_raw data.xlsx")
    z.add_argument("--store", default="kinemimic_runs/zeng")
    z.add_argument("--n-motifs", type=int, default=8)
    z.set_defaults(fn=cmd_zeng)

    g = sub.add_parser("ingest", help="process real videos")
    g.add_argument("video", nargs="+")
    g.add_argument("--id", nargs="+", required=True)
    g.add_argument("--store", default="kinemimic_runs")
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
    sv.add_argument("--reference", default=None,
                    help="reference atlas dir: enables Find Similar upload")
    sv.add_argument("--port", type=int, default=8694)
    sv.set_defaults(fn=cmd_serve)

    v = sub.add_parser("viz", help="export murmuration HTML from episodes.jsonl")
    v.add_argument("episodes")
    v.add_argument("--out", default="murmuration.html")
    v.set_defaults(fn=cmd_viz)

    it = sub.add_parser("interact", help="Interaction Layer V0: scenes + pairwise context")
    it.add_argument("episodes", help="run directory or episodes.jsonl")
    it.add_argument("--window-s", type=float, default=2.0, help="scene window length (s)")
    it.add_argument("--stride-s", type=float, default=1.0, help="scene window stride (s)")
    it.add_argument("--radius", type=float, default=None,
                    help="neighbor radius (cm if calibrated, else px; default 200)")
    it.add_argument("--require-human", action="store_true",
                    help="use only human-confirmed labels for taxon context")
    it.add_argument("--n-shuffle", type=int, default=200, help="null shuffles")
    it.set_defaults(fn=cmd_interact)

    br = sub.add_parser("build-reference", help="build versioned Reference Atlas Index")
    br.add_argument("episodes", nargs="+", help="episodes.jsonl paths")
    br.add_argument("--out", default="reference_atlas")
    br.add_argument("--dim", type=int, default=10, help="behavior vector dimension")
    br.add_argument("--policy", default="not-rejected",
                    choices=["all", "not-rejected", "reviewed"])
    br.add_argument("--min-duration-s", type=float, default=3.0)
    br.add_argument("--require-human", action="store_true",
                    help="only human-confirmed episodes enter the index")
    br.add_argument("--parent", default=None, help="previous index version id")
    br.set_defaults(fn=cmd_build_reference)

    fs = sub.add_parser("find-similar", help="drop in a video, find what moves like it")
    fs.add_argument("video")
    fs.add_argument("--query-id", required=True)
    fs.add_argument("--reference", required=True, help="reference atlas dir (contains index.json)")
    fs.add_argument("--store", default=None, help="episode store for the query run")
    fs.add_argument("--out", default=None, help="dir for query.json")
    fs.add_argument("--atlas", default=None, help="atlas dir to inject query.json into")
    fs.add_argument("--k", type=int, default=6)
    fs.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    fs.add_argument("--target-fps", type=float, default=30.0)
    fs.add_argument("--min-duration-s", type=float, default=3.0)
    fs.add_argument("--min-area", type=int, default=None,
                    help="detector: min component area (tiny-animal footage)")
    fs.add_argument("--threshold", type=int, default=None,
                    help="detector: background-difference threshold")
    fs.add_argument("--no-morph-open", action="store_true",
                    help="detector: skip 3x3 opening (keeps few-pixel targets)")
    fs.add_argument("--max-gap", type=int, default=3,
                    help="tracker: missed frames a track may bridge")
    fs.add_argument("--vision-mode", default="legacy",
                    choices=["legacy", "fast", "accurate", "assisted"],
                    help="video frontend for the query (multi-animal capable)")
    fs.set_defaults(fn=cmd_find_similar)

    ev = sub.add_parser("eval-retrieval", help="retrieval benchmark (leave-video-out etc.)")
    ev.add_argument("reference")
    ev.add_argument("--n-query", type=int, default=60)
    ev.add_argument("--k", type=int, default=6)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(fn=cmd_eval_retrieval)

    vd = sub.add_parser("vision-doctor", help="GPU/dependency probe + recommendations")
    vd.set_defaults(fn=cmd_vision_doctor)

    v = sub.add_parser("vision", help="modern multi-animal vision frontend")
    v.add_argument("video")
    v.add_argument("--id", required=True)
    v.add_argument("--store", default="kinemimic_runs")
    v.add_argument("--mode", default="fast", choices=["fast", "accurate", "assisted"])
    v.add_argument("--detector", default="legacy", choices=["legacy", "yolo", "rfdetr"])
    v.add_argument("--tile", type=int, default=1024, help="tile size (accurate mode)")
    v.add_argument("--overlap", type=float, default=0.2)
    v.add_argument("--stride", type=int, default=1, help="analyze every Nth frame")
    v.add_argument("--max-gap", type=int, default=8, help="tracker coast window (frames)")
    v.add_argument("--stabilize", action="store_true",
                   help="global camera-motion compensation")
    v.add_argument("--threshold", type=int, default=None)
    v.add_argument("--min-area", type=int, default=None)
    v.add_argument("--no-morph-open", action="store_true")
    v.add_argument("--roi", default=None,
                   help="region of interest x,y,w,h (recorded in provenance)")
    v.add_argument("--min-episode-s", type=float, default=3.0)
    v.add_argument("--n-motifs", type=int, default=8)
    v.set_defaults(fn=cmd_vision)

    vb = sub.add_parser("vision-benchmark", help="KineMimic tracking benchmark")
    vb.add_argument("--mode", default="fast", choices=["fast", "accurate", "assisted"])
    vb.add_argument("--detector", default="legacy", choices=["legacy", "yolo", "rfdetr"])
    vb.add_argument("--out", default="benchmarks/tracking_benchmark.json")
    vb.set_defaults(fn=cmd_vision_benchmark)

    b = sub.add_parser("benchmark", help="atlas scale benchmark")
    b.add_argument("--n", type=int, nargs="+", default=[1000, 5000, 10000, 20000])
    b.add_argument("--out", default="benchmarks")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_benchmark)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
