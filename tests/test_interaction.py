"""Interaction Layer V0: synthetic scenes with theoretical expectations.

Each scene encodes a known geometry (parallel / approaching / separating /
stationary / crossing) and the test asserts the metric behaves as theory
predicts — plus units, alignment, annotation filtering, provenance,
sampling hierarchy and null-model behavior.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinemimic.interaction import (SceneWindow, analyze_run, build_scene_windows,
                                     circular_mean_deg, concurrent_groups,
                                     neighbor_context, pairwise_geometry, write_outputs)
from kinemimic.schema import Episode


def mk_ep(eid, frames_xy, fps=30.0, video="v1", label=None, human=False,
          px_per_cm=None, site="siteA", session="d1"):
    frames = sorted(frames_xy)
    xy = [list(frames_xy[f]) for f in frames]
    ep = Episode(episode_id=eid, source_video_id=video,
                 source_video_path=f"mem:{video}", start_frame=frames[0],
                 end_frame=frames[-1], fps=fps, frames=frames,
                 centroids_px=xy, detection_confidence=[1.0] * len(frames),
                 px_per_cm=px_per_cm,
                 site_id=site, session_id=f"{site}/{session}")
    if label:
        if human:
            ep.human_label, ep.human_source, ep.annotator = label, "human", "t"
            ep.annotation_status = "accepted"
        else:
            ep.machine_label, ep.machine_source = label, "model:test"
        ep.bio_label = label
    ep.trajectory_features = {"speed_mean": 1.0, "turn_rate_mean": 0.5,
                              "frac_time_moving": 0.8, "speed_cv": 0.3}
    return ep


def line(t, p0, p1, f0=0):
    """positions along a straight segment, one point per frame."""
    n = len(t)
    return {f0 + i: (p0[0] + (p1[0] - p0[0]) * i / (n - 1),
                     p0[1] + (p1[1] - p0[1]) * i / (n - 1)) for i in range(n)}


def test_parallel_movement_aligns():
    """Two animals moving in parallel: distance constant, headings equal,
    alignment ~1, closing rate ~0."""
    a = mk_ep("a", line(range(60), (0, 0), (59, 0)))
    b = mk_ep("b", line(range(60), (0, 50), (59, 50)))
    rec = pairwise_geometry(a, b, "siler", "ant")
    assert rec is not None and rec.units == "px"
    assert abs(rec.distance_mean - 50) < 1.5
    assert abs(rec.distance_max - rec.distance_min) < 1.5
    assert rec.alignment > 0.99
    assert rec.heading_difference_mean_deg < 2
    assert abs(rec.closing_rate_mean) < 0.2
    assert rec.focal_speed_mean > 0


def test_approaching_then_separating():
    """Approaching: positive closing rate, minimum distance at the end of
    the shared interval. Separating: negative."""
    a = mk_ep("a", line(range(60), (0, 0), (59, 0)))
    b_in = mk_ep("b", line(range(60), (0, 200), (59, 50)))     # moving toward
    rec = pairwise_geometry(a, b_in, "siler", "ant")
    assert rec.closing_rate_mean > 0
    assert abs(rec.distance_min - 50) < 3
    b_out = mk_ep("c", line(range(60), (0, 50), (59, 200)))    # moving away
    rec2 = pairwise_geometry(a, b_out, "siler", "ant")
    assert rec2.closing_rate_mean < 0


def test_stationary_neighbor_speed_zero():
    a = mk_ep("a", line(range(60), (0, 0), (59, 0)))
    b = mk_ep("b", {f: (100, 100) for f in range(60)})
    rec = pairwise_geometry(a, b, "siler", "ant")
    assert rec.neighbor_speed_mean < 1e-6
    assert rec.focal_speed_mean > 0


def test_crossing_min_distance_at_crossing():
    """Perpendicular paths crossing at the middle: the minimum distance is
    near zero and occurs near the middle of the interval."""
    a = mk_ep("a", line(range(61), (0, 0), (60, 0)))
    b = mk_ep("b", line(range(61), (30, -30), (30, 30)))
    rec = pairwise_geometry(a, b, "siler", "ant")
    assert rec.distance_min < 2
    assert abs(rec.min_distance_time_s - 1.0) < 0.35   # crossing at t = 1s of 2s


def test_calibrated_units_cm():
    a = mk_ep("a", line(range(60), (0, 0), (59, 0)), px_per_cm=10.0)
    b = mk_ep("b", line(range(60), (0, 100), (59, 100)), px_per_cm=10.0)
    rec = pairwise_geometry(a, b, "siler", "ant")
    assert rec.units == "cm" and rec.px_per_cm == 10.0
    assert abs(rec.distance_mean - 10) < 0.2          # 100 px = 10 cm


def test_missing_frames_skipped_and_aligned():
    """Frames without a valid centroid are skipped; alignment is by frame
    index, not array position."""
    a_xy = line(range(60), (0, 0), (59, 0))
    b_xy = line(range(60), (0, 100), (59, 100))
    for f in range(10, 20):
        del b_xy[f]                                    # neighbor disappears
    b_xy[40] = (float("nan"), 100.0)                   # single bad sample
    a, b = mk_ep("a", a_xy), mk_ep("b", b_xy)
    rec = pairwise_geometry(a, b, "siler", "ant")
    assert rec.n_frames_aligned == 60 - 10 - 1         # gap + NaN removed
    assert abs(rec.distance_mean - 100) < 2


def test_concurrent_groups_exclude_velocity_episodes():
    a = mk_ep("a", line(range(60), (0, 0), (59, 0)))
    b = mk_ep("b", line(range(60), (0, 50), (59, 50)))
    vel = mk_ep("v", {0: (1.0, 1.0)})
    vel.centroids_px = []                              # velocity episode: no xy
    vel.frames = list(range(240))
    g = concurrent_groups([a, b, vel])
    assert len(g) == 1 and len(list(g.values())[0]) == 2


def test_scene_windows_counts_and_hierarchy():
    a = mk_ep("a", {f: (f, 0) for f in range(120)}, label="siler")
    b = mk_ep("b", {f: (f, 50) for f in range(120)}, label="ant", human=True)
    c = mk_ep("c", {f: (f, 90) for f in range(120)}, label="ant", human=True)
    wins = build_scene_windows([a, b, c], "v1", window_s=1.0, stride_s=1.0, labels={})
    assert wins and wins[0].animal_count == 3
    assert wins[0].taxon_counts == {"siler": 1, "ant": 2}
    assert wins[0].site_id == "siteA" and "siteA/d1" in wins[0].session_id
    assert wins[0].source_video_id == "v1"
    assert wins[0].end_time_s - wins[0].start_time_s == 1.0
    assert all(isinstance(w, SceneWindow) for w in wins)


def test_neighbor_context_multi_ant():
    """3 ants at different bearings/distances: nearest selection, count
    within radius, circular mean heading."""
    focal = mk_ep("f", {f: (100 * f / 59, 0) for f in range(60)}, label="siler")
    n1 = mk_ep("n1", {f: (100 * f / 59, 40) for f in range(60)}, label="ant", human=True)  # near, heading +x
    n2 = mk_ep("n2", {f: (100 * f / 59, 80) for f in range(60)}, label="ant", human=True)  # within 200 px
    n3 = mk_ep("n3", {f: (100 * f / 59, 500) for f in range(60)}, label="ant", human=True) # far
    labels = {"f": "siler", "n1": "ant", "n2": "ant", "n3": "ant"}
    ctx = neighbor_context(focal, [n1, n2, n3], labels, radius=200.0, fps=30.0)
    assert ctx["units"] == "px"
    assert ctx["n_ants_max"] == 2 and ctx["n_ants_mean"] > 1.5
    assert abs(ctx["nearest_ant_dist_min"] - 40) < 3
    assert ctx["has_ant"] and ctx["frac_frames_with_ant"] > 0.99
    # all nearby ants move in +x: circular mean heading ~ 0 deg
    assert abs((ctx["mean_ant_heading_deg"][0] + 180) % 360 - 180) < 10
    # distances to nearest ant are the n1 offset everywhere
    assert all(abs(d - 40) < 3 for d in ctx["nearest_ant_dist"] if d is not None)


def test_circular_mean_deg():
    assert abs(circular_mean_deg([350, 10]) % 360 - 0) < 1e-6   # wraps correctly
    assert abs(circular_mean_deg([90, 90]) - 90) < 1e-6


def test_annotation_filtering_require_human():
    """With require_human_labels, a machine-labeled ant is not an ant:
    no ant context is produced for the focal Siler."""
    focal = mk_ep("f", line(range(60), (0, 0), (59, 0)), label="siler", human=True)
    ant_m = mk_ep("m", line(range(60), (0, 30), (59, 30)), label="ant", human=False)
    labels = {"f": "siler", "m": "unannotated"}
    ctx = neighbor_context(focal, [ant_m], labels, radius=200.0, fps=30.0)
    assert not ctx["has_ant"] and ctx["n_ants_max"] == 0
    assert ctx["nearest_ant_dist_min"] is None


def test_analyze_run_end_to_end(tmp_path):
    """Full pipeline: records, windows, contexts, comparison with null."""
    # 12 siler: half with an ant marching right beside them, half alone
    rng = np.random.default_rng(11)
    eps = []
    for i in range(6):
        y = 100 * (i + 1)
        # with-ant Siler: ant beside it AND faster locomotion (synthetic effect)
        s = mk_ep(f"s{i}", line(range(90), (0, y), (178, y)), label="siler", human=True)
        # continuous values: a coarse two-level sample makes the median
        # shuffle null degenerate
        s.trajectory_features["speed_mean"] = 60.0 + float(rng.uniform(-4, 4))
        eps.append(s)
        eps.append(mk_ep(f"a{i}", line(range(90), (0, y + 30), (89, y + 30)),
                         label="ant", human=True))
    for i in range(6):
        s = mk_ep(f"s_far{i}", line(range(90), (0, 3000 + 40 * i), (89, 3000 + 40 * i)),
                  label="siler", human=True)
        s.trajectory_features["speed_mean"] = 30.0 + float(rng.uniform(-4, 4))
        eps.append(s)
    out = analyze_run(eps, window_s=1.0, stride_s=1.0, radius=200.0, n_shuffle=99)
    assert len(out["records"]) > 0
    assert out["windows"], "scene windows should be generated"
    assert len(out["contexts"]) == 18
    assert all(c["units"] == "px" for c in out["contexts"].values())
    cmp_ = out["comparison"]
    assert "speed" in cmp_, "strong synthetic separation should yield both groups"
    s = cmp_["speed"]
    assert s["n_with_ant"] == 6 and s["n_without_ant"] == 6
    assert s["p_perm"] <= 0.05, "clear synthetic effect must exceed the shuffle null"
    assert 0 <= s["p_perm"] <= 1
    # D_ant contrast present with reference distribution from 6 ants
    assert "d_ant" in cmp_ and cmp_["d_ant"].get("n_with_ant") == 6
    # provenance records every analysis parameter
    p = out["provenance"]["parameters"]
    assert p["window_s"] == 1.0 and p["radius"] == 200.0 and p["n_shuffle"] == 99
    assert "not ground truth" in p["labels_note"] or "human-confirmed" in p["labels_note"]
    # persistence round-trip
    files = write_outputs(tmp_path, out)
    assert files["interactions"].exists() and files["summary"].exists()
    loaded = json.loads(files["summary"].read_text(encoding="utf-8"))
    assert loaded["contexts"]["s0"]["has_ant"]
    rows = [json.loads(l) for l in open(files["interactions"], encoding="utf-8")]
    assert any(r["focal_episode_id"] == "s0" and r["neighbor_episode_id"] == "a0"
               for r in rows)


def test_scene_identity_preserved_in_hierarchy():
    """Two sites: scene windows and contexts must never merge them."""
    a = mk_ep("a", line(range(60), (0, 0), (59, 0)), site="siteX")
    b = mk_ep("b", line(range(60), (0, 50), (59, 50)), site="siteX")
    c = mk_ep("c", line(range(60), (0, 0), (59, 0)), site="siteY")
    d = mk_ep("d", line(range(60), (0, 50), (59, 50)), site="siteY")
    out = analyze_run([a, b, c, d], window_s=1.0, stride_s=1.0, radius=200.0)
    vids = {w.source_video_id for w in out["windows"]}
    assert vids == {"v1"}                      # same video id, but sites differ
    sites = {(w.site_id) for w in out["windows"]}
    assert sites == {"siteX", "siteY"}         # scene windows carry their site


def test_null_p_not_extreme_for_random_contexts():
    """When context is random relative to movement, the shuffle p should
    usually be non-significant (no fabricated interaction)."""
    rng = np.random.default_rng(3)
    eps = []
    for i in range(12):
        y = 500 * i
        eps.append(mk_ep(f"s{i}", line(range(90), (0, y), (89, y)),
                         label="siler", human=True))
        eps[-1].trajectory_features["speed_mean"] = float(rng.uniform(0, 1))
        if i % 2 == 0:   # ant next door; odd episodes stay >200px from any ant
            eps.append(mk_ep(f"a{i}", line(range(90), (0, y + 30), (89, y + 30)),
                             label="ant", human=True))
    out = analyze_run(eps, window_s=1.0, stride_s=1.0, radius=200.0, n_shuffle=99)
    s = out["comparison"]["speed"]
    assert 0 <= s["p_perm"] <= 1
    # with random speeds the observed diff is inside the null envelope
    lo, hi = s["shuffle_null_median_diff"]
    assert lo <= s["observed_median_diff"] <= hi


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(Path(f"/tmp/ms_test_{name}"))
            print(f"{name} OK")
