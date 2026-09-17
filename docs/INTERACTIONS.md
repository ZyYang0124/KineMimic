# Interaction Layer (V0)

Layer 2 of KineMimic's science stack:

```
Movement     how does one animal move?              (episodes, features, space)
Interaction  how does movement depend on another?   (this document)
Behavior     how are movements organized?           (future — grammar, states)
```

KineMimic must not only split animals into isolated trajectories. Real
animals live inside a context of other animals, space and time. The
Interaction Layer puts every Movement Episode back into the world it came
from — without abandoning the episode as the unit of analysis.

## What the V0 milestone answers

Given a real field video, for any Siler episode:

- were ants nearby during this movement?
- how far was the nearest ant?
- how many ants were within the analysis radius?
- roughly where were they heading, and how active were they?
- did Siler approach or move away from them?
- does Siler's movement state differ between ant-nearby and ant-absent
  episodes?

Everything traces back to the same source video.

## Core objects

**SceneWindow** — a sliding window (configurable `window_s`, `stride_s`)
of one source video with every episode active in it: `scene_window_id`,
frame/time interval, `active_episode_ids`, `animal_count`, `taxon_counts`,
site/session lineage. Stored in `scene_windows.jsonl`.

**InteractionRecord** — pairwise geometry of two concurrently tracked
episodes: aligned frame interval, distance statistics (min/mean/max, time
of minimum), relative position of the neighbor (front / side / rear
fractions relative to focal heading), mean heading difference, alignment
(mean cos of heading difference), mean closing rate (+ = approaching),
mean speeds of both animals, downsampled distance series, explicit
`units`. Stored in `interactions.jsonl`.

**Episode context** — per-episode social summary: nearest-ant distance
series, ants-within-radius count series, local ant heading (circular
mean), local ant activity, `frac_frames_with_ant`, and a boolean
`has_ant` (ant within radius for ≥ half the episode). Embedded in the
atlas data and each `meta/<id>.json`.

## What counts as "concurrent"

Only episodes from the **same source video and same site** with real xy
trajectories, overlapping in frames, aligned by frame index (not array
position). Long-term identity is still out of scope: a pair is two
short-term tracks reliably observed in the same finite window. Episodes
without trajectories (velocity-only, e.g. Zeng 2023 gait data) are
excluded from spatial interaction and reported as excluded.

## The scene video must survive

Interaction analysis is only meaningful against the full scene. Episodes
keep `source_video_path`; scene blocks in the atlas reference the same
full-frame video, and Interaction playback overlays concurrent
trajectories on it. Cropped single-animal clips are never the basis of
interaction claims.

## Proximity ≠ interaction

Two animals appearing in the same window is co-occurrence, nothing more.
The atlas distinguishes what is measured (geometry: distance, bearing,
heading difference) from what is inferred; V0 stores and shows geometry
only. No edge, color, or animation in the interface asserts a biological
interaction.

## Correlation ≠ causation

Every contrast reported by the Interaction Layer — e.g. Siler speed with
ants nearby vs without — is a **predictive association**. The UI and this
documentation never call it a response, influence, or following, because
a single observational video cannot exclude: shared attraction to a
resource, common response to substrate/wind/light, or coincidence within
one scene. Causal language requires experimental designs we do not have.

## Episodes are not replicates — scene identity is kept

Hundreds of concurrent pairs may come from ONE video on ONE afternoon at
ONE site. Every SceneWindow and InteractionRecord carries
site → session → video lineage, and `concurrent_groups` never merges
animals across sites. Formal statistics must use the sampling hierarchy
(`hierarchy.py`, hierarchical bootstrap), not raw pair counts.

## Which metrics are descriptive (V0)

Distance to nearest neighbor, neighbors within radius, local density,
relative bearing, front/side/rear position, heading difference,
alignment, closing rate, speeds. Plus the Siler × ant read-out:
nearest-ant distance, ants nearby (count, mean heading, mean speed,
activity) and the with/without contrasts below. All computed from
trajectories — no pose, no learning model.

## Role in the mimicry framework

Interaction context exists to answer a mimicry question, not a general
one: is ant-like movement **context-dependent** (Question 5 of the core
framework)? Ant density, nearest-ant distance and local ant activity are
the ecological variables a mimic actually experiences; comparing mimic
episodes in ant-rich vs ant-poor moments — always against controls — is
how contextual plasticity of behavioral mimicry will eventually be tested.

## The with/without contrast and its null

For Siler episodes grouped by `has_ant` (at the analysis radius), we
report per-metric medians, a Mann-Whitney U p-value, and an
**episode-shuffle null**: context labels are permuted among Siler
episodes of the same video, giving a null distribution of the median
difference; `p_perm` is the rank of the observed difference in that null
(+1 correction). A distance-response scan repeats the contrast at
0.5× / 1× / 2× / 4× the radius, so no single arbitrary threshold decides
the result (radii where one group has <3 episodes are dropped — a fact
about scene density, not an error).

**D_ant(z)** — distance from a Siler episode to the ant behavioral
distribution in the same standardized feature space the Movement Atlas
uses (labels enter only as the reference distribution). We compare
D_ant(Siler | ants nearby) vs D_ant(Siler | absent): does Siler sit
closer to ant-like movement when ants are around? No answer is assumed;
a null result is a real result.

On the current field run (83 episodes, machine pre-labels, uncalibrated
px units), at the 100 px radius the contrast is speed Δ ≈ +73 px/s with
p_perm ≈ 0.14 and D_ant Δ ≈ +0.03 with p_perm ≈ 0.85 — **no significant
effect**, which is the honest V0 read-out until more annotated footage
exists.

## Labels

Interaction read-outs follow the same rule as the rest of KineMimic:
machine labels are pre-screening; only human-confirmed labels are ground
truth. The summary records how many episodes are human-labeled, and
`kinemimic interact --require-human` restricts taxon context to
human-confirmed episodes outright.

## Interaction Mode in the atlas

Every visual encoding is data from the same source video:

| element | meaning |
|---|---|
| hollow Siler particle | no ants within the analysis radius during that episode |
| filled particle | ants nearby (post-Reveal only) |
| scene overlay on the clip | real trajectories of the focal (white) and concurrent animals, drawn up to the shared playhead |
| "nearest ant" trace | real distance to the nearest ant over the episode (gaps = no ant visible) |
| "ants in radius" trace | real count of ants within the radius over time |
| Context line | "Alone — no ants within R" or "Ants nearby — mean n within R · nearest d · local ant activity" |

Compare buttons let you place the same movement region with contrasting
social context side-by-side. No arrows, no attraction animation, no
predicted paths — nothing is drawn that was not recorded.

## V1 and beyond (not in this build)

V1: lagged coupling (Corr(ant turning(t), Siler turning(t+τ))), movement
synchrony, turn/speed coupling, spatially constrained nulls. V2:
unsupervised interaction representation, Interaction Space, Interaction
Dictionary (machine-numbered I001…, named only by humans after watching).
V3: pose-level interaction (Leg-I vs antennae). The three showcase
questions in order: (1) Does Siler move like an ant? (2) Does Siler
movement depend on the presence and movement of ants? (3) Does Siler
become more ant-like in ant-rich contexts? — the third is a hypothesis
under test, never a preset conclusion.

## Running it

```bash
python -m kinemimic interact <run_dir> --window-s 2 --stride-s 1
# optional: --radius 150 --require-human --n-shuffle 500
python -m kinemimic atlas <run_dir>/episodes.jsonl --out <run_dir>/atlas_v2
python -m kinemimic serve <run_dir>/atlas_v2 --episodes <run_dir>/episodes.jsonl
```

Outputs: `interactions.jsonl`, `scene_windows.jsonl`,
`interaction_summary.json` (all append-only run artifacts with full
provenance: software version, window/stride/radius parameters, label
source, shuffle count, seed).
