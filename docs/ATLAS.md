# The Atlas — The Murmur

The deliverable of MOTIONSCAPE is not a dashboard; it is a **movement space
you can walk into**. This document explains what you see, why it is
trustworthy, and how it scales.

## Entering (the four stages)

1. **A real trajectory** grows on a dark field under the showcase question
   ("How does a spider move like an ant?").
2. **More episodes** fade in as points.
3. The points **glide into behavioral space** — each to its own coordinates.
4. The hero dissolves; you are inside **The Murmur**.

Click anywhere to skip. With `prefers-reduced-motion` the intro is skipped
entirely.

## The blind contract

On entry, **nothing in the interface knows or shows species identity**:
every particle is the same neutral color, similar movements are listed as
"Movement 1..k", motif occupancy bars are withheld, Behavior River and the
fingerprint stay locked. The space itself was built without labels
(embedding = kinematics only). **Reveal species** then fades the labels in
over ~2 s — the moment you discover structure you have already been
exploring. `R` toggles it any time.

## What each channel means

| channel | data |
|---|---|
| base position | behavioral-space embedding (PCA of 16 kinematic features) |
| drift | the episode's own sliding-window path z₁…z_t (real temporal structure) |
| flutter | speed_cv — intermittency; stop-and-go episodes visibly quiver |
| color | hidden → effective biological label after Reveal |
| click | the real episode (video or faithful replay) + provenance |

Episodes without windowable temporal data sit still. Nothing moves that is
not data.

## Exploring

- **Click a particle** → detail panel: the episode's GIF (real source
  frames when the video is reachable), real speed/turn/moving series,
  trajectory replay, features, full provenance chain back to video+frames.
- **Similar movements** — nearest neighbors in the standardized 17-D
  feature space (exact; k-d tree at scale), *not* screen distance. Before
  Reveal they are anonymous; after Reveal each carries its identity, and
  `⇄ compare` opens a **synchronized side-by-side**: shared playhead, both
  real trajectories, both speed traces — you see *why* they are similar.
- **Click empty space** → region exploration: who lives in this
  neighborhood of movement space, with representative clips.
- **Explore drawer**:
  - *Motion Dictionary* — machine-found motifs M0…Mk with counts and
    representative clips. They have **numbers, not names**: type a human
    name after watching (persisted via the server to
    `motif_annotations.json`).
  - *Mimicry fingerprint* — per-dimension Siler↔ant overlap
    (Bhattacharyya, 0–1), deliberately **not** collapsed into one number.
    Future dimensions (pose, leg-I dynamics, motif occupancy, grammar) are
    listed as pending, not faked.
  - *Behavior River* — species movement volume across the day.
  - *Data & provenance* — the sampling hierarchy tree
    (site → session → video → episode), embedding model + explained
    variance, run manifest, and whether server media is on.

## Opening an atlas

```bash
python -m motionscape serve <run>/atlas_v2 --episodes <run>/episodes.jsonl
# → http://127.0.0.1:8694
```

`serve` is recommended: it decodes source video **on demand** (only for
clips you actually open, cached under `cache/clips/`). A statically served
atlas (plain `python -m http.server`) still works — clips fall back to
in-browser trajectory replay, which is the same real path.

## Scale

Design: `data.json` carries ~0.32 KB per episode (embedding, drift path,
minimal metadata); per-episode detail lives in `meta/<id>.json` and is
fetched only on selection; no media is pre-rendered; particles render on
one canvas (no DOM per particle); nearest neighbors use a k-d tree.

Measured (`python -m motionscape benchmark`, recorded in `benchmarks/`):

| episodes | atlas build | NN (feature space) | data.json |
|---:|---:|---:|---:|
| 1,000 | 7.0 s | 3.75 s (brute force) | 0.3 MB |
| 5,000 | 32 s | 0.08 s | 1.6 MB |
| 10,000 | 74 s | 0.16 s | 3.2 MB |
| 20,000 | 142 s | 0.81 s | 6.6 MB |

Build time is dominated by sliding-window path projection and scales
linearly (~7 s per 1,000 trajectory episodes). In-browser frame rate is
shown live in the bottom-right corner (`fps · N particles`).

## Scientific integrity knobs

- 2-D projection ≠ evidence: every similarity number comes from feature
  space (see `SCIENTIFIC_ASSUMPTIONS.md` §5, §7).
- Synthetic/demo atlases carry `site="synthetic"` in their provenance; real
  interpretation only on real runs.
- Reveal shows the data as it is — overlapping regions, separated regions —
  and nothing in the render pipeline nudges either.
