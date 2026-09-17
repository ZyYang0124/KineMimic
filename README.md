# MOTIONSCAPE — An atlas of animal movement

MOTIONSCAPE is an **explorable atlas of animal movement** growing into
an atlas of movement, interaction, and behavior:

```
Movement     how does one animal move?               — episodes, behavioral space
Interaction  how does movement depend on others?     — scenes, pairwise context
Behavior     how are movements organized?            — future
```

Real movement episodes, computationally characterized, organized into a
space you walk into and explore — the way bird-call projects let people
explore the structure of sound. Behavioral analysis exists to keep the
atlas scientifically honest.

```
MOTIONSCAPE
└── The Murmur          — the explorable movement space (the atlas itself)
    └── Ant Mimicry     — first biological showcase:
                          How does a spider move like an ant?
```

Future showcases (courtship, predation, any taxon) use the same
architecture — no species concepts are baked into the pipeline.

## Quick start

```bash
pip install -e .          # or just run from the repo root
python -m motionscape demo        # synthetic end-to-end tour (~2 min)
python -m motionscape serve <run>/atlas   # open the atlas
```

The demo renders four synthetic field videos (ants: smooth persistent
walking; Siler: intermittent stop-and-go with jumps), runs the full
pipeline, and writes a run directory with `episodes.jsonl`,
`summary.json`, and `atlas/` — **The Murmur**.

### Inside the atlas

- You land **blind**: the movement space shows structure, not species —
  every particle is one real episode, positioned by behavioral similarity,
  drifting along its own real sliding-window path through the space,
  quivering with its movement intermittency.
- **Click any particle**: the real episode plays (source video when
  reachable; otherwise a faithful replay), with real speed / turn / moving
  time series, trajectory, and provenance back to video and frames.
- **Similar movements** are nearest neighbors in the original 17-D feature
  space — never screen distance. `⇄ compare` plays selected vs. neighbor
  side-by-side with a shared playhead: see *why* they are similar.
- **Reveal species** (`R`) is the signature moment: only now do Siler /
  Ant / Other identities fade in over the structure you've been exploring.
- **Motion Dictionary**: machine-discovered motifs with numbers, not
  names — watch the representative episodes, then name them (persisted).
- **Mimicry fingerprint**: per-dimension Siler↔ant overlap, deliberately
  never collapsed into one number.
- **Behavior River** and the sampling-hierarchy tree live in the Explore
  drawer; every statistic respects Site → Session → Video → Episode.

Full tour: **`docs/ATLAS.md`** · what each channel encodes, and every
scientific assumption behind the interface:
**`docs/SCIENTIFIC_ASSUMPTIONS.md`**.

## Commands

```bash
python -m motionscape ingest VIDEO.mp4 --id siteA_01 --site siteA
python -m motionscape annotate <run>/episodes.jsonl   # human QC + labels
python -m motionscape atlas <run>/episodes.jsonl --out <run>/atlas_v2
python -m motionscape serve <run>/atlas_v2 --episodes <run>/episodes.jsonl
python -m motionscape interact <run_dir>             # scenes + social context
python -m motionscape benchmark --n 5000 10000        # scale checks
```

## Your own videos

Film spiders and ants together, then `ingest` → `annotate` → `serve`
(filming tips: **`docs/UPLOAD.md`**). Validated end-to-end on real field
footage (83 episodes from 57 s of 120 fps GoPro video). Annotation is the
line between a demo and a result: machine pre-labels share features with
the movement analysis and can never serve as ground truth — the workbench
(`docs/ANNOTATION.md`) makes human review fast enough to actually do.

## Design principles

1. **The episode, not the individual, is the unit.** A movement episode is
   one continuous reliable observation (~3–30 s); reappearance creates a
   new episode; long-term identity is deliberately out of scope.
2. **Every point links back to life.** Episodes store video id, frame
   range, raw trajectory, features, and a full processing-history chain.
3. **Observation ≠ annotation.** Machine predictions and human labels live
   in separate fields; effective label = human ⊕ machine (human wins);
   biological labels never influence the behavioral space (blind
   embedding, enforced by test).
4. **Never overwrite science.** Append-only runs, append-only annotation
   log, versioned atlas directories, full provenance.
5. **Two discovery routes.** Interpretable kinematics and unsupervised
   structure (PCA space, k-means motifs); mimicry measured per dimension.
6. **Visual motion is data.** Position = embedding, drift = the episode's
   real path through the space, flutter = intermittency; episodes without
   windowable data sit still. Decorative animation is prohibited and the
   few UI transitions respect `prefers-reduced-motion`.
7. **Episodes ≠ replicates.** The sampling hierarchy is stored and
   statistics must respect it.

## Status (V0.3)

- Detection (swappable bg-diff), short-term greedy tracker, episode QC
- 16 kinematic features; transparent machine pre-classifier (separate
  from ground truth)
- **Annotation workbench** — keyboard-first human review, per-keystroke
  persistence, QC states, machine/human separation
- Blind PCA behavioral space, k-means motifs, per-dimension mimicry
  fingerprint
- **The Murmur** — blind-first atlas: staged intro, Reveal species,
  feature-space nearest neighbors, synchronized side-by-side comparison,
  region exploration, Motion Dictionary with human naming, Behavior
  River, hierarchy & provenance panels
- Scale: canvas rendering + lazy media + exact k-d-tree neighbors;
  benchmarked to 20,000 episodes per build (~0.3 KB/episode payload);
  target ≥5,000 QC-approved real episodes next
- **Interaction Layer V0** — SceneWindows + InteractionRecords from
  concurrent tracks of the same video (no long-term identity), nearest-ant
  context per Siler episode (distance/count/heading/activity), full-scene
  synchronized playback in the atlas, with/without-ant contrasts with an
  episode-shuffle null and a distance-response scan; proximity ≠
  interaction, correlation ≠ causation, scene identity kept
  (docs/INTERACTIONS.md)
- Real data: Shamble 2017 (228 episodes, gold labels), Zeng 2023 (64
  velocity/pose episodes incl. non-mimetic control), own field video

## Roadmap

Interaction V1 (lagged coupling, synchrony nulls), Interaction Space +
Dictionary (V2), pose-level interaction (V3), behavioral grammar, more
sites/sessions of field footage toward the 5,000-episode goal, UMAP as an
alternative (still-blind) space.
