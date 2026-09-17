# MOTIONSCAPE

**Exploring how ant-mimicking lineages independently evolve to move like ants.**

An explorable behavioral phenotype space for the evolution of ant mimicry.

> 核心理念:用可探索的行为表型空间,研究不同拟蚁类群如何独立演化出
> "像蚂蚁一样运动"的能力。

## 1. What is MOTIONSCAPE?

MOTIONSCAPE is a scientific instrument, not a general animal-tracking
tool. It turns naturalistic video into **movement phenotypes** —
quantitative descriptions of how animals move — and organizes them into
an explorable behavioral space, so that behavioral mimicry can be studied
as a comparative, evolutionary phenomenon: visible, measurable, and
traceable back to raw footage.

## 2. Core scientific question

> Do independently evolved ant-mimicking lineages repeatedly converge on
> ant-like movement — and through which behavioral dimensions?

Two questions anchor the project's highest-level acceptance criteria:

1. Do independently evolved ant mimics repeatedly enter the same regions
   of ant behavioral phenotype space?
2. Do different mimic lineages achieve ant-like movement through the same
   or different behavioral dimensions?

## 3. Why ant mimicry?

Ant mimicry has evolved **repeatedly and independently** across arthropods
— in spiders (e.g. *Siler*, *Myrmarachne*), beetles, hemipterans and
others — yet most work focuses on morphology. Movement itself is rarely
treated as a comparative phenotype. Because the same model (ants) has
many independent mimics, ant mimicry is a natural experiment for asking
whether evolution repeats itself in behavior.

## 4. Behavioral phenotype space

The unit of analysis is the **movement episode**; a species is represented
by the distribution of its episodes — a behavioral cloud, never a single
point. The comparison framework always holds four biological roles apart:

| role | meaning |
|---|---|
| **Model** | true ants — the Ant Behavioral Reference Space |
| **Mimic** | independently evolved ant-mimicking lineages |
| **Phylogenetic control** | close non-mimic relatives (is ant-like movement just this lineage's way of moving?) |
| **Ecological control** | similar small arthropods, non-mimicking (is it just how small ground arthropods move?) |

Roles are metadata for comparison and evaluation; they never enter the
encoders or the behavioral space. Ants themselves are diverse (exploring,
foraging, recruiting, carrying…) — so the question is not how close a
mimic is to "the average ant", but **which regions of the ant behavioral
distribution mimics occupy**. Candidate ant-like behavioral regions must
emerge from the data; they are never defined in advance.

**Scope boundary.** MOTIONSCAPE is not intended to compare all animal
movement in a single universal latent space. It focuses on biologically
comparable locomotor systems relevant to ant mimicry.

## 5. Current showcase — Siler × Ant

The first system: *Siler collingwoodi*, a jumping spider that moves like
the sympatric ants it lives among, with the Zeng et al. 2023 ant community
as models and *Phintelloides versicolor* as a non-mimic salticid control.
This showcase validates the pipeline end-to-end: natural video → movement
phenotypes → comparable behavioral space → controls → explorable atlas.

## 6. Pipeline

```
Video → Detection → Tracking → Movement Episodes
      → Kinematic Features → Behavioral Space (label-blind PCA)
      → Motifs → Behavioral Mimicry Fingerprint (per-dimension)
      → Interactive Atlas
      + Human Annotation / QC + Interaction Context (scenes, pairwise)
      + Behavioral Retrieval (Find Similar)
```

Human annotation is the only admissible ground truth: machine
pre-classification shares features with the movement analysis and is
treated as pre-screening only.

## 7. The Murmur

The Murmur is the core exploration interface: **an explorable ant-mimicry
behavioral landscape**. One particle = one real movement episode;
positions come from a label-blind embedding; drift is the episode's own
path through the space; color is hidden until you **Reveal biology** —
then species, and their roles (model / mimic / controls), appear. The
signature picture this interface aims at: different independent mimic
lineages entering the same ant-occupied regions — behavioral convergence
made visible, if the data support it.

## 8. Behavioral retrieval

Drop in a video and it is detected, segmented and encoded, then placed at
its real position in the landscape: **where does this animal sit in the
ant-mimicry behavioral reference space?** Nearest ant episodes, nearest
mimic episodes, nearest controls, closest motifs, similarity breakdown
from real distance decomposition, out-of-distribution flags, and human
similarity ratings — all versioned (docs/RETRIEVAL.md,
docs/MODEL_CARD.md). Not a species identifier.

## 9. Data sources

- **Shamble et al. 2017** (Dryad fd612): *Myrmarachne formicaria* (mimic),
  *Salticus senicus* (non-mimic control), ants — gold trajectory labels.
- **Zeng et al. 2023** (iScience): *Siler collingwoodi* velocity/pose
  episodes, five sympatric ant species, *Phintelloides* control.
- **Own field footage** for matched, same-site mimic–model comparisons.
- Public repositories (Dryad, Zenodo, Figshare, Mendeley Data, OSF,
  Wikimedia; license-checked, fully provenance-tracked) are prioritized
  for the **Ant Reference Library**; own collections remain the basis of
  matched local comparisons. Sampling follows the tiers: ant models →
  known mimics → close non-mimic relatives → ecological controls.

## 10. Scientific controls

- Machine predictions are pre-screening, never ground truth (circularity).
- Embeddings are label-blind by construction and enforced by tests.
- Phylogenetic and ecological controls accompany every comparison.
- Global (public-data) and local (sympatric, matched-condition) reference
  sets are kept distinct.
- Interaction analyses distinguish proximity from interaction and
  association from causation; scene identity is preserved.
- Episodes are not independent replicates: Site → Session → Video →
  Episode hierarchy is stored and statistics must respect it.

## 11. Roadmap — by scientific phase

| Phase | scope |
|---|---|
| **1** | Siler–ant movement space (current showcase) |
| **2** | a second, independently evolved mimic lineage (Myrmarachnini) |
| **3** | non-spider ant mimics (beetles, hemipterans) — true convergence tests |
| **4** | pose and appendage mimicry (leg-I vs antenna) |
| **5** | interaction and context dependence |
| **6** | comparative evolution of behavioral ant mimicry (incl. phylogenetic integration, strategy space, behavioral grammar) |

Technology (encoders, ANN backends, rendering) serves these phases; it is
never the roadmap itself. Long-term hypotheses — e.g. a
morphology↓/behavior↑ compensation between static and dynamic mimicry —
remain hypotheses under test, never UI claims.

## 12. Reproducibility

Append-only runs with full provenance; versioned reference indexes and
encoders (weights checksums); append-only annotation and evaluation logs;
every atlas particle traces back to source video and frames; label-blind
embedding is enforced by tests. Benchmarks live in `benchmarks/`.
Scientific assumptions and their limits: docs/SCIENTIFIC_ASSUMPTIONS.md.

## 13. Quick start / citation / contribution

```bash
pip install -e .
python -m motionscape demo                      # synthetic end-to-end tour
python -m motionscape serve <run>/atlas         # explore the atlas
python -m motionscape annotate <run>/episodes.jsonl
python -m motionscape interact <run>            # interaction context
python -m motionscape build-reference RUN/episodes.jsonl --out reference
python -m motionscape find-similar NEW.mp4 --query-id q1 \
    --reference reference --atlas <atlas_dir>   # drop in a video
```

Contributions that sharpen the comparative framework — new mimics with
citations, control taxa, matched-condition footage, curated ant reference
data with provenance — are prioritized over generic feature work.
