# SCIENTIFIC ASSUMPTIONS

Every claim KineMimic's interface makes rests on the assumptions below.
They are stated so a reviewer can challenge any of them individually.

## 0. What the project is — and is not

KineMimic studies one specific, repeatedly evolved phenomenon:
behavioral ant mimicry across independent lineages. Its scope is bounded
by that question. It is **not** a universal animal-movement comparator:
taxa whose scale, locomotor apparatus, medium, and ecology differ too
widely from the mimicry systems under study do not enter the same
behavioral similarity space, because cross-system distances there carry
no biological meaning.

The long-term questions (all under test, none assumed):

1. Do independent ant-mimic lineages converge on similar ant-like
   movement space?
2. In which behavioral dimensions does any convergence occur (trajectory
   geometry, speed, turning, stop–go rhythm, pause structure, appendage
   motion, pose, interaction responses)?
3. Do different lineages reach mimicry through different behavioral
   strategies?
4. How do static (morphological) and dynamic (behavioral) mimicry
   relate — synergistic, independent, or compensatory? (e.g. the
   morphology↓ / behavior↑ trade-off is a working HYPOTHESIS, never a
   UI claim.)
5. Is behavioral mimicry context-dependent (ant-rich environments)?
6. Where do convergence, trade-offs, and constraints appear across
   lineages?

Analysis units are **movement phenotypes and behavioral distributions**:
one species = a cloud P(z | species), never one point. Ants are equally
diverse (exploration, foraging, recruitment, carrying…), so the question
is which regions of the ant behavioral distribution mimics occupy — and
candidate "ant-like regions" must emerge from occupancy data (ant-high,
mimic-high, control-low), never be defined a priori. A future
**Strategy Space** (per-lineage fingerprint across trajectory / rhythm /
turning / pose / appendage / interaction mimicry dimensions) follows the
same rule: many roads to a similar result.

## 1. What an episode is

An **episode** is one continuous, sufficiently reliable observation of one
animal (~3–30 s): detection coverage and displacement sanity hold from first
to last frame. It ends when the animal disappears, is occluded, exits, or
tracking becomes unreliable — deliberately, because stitching across gaps
would fabricate movement that was never observed.

## 2. Why individual identity is not required

Long-term re-identification is out of scope **on purpose**. The scientific
question ("how does this animal's *movement* compare to that one's?") lives
at the movement level, not the individual level. Re-identification would add
error (identity mistakes) without adding signal to the movement comparison.
An animal that reappears simply contributes another episode.

## 3. Why machine labels are not ground truth

The machine pre-classifier (`heuristic-v1`) predicts Siler / ant / other from
**movement features** — the very quantities the mimicry analysis then
studies. Using those predictions as truth would be circular: the classifier
could "discover" that Siler move like ants because intermittency was part of
its own decision rule. Therefore:

- machine predictions live in `machine_label` and are shown in the
  workbench with an explicit circularity warning;
- only **independent human annotation** (`human_label`, from the annotation
  workbench or dataset metadata) is admissible ground truth;
- the atlas shows no species identity until "Reveal species", and the
  reveal overlay uses effective labels = human ⊕ machine (human wins).

## 4. Is the embedding blind to biological labels?

**Yes.** The behavioral space is PCA over standardized kinematic features
only. The `fit_transform` API has no label input at all; motifs are k-means
over the same features. Labels are applied *after* the space exists, as
overlays. This is enforced by a test
(`test_pipeline_embedding_ignores_annotation`) that re-runs the full
pipeline with different human labels and asserts the embeddings are
bit-identical.

## 5. What PCA (and UMAP) are for here

PCA is a **visualization and neighbor-hunting device**, not evidence. A 2-D
projection distorts distances; two particles close on screen are not
necessarily similar. All similarity claims in the interface ("similar
movements", comparisons) come from distances in the **original standardized
16-D feature space** (k-d tree for speed at scale — exact, not
approximate). The projection's job is to let a human see the shape of the
space; the numbers behind any claim never come from screen distance.

## 6. Why an episode is not a biological replicate

Episodes from one video share camera, weather, time of day, substrate, and
the same few individuals. Treating 5,000 episodes as 5,000 independent
replicates is pseudo-replication. The data model therefore stores an
explicit hierarchy — **Site → Session → Video → Episode** — and
`hierarchy.hierarchical_bootstrap` resamples whole sites (then sessions,
then videos when only one site exists) so confidence intervals respect the
design. Report `n_sites / n_sessions / n_videos` alongside episode counts.

## 7. How "nearest movement" is computed

Standardized features (z-scores per dimension), Euclidean distance in full
feature space, k nearest episodes. Above 800 episodes this uses a scipy
k-d tree — an exact algorithm, not an approximation. Before "Reveal
species", neighbors are shown as "Movement 1..k" so identity cannot leak;
the computation itself never used labels (the space is blind).

## 8. What the Murmur animation encodes

Every channel is data:

| channel | meaning |
|---|---|
| base position | episode's location in behavioral space |
| drift | the episode's own sliding-window path z₁…z_t through that space (real temporal structure, when windows are computable) |
| flutter amplitude | speed_cv (movement intermittency) |
| color | hidden until Reveal; then the effective biological label |

Episodes whose temporal data cannot support windowing get a **static
position** — the flock then simply doesn't move; no decorative motion is
substituted. Drift phases are per-episode (deterministic hash), so the
flock is not synchronized swimming.

## 9. Which animations are only UI transitions

The hero intro (trajectory drawing, particle spawn-in), panel slide,
reveal color interpolation, and hover/focus fades carry **no data** — they
exist to sequence attention. They are kept few, slow, and skippable
(click), and all data motion is disabled under
`prefers-reduced-motion`. If an animation cannot answer "what does this
encode?", it belongs in this list or it gets deleted.

## 10. What we do not claim

- The atlas does not prove mimicry; it organizes evidence and points at
  testable dimensions (the fingerprint is a descriptive profile, not a
  statistic; formal tests happen in analysis notebooks with the hierarchy
  respected).
- Synthetic demo atlases exist to exercise the pipeline; real biological
  interpretation is only admissible on real runs (check Data & provenance).
- Nothing in the interface was tuned to make Siler look ant-like: the
  embedding, motifs, and neighbors are computed before any label is read.
