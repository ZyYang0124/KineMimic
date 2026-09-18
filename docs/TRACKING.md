# TRACKING — why KineMimic splits instead of guessing

## The principle

KineMimic does not need to know whether the ant at minute 10 is the same
individual as "Ant #37" at minute 0. Long-term re-identification is
explicitly out of scope. What matters is:

> **One movement episode, from first to last frame, belongs to one real
> animal.**

We call this **episode purity**, and it outranks identity continuity.
When animals approach, touch, cross, or merge and identity cannot be
judged reliably, the policy is:

> **prefer splitting over guessing.**

A real trajectory split into two episodes loses a little continuity. Two
animals welded into one trajectory **poison every downstream number** —
speed, turning, rhythm, mimicry overlap — while looking perfectly
plausible. That asymmetry is quantified in the benchmark: the cost
function weights a false merge 5× higher than fragmentation
(docs/VISION_BENCHMARK.md).

## When the tracker splits

The two-stage tracker associates detections to tracklets by gated motion
distance (constant-velocity prediction, velocity decayed while coasting)
plus an IoU bonus. A match is rejected as ambiguous when the runner-up
candidate is nearly as good as the best — tested both multiplicatively
and by an absolute pixel margin (the margin catches coincident
detections, where the ratio test degenerates to 0 < 0).

On ambiguity:

1. both involved tracklets **terminate at their last confident frame**;
2. the contested detection seeds fresh tracklet(s) that inherit the
   terminated track's *motion state* (velocity) — motion continuity, not
   identity guessing — and every new tracklet carries an
   `AmbiguityEvent`;
3. when two active tracks converge on one detection (a collision blob),
   both terminate and one new tracklet continues from the blob; when the
   animals separate again, fresh tracklets begin — exactly the
   "A1, B1 → ambiguous collision → C1, D1" outcome.

## Stationary animals are not tracking failures

A real ant stops. With detection-based frontends a stationary animal is
still detected (unlike pure background subtraction), so a 2-second pause
is just zero velocity — the track continues. Stopping is **stop–go
behavior**, one of the phenotypes KineMimic measures; silently dropping
or splitting stationary stretches would fabricate behavior.

## observed / interpolated / propagated — never silently mixed

Every trajectory point carries its state:

| state | meaning |
|---|---|
| `observed` | a detector actually saw the animal in this frame |
| `interpolated` | linear fill across a short miss (≤ `max_interpolated_gap`) |
| `propagated` | model-predicted position (gap-recovery backend; future) |
| `human_corrected` | moved/added by an annotator |

Gap policy: short gaps are bridged and flagged; gaps longer than
`max_gap_frames` finalize the tracklet — reappearance starts a NEW one.
Downstream analysis chooses which states to accept; the data never
pretends an interpolation was a measurement.

## Detectors vs taxonomy

Detector classes are coarse visual categories (`ant`, `spider`,
`other_arthropod`, `unknown`). A human taxon annotation
(*Siler collingwoodi*, *Crematogaster* sp.) is a separate layer that a
detector class can never overwrite: **Observation ≠ Biological
Annotation**.

## Ambiguity is information

`ambiguity_events` on a tracklet are not errors — they are honest markers
of the moments identity was uncertain. They surface in the workbench
("⚠ Association ambiguity: N event(s)") so a researcher can review the
raw video at exactly those frames, and they feed the False-Merge /
Ambiguous-Collision error rates of the benchmark.
