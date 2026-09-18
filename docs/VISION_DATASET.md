# VISION DATASET — training & evaluation data policy

Small, diverse, high quality beats large and sloppy. The KineMimic
vision models (when trained) and the benchmark must be built from
footage that looks like the science: small animals on natural
substrates.

## Classes

First release (coarse, observational only — never taxonomy):

- `ant`
- `spider`
- `other_arthropod`
- `unknown` / background

Fine-grained taxonomy (Siler collingwoodi, Crematogaster sp., …) lives in
the human annotation layer, strictly separated (Observation ≠ Biological
Annotation).

## Sampling targets

- natural backgrounds: bark, leaves, soil, arena paper
- multiple species (ants of different subfamilies; jumping spiders;
  non-mimic controls)
- different scales (tiny 4-px ants to 100-px spiders)
- different light (overcast, direct sun, shade, dawn/dusk)
- different cameras and mounting (tripod, attempted-stabilized handheld)
- different densities (1 animal → 10+; crossings; touching; occlusions)

## Hard negatives (§48) — mandatory in training AND benchmark

The project's own data already proves why: the first "field validation"
produced 83 episodes that were mostly **handwriting strokes on the field
card** and camera-settling artifacts. Therefore these must be collected
as explicit negatives:

- moving leaves / vegetation
- shadows and glare
- field cards, handwritten labels, timestamp overlays, camera UI
- small debris, soil crumbs
- non-mimic arthropods that look movement-similar

Also: **ignore regions** (§49) and optional **ROI** (§50) must be
supported and recorded in provenance so fixed overlays never become
phantom tracks.

## Annotation protocol

1. Sample frames/segments with the active-learning selector
   (low-confidence detections, ambiguity events, collisions, domain
   shift) — not random frames (§47).
2. Annotate bbox (+ instance mask where feasible) with the workbench;
   class from the coarse set above.
3. Record annotator, timestamp, and source video provenance (§45):
   source URL/DOI, license, taxon, field/lab, fps, scale/calibration,
   substrate, camera.
4. QA pass by a second annotator on a sample.
5. Splits are grouped by VIDEO (never by frame) to prevent leakage —
   matching the leave-video-out rule of the retrieval evaluation.

## License rule

Only footage with a license that permits redistribution of derived
annotations enters the training set (CC-BY/CC0 preferred); every asset's
license and provenance are recorded alongside it.
