# KineMimic Vision Benchmark Dataset

Small, high-quality, real-footage benchmark for the vision gate
(roadmap §8–26). Clips are **not git-tracked** (`.gitignore`); ground
truth, splits and this manifest are.

## Layout

- `clips/<scene>.mp4` + `clips/manifest.json` — real scenes, 4 s each
  (expand to 20–50 clips as footage allows)
- `ground_truth/<scene>.json` — per-animal frame-level boxes + track
  identity; `TEMPLATE.json` defines the schema
- `splits.json` — development / validation / holdout (holdout never
  used for threshold tuning)

## Current scenes (from Shamble 2017 GoPro footage, DOI 10.5061/dryad.fd612)

| scene | role | why it matters |
|---|---|---|
| `fieldcard_hardneg` | hard negative | field-card intro; system must output **zero** episodes (regression: the invalid 83-episode baseline) |
| `tiny_ants_body` | tiny-animal recall | real ants invisible to bgdiff; core detector-recall scene |
| `tiny_ants_late` | holdout | same class, held out from tuning |
| `density_crossing` | ambiguity | candidate crossings/body contacts |

## Ground-truth rules

- bbox + centroid + track identity first; masks only for a subset.
- If humans cannot resolve identity after contact, mark
  `identity_ambiguous` — the correct system behavior is a **split**,
  never a guessed continuation.
- Classes are coarse observational categories (`ant` / `spider` /
  `other_arthropod` / `unknown`); fine taxonomy belongs to the human
  annotation layer.

## Regression gate

`tests/test_vision_benchmark.py::test_fieldcard_hardneg_zero_episodes`
must hold for every vision backend before it can be chosen as
production default.
