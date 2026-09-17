# MOTIONSCAPE — The Atlas of Animal Movement

最终产品不是行为分析软件，而是一张**可以探索的动物运动图谱**；
行为分析算法负责让这张图谱科学可信。第一生物系统是拟蚁跳蛛
*Siler* 与同域蚂蚁，但图谱本身与物种无关，可扩展到任何动物运动。

**Computational ethology for ant-mimicking jumping spiders.**
How does a *Siler* jumping spider move like an ant? MOTIONSCAPE turns naturalistic
video into a quantitative, explorable movement space — a **movement
murmuration** where every particle is one real movement episode.

```
Video → Detection → Movement Episodes → Trajectory → Kinematics
      → Features + Motifs → BEHAVIOR SPACE → Siler↔Ant comparison
                                              → Behavioral mimicry
```

## Quick start

```bash
pip install -e .          # or just run from the repo root
python -m motionscape demo     # synthetic end-to-end tour (~2 min)
```

The demo renders four synthetic field videos (ants: smooth persistent
walking; Siler: intermittent stop-and-go with jumps), runs the full
pipeline, and writes a run directory containing `episodes.jsonl`,
`summary.json` (including a first **mimicry fingerprint**), and
`atlas/index.html` — **The Atlas**: serve it (`python -m http.server`
inside the atlas dir) and open in a browser.

### The Movement Galaxy experience

- **Hero intro** — a real trajectory grows on a dark field under the question
  *"How does a spider move like an ant?"*; one click dissolves into the galaxy.
- **Explore the flock** — the landing view is a living murmuration.
  Particles are positioned by behavioral similarity only (species colors
  hidden); each particle drifts along its episode's *real* sliding-window
  embedding path z₁…z_t, so flock motion is data, not decoration.
  Press **Reveal species** and the flock colors in over ~2 s — the moment
  you discover which regions are ants, which are Siler.
- **Explore a movement** — click any particle: the original video clip
  plays, with live trajectory trace, speed sparkline, pause statistics,
  playing alongside, with **speed / turn / moving** time-series rows, and
  **Compare with nearest ant** for a split-screen side-by-side. Every clip is
  either the true source video (where available) or a trajectory replay whose
  motion is the real path — never decoration.
- **Explore a motif (Motion Dictionary)** — select a discovered motif; the
  rest of the galaxy fades and the panel shows species-occurrence bars plus
  representative clips. Machine discovers behaviors; humans name them after
  watching (e.g. M2: 65.6% ant, 28.1% non-mimic, 6.3% mimic).
- **Behavior River** — species movement volume flowing across the day
  (hour-of-day when timestamps exist): rhythmic differences between ants,
  mimics and non-mimics become visible science questions.

## Commands

```bash
python -m motionscape ingest VIDEO.mp4 --id siteA_clip01 --store murmur_runs
python -m motionscape viz murmur_runs/runs/<run_id>/episodes.jsonl --out murmuration.html
```

## Design principles

1. **The episode, not the individual, is the unit.** A movement episode is
   one continuous reliable observation (~3–30 s). Reappearing animals simply
   create new episodes; long-term identity is deliberately out of scope.
2. **Every point links back to life.** Episodes store video id, frame range,
   raw trajectory, features, and a full processing-history chain. Clicking a
   particle in the visualization shows this provenance and replays the real
   trajectory.
3. **Observation ≠ annotation.** Biological labels (Siler / ant / other
   spider / other arthropod / unknown) are model-or-human annotations stored
   separately and refinable (e.g. ant → *Crematogaster*) without reprocessing.
4. **Never overwrite science.** Every analysis is an append-only run with a
   manifest (software version, model, parameters, timestamp, parent run).
5. **Two discovery routes.** Interpretable kinematics (speed, turning,
   stop–go rhythm, sinuosity) *and* unsupervised structure (PCA behavioral
   space, k-means motifs). Mimicry is measured per dimension
   (Bhattacharyya overlap) toward a Behavioral Mimicry Fingerprint.
6. **Visual motion is data.** In the atlas: position = embedding path
   (windowed, real drift through behavioral space), color = label,
   flutter amplitude = movement intermittency (speed CV), clips = the
   actual source-video frames.

## Status (V1)

- Detection: background-differencing detector for small dark arthropods
  (swappable; model name/version recorded per run)
- Short-term greedy tracker; episodes with QC
- 17 kinematic features incl. move–pause rhythm
- Heuristic transparent classifier (body elongation × intermittency)
- PCA behavioral space, k-means motifs, mimicry fingerprint
- Self-contained HTML murmuration visualization

## Roadmap

Pose (esp. Siler foreleg-I vs ant antennae), behavioral grammar
(motif transition sequences), non-mimetic jumping spider controls,
video-clip playback on episode click, real field data ingest.

## Your own videos

Film spiders and ants together, then:

```bash
python -m motionscape ingest myvideo.mp4 --id siteA_01
```

Detection → episodes → galaxy → mimicry comparison, fully automatic.
Filming tips, tuning and annotation refinement: **`docs/UPLOAD.md`**.
Validated end-to-end on real field footage (83 episodes from 57 s of
120 fps GoPro video).

See `docs/DESIGN.md` for the full data model and provenance contract.
