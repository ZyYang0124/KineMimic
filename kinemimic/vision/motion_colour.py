"""Motion-as-colour frame rendering (after BehaveAI, Troscianko et al. 2026).

Renders temporal motion information as false-colour images so that
moving animals become visually salient even when tiny or camouflaged
against textured substrates -- the exact failure class the Phase A
field-footage audit found for appearance-based detectors (body median
0 detections; COCO-pretrained YOLO useless on arthropods).

Representation (BehaveAI, PLOS Biology 2026, doi:10.1371/journal.pbio.3003632;
open source, Zenodo 10.5281/zenodo.18232555):

- "exponential": per-channel exponential moving averages of absolute
  frame differences with distinct response rates. Recent motion lights
  all channels (whitish); as it ages, the fast channels (B, then G)
  fade first, so a trail decays white -> red. Acceleration/deceleration
  read as colour streaks along the path.
- "sequential": absolute differences against frames t-skip, t-2*skip,
  t-3*skip coded into R/G/B -- preserves spatial detail of the moving
  part (body/wing shape), at the cost of trails only 3 skips long.

Reported effect in BehaveAI: YOLO trained on motion-colour frames
detected ~2-pixel-wide moths at P/R 0.81/0.80 where appearance-only
models collapsed (0.79/0.54); camouflaged isopods recall 0.92.

Uses here (docs/VISION.md):

1. Filming QC -- see at a glance which animals motion makes salient in
   candidate field footage, before any detector is trained.
2. Dual-stream training frames -- input for fine-tuning a detector on
   the CUDA server; the static stream stays our legacy/RGB path.

Scientific contract: this is a *representation transform only*. It
detects nothing by itself, and any detector trained on these frames
still has to pass the KineMimic tracking benchmark before entering the
pipeline. Stationary animals are invisible to the motion stream -- the
documented reason BehaveAI is dual-stream, and why absence from a
motion view is never evidence of absence (an animal that has stopped is
still there). Channels before enough history exists are dark, not
fabricated.
"""

from __future__ import annotations

import numpy as np

STRATEGIES = ("exponential", "sequential")


class MotionColourEncoder:
    """Stateful per-video encoder: push grayscale/BGR frames, get RGB."""

    def __init__(self, strategy: str = "exponential", skip: int = 1,
                 decay=(0.55, 0.28, 0.14), gain: float = 3.0):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if skip < 1:
            raise ValueError("skip must be >= 1")
        if len(decay) != 3 or not all(0 < a <= 1 for a in decay):
            raise ValueError("decay needs 3 response rates in (0, 1]")
        self.strategy = strategy
        self.skip = skip
        self.decay = tuple(decay)   # fast, medium, slow -> B, G, R
        self.gain = gain
        self._prev: np.ndarray | None = None      # grayscale float
        self._ema = None                          # 3-channel EMA stack
        self._hist: list[np.ndarray] = []         # sequential-mode buffer

    # ------------------------------------------------------------------
    def push(self, frame) -> np.ndarray:
        """Consume one frame (uint8, gray or 3-channel); return HxWx3 uint8."""
        f = frame.astype(np.float32)
        if f.ndim == 3:
            f = f.mean(axis=2)
        if self.strategy == "exponential":
            rgb = self._push_exponential(f)
        else:
            rgb = self._push_sequential(f)
        return rgb

    def _push_exponential(self, f: np.ndarray) -> np.ndarray:
        diff = np.zeros_like(f) if self._prev is None else np.abs(f - self._prev)
        self._prev = f
        if self._ema is None:
            self._ema = np.zeros((3,) + f.shape, dtype=np.float32)
        for c, a in enumerate(self.decay):        # fast -> slow
            self._ema[c] = (1.0 - a) * self._ema[c] + a * diff
        chans = [self._ema[2], self._ema[1], self._ema[0]]  # slow->R, fast->B
        return self._to_uint8(chans)

    def _push_sequential(self, f: np.ndarray) -> np.ndarray:
        self._hist.append(f)
        need = 3 * self.skip + 1
        if len(self._hist) > need:
            self._hist = self._hist[-need:]
        # channel c compares against t-(c+1)*skip; dark until history exists
        chans = []
        for c in (1, 2, 3):
            j = len(self._hist) - 1 - c * self.skip
            d = np.abs(f - self._hist[j]) if j >= 0 else np.zeros_like(f)
            chans.append(d)
        return self._to_uint8(chans)              # R = nearest, B = oldest

    def _to_uint8(self, chans) -> np.ndarray:
        rgb = np.stack([np.clip(c * self.gain, 0, 255) for c in chans],
                       axis=-1).astype(np.uint8)
        return rgb

    def reset(self):
        self._prev = None
        self._ema = None
        self._hist = []

    def provenance(self) -> dict:
        return dict(transform="motion-as-colour (after BehaveAI, "
                              "doi:10.1371/journal.pbio.3003632)",
                    strategy=self.strategy, skip=self.skip,
                    decay=list(self.decay), gain=self.gain)


def motion_colour_video(frames, **encoder_kw) -> list[np.ndarray]:
    """Render an iterable/list of frames; convenience for tests and QC."""
    enc = MotionColourEncoder(**encoder_kw)
    return [enc.push(f) for f in frames]
