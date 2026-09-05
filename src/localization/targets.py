"""`fake_periods` (seconds) -> per-frame targets (P7-3/P7-4; section 6.3).

Phase 7 is where the project stops asking *"is this clip fake?"* and starts asking
*"which frames are fake?"* — and that question needs a `[T]` target array beside every
`[T, D]` feature array. This module builds it.

**Section 6.3 calls this "the highest-risk step in the project", and it is.** An off-by-one
here is invisible: training converges, AUC looks fine, and every localization number in
Part 8 is silently wrong by 40 ms. So the mapping is written once, here, and every consumer
uses it rather than re-deriving `int(start * fps)` inline.

**The mapping.** Frame `t` covers the half-open interval `[t/fps, (t+1)/fps)` — 40 ms at
25 fps. A frame is forged if that interval overlaps a labelled span at all:

    frame t is fake  <=>  t/fps < end  and  (t+1)/fps > start
                     <=>  floor(start * fps) <= t < ceil(end * fps)

Overlap, not containment. A span covering the last 30% of a frame still manipulated that
frame's pixels, and rounding it away would systematically shrink every target by up to two
frames — which biases boundary error in one direction and would quietly cap IoU against
the ~16-frame median span (P1-11).

⚠️ **Not yet the full P10-1 contract.** Phase 10 adds the inverse-transform assertion
(target -> intervals must reproduce the original within one frame) and the post-processing
chain that turns scores back into seconds. This module is the forward half only: enough for
Phase 7's frame head and frame AP, and the thing P10-1 will assert *against*.

**`n_frames` comes from the feature array, not the manifest.** `duration` in LAV-DF's
metadata is a padded audio-derived figure (`LAVDF_DURATION_PAD_S`), and the visual cache is
truncated to the common audio/video grid by `MultimodalFeatureDataset`. Labelling `T`
frames when the model sees `T - 3` would misalign every target after the first drop.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from src.config import LAVDF_FPS


def normalise_periods(fake_periods: object) -> list[tuple[float, float]]:
    """Coerce a manifest `fake_periods` cell into a list of `(start, end)` in seconds.

    Parquet round-trips these as a ragged object array of object arrays, so the cell for a
    real video is `array([], dtype=object)` rather than `[]`, and a two-span fake is an
    array of arrays rather than a list of lists. Normalising once here keeps every caller
    from re-discovering that.

    ⛔ Asserts `start < end` and `start >= 0`. A reversed or negative span would produce a
    silently empty target slice — a fake video labelled entirely real — which is precisely
    the failure that never shows up in an aggregate metric.
    """
    if fake_periods is None:
        return []
    if isinstance(fake_periods, float) and math.isnan(fake_periods):
        return []
    if not isinstance(fake_periods, Iterable) or isinstance(fake_periods, str | bytes):
        raise TypeError(f"fake_periods must be a sequence of [start, end], got {fake_periods!r}")

    out: list[tuple[float, float]] = []
    for i, span in enumerate(fake_periods):
        pair = np.asarray(span, dtype=np.float64).ravel()
        if pair.size != 2:
            raise ValueError(f"span {i} has {pair.size} values, expected [start, end]: {span!r}")
        start, end = float(pair[0]), float(pair[1])
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"span {i} is not finite: [{start}, {end}]")
        if start < 0:
            raise ValueError(f"span {i} starts at {start} s, before the clip does")
        if end <= start:
            raise ValueError(f"span {i} is empty or reversed: [{start}, {end}]")
        out.append((start, end))
    return out


def frame_targets(
    fake_periods: object,
    n_frames: int,
    fps: float = LAVDF_FPS,
    *,
    dtype: type = np.float32,
) -> np.ndarray:
    """`[T]` array, 1.0 where frame `t` overlaps a labelled forged span.

    Spans past the end of the array are clipped rather than raising: the visual and audio
    caches are truncated to the common grid, and a span whose tail falls outside that grid
    is a truncation artefact, not a corrupt label.

    ⛔ P10-2 falls out of this by construction: a real video has no spans, so the array is
    exactly zero — `frame_targets([], T).sum() == 0`, not "approximately zero".
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be >= 0, got {n_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    target = np.zeros(int(n_frames), dtype=dtype)
    for start, end in normalise_periods(fake_periods):
        lo = max(0, math.floor(start * fps))
        hi = min(int(n_frames), math.ceil(end * fps))
        if hi > lo:
            target[lo:hi] = 1.0
    return target


def target_positive_rate(targets: Iterable[np.ndarray]) -> float:
    """Fraction of frames that are forged, across a collection of per-clip targets.

    This is the number that sets `frame_pos_weight`. It is *far* smaller than the clip-level
    fake rate — the median span is ~0.65 s against clips averaging ~8 s — so a frame head
    trained on unweighted BCE will happily predict "real" everywhere and score well on loss
    while being useless for localization.
    """
    total, positive = 0, 0.0
    for t in targets:
        arr = np.asarray(t)
        total += arr.size
        positive += float(arr.sum())
    return positive / total if total else 0.0
