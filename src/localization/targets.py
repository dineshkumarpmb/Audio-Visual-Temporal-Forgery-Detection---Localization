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

**Both directions live here (P10-1).** `targets_to_intervals` is the inverse: a `[T]`
target back to seconds, and `check_roundtrip` asserts that the pair reproduces every labelled
span to within one frame. `scripts/32_check_gt_alignment.py` runs that assertion over the
whole manifest. The post-processing chain that turns *scores* into seconds uses the same
`frame_runs` -> `i / fps` convention (`src/localization/postprocess.py`), so a prediction and
its ground truth can never disagree by construction about where a frame begins.

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

# ⛔ P10-1 found this. `1.16 * 25` is 28.999999999999996, so `floor` put the span's first
# frame at 28 instead of 29 -- a forged frame that is not forged. Across the manifest,
# 1,888 span starts landed one frame early and 1,050 ends one frame late (2,938 of 228,506
# boundaries, every split). A frame boundary within 1e-6 frames (40 ns) of an integer is
# that integer: no label is that precise, and the snap only ever removes float error.
FRAME_SNAP = 1e-6


def frame_range(start: float, end: float, fps: float, n_frames: int) -> tuple[int, int]:
    """Half-open frame range `[i0, i1)` a `(start, end)` span in seconds overlaps, clipped.

    The one place the seconds -> frames rule lives; `frame_targets` and `check_roundtrip`
    both call it, so the transform and its test cannot drift apart.
    """
    lo = math.floor(start * fps + FRAME_SNAP)
    hi = math.ceil(end * fps - FRAME_SNAP)
    return max(0, lo), min(int(n_frames), hi)


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
        lo, hi = frame_range(start, end, fps, n_frames)
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


def frame_runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as half-open frame ranges `[i0, i1)`."""
    b = np.asarray(binary).astype(bool).ravel()
    if not b.size:
        return []
    edges = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(i0), int(i1)) for i0, i1 in zip(starts, ends, strict=True)]


def targets_to_intervals(target: np.ndarray, fps: float = LAVDF_FPS) -> list[tuple[float, float]]:
    """⛔ P10-1, the inverse of `frame_targets`: `[T]` target -> `(start, end)` in seconds.

    A run of forged frames `[i0, i1)` covers `[i0/fps, i1/fps)`, the exact union of the
    frames' own intervals. So the round trip widens a span to whole frames -- never by more
    than one frame at either end, which is the contract `check_roundtrip` asserts.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return [(i0 / fps, i1 / fps) for i0, i1 in frame_runs(np.asarray(target) > 0.5)]


def check_roundtrip(
    fake_periods: object, n_frames: int, fps: float = LAVDF_FPS
) -> dict[str, float | int | bool]:
    """⛔ P10-1: labelled spans -> target -> intervals must reproduce the labels within a frame.

    Two legitimate reasons the recovered list can differ in *shape* from the labels, both
    handled rather than treated as failures:

    * spans that overlap or touch at frame resolution merge into one run -- the target
      cannot represent a zero-frame gap, so the expected answer merges them too;
    * a span past `n_frames` is clipped (see `frame_targets`) and is compared against its
      clipped extent.

    Returns the worst boundary error in frames and whether every error is `< 1` frame.
    Also carries ⛔ P10-2: for a video with no spans, `target.sum()` must be exactly 0.
    """
    spans = normalise_periods(fake_periods)
    target = frame_targets(spans, n_frames, fps)
    recovered = targets_to_intervals(target, fps)

    # Expected: each span's own frame range, clipped, then merged where they touch.
    ranges = sorted(frame_range(s, e, fps, n_frames) for s, e in spans)
    merged: list[list[float]] = []  # [i0, i1, earliest start_s, latest end_s]
    for (i0, i1), (s, e) in zip(ranges, sorted(spans), strict=True):
        if i1 <= i0:
            continue  # entirely past the truncated grid
        if merged and i0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], i1)
            merged[-1][3] = max(merged[-1][3], e)
        else:
            merged.append([i0, i1, s, e])

    worst = 0.0
    shape_ok = len(recovered) == len(merged)
    if shape_ok:
        for (rs, re_), (i0, i1, s, e) in zip(recovered, merged, strict=True):
            if abs(rs - i0 / fps) > 1e-9 or abs(re_ - i1 / fps) > 1e-9:
                shape_ok = False
            # Seconds error against the label, unless the label was clipped at the grid end.
            worst = max(worst, abs(rs - s) * fps)
            if e * fps <= n_frames:
                worst = max(worst, abs(re_ - e) * fps)
    real_exact = bool(spans) or float(target.sum()) == 0.0
    return {
        "n_spans": len(spans),
        "n_runs": len(recovered),
        "worst_error_frames": worst,
        "within_one_frame": bool(shape_ok and worst < 1.0),
        "real_target_exactly_zero": real_exact,
    }
