"""Per-frame scores -> forged segments in seconds (P10-6, section 6.4).

    1. SMOOTH      median filter, `median_kernel` frames (1 = off)
    2. THRESHOLD   binary = smoothed > `threshold`
    3. MORPHOLOGY  binary closing: fill interior holes shorter than `closing` frames
    4. EXTRACT     contiguous runs of 1 -> candidate segments
    5. MIN-DURATION  discard segments shorter than `min_duration_s`
    6. MERGE-GAPS  merge segments separated by less than `merge_gap_s`
    7. CONFIDENCE  per segment = mean *raw* score over its frames
    8. TO SECONDS  start = i0 / fps, end = i1 / fps
   (9. REFINE      optional: snap each boundary to the boundary head's peak within
                   `refine_radius` frames -- only when a boundary head exists)

**The seconds convention is shared with the ground truth, not re-derived.** A run of frames
`[i0, i1)` becomes `[i0/fps, i1/fps)` -- exactly what `targets_to_intervals` does to a label.
So a prediction that recovers a target frame-perfectly scores IoU 1.0 against it, and any
residual error is the model's, never a disagreement about where a frame begins.

⚠️ **Section 6.4's defaults are not safe on this dataset, and the grid search knows it.**
`min_duration_s = 0.4` is the plan's default, but P1-11 measured the forged spans: p25 is
0.40 s and the minimum 0.164 s. The default would therefore discard roughly a quarter of all
true forgeries *before* they could be scored. Section 6.5's tuning range starts at 0.1 s,
and `scripts/35_experiment_g.py` searches from 0 upward -- the frozen value is whatever dev
says, not the plan's guess.

**Steps 5 and 6 are order-sensitive**, and section 6.4 lists min-duration first. That drops
two short fragments of one span before they could be merged into a segment long enough to
keep. `merge_first` swaps the order; it is a searched parameter rather than a silent
deviation, so the frozen config records which one dev preferred.

**Closing is defined on interior holes only.** Textbook binary closing with a size-`k`
element fills gaps of up to `k - 1` frames between two positive runs; at the clip edges its
behaviour depends on the padding convention. Filling only holes bounded on both sides makes
the operation exact, border-independent and trivially testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter

from src.config import LAVDF_FPS
from src.localization.targets import frame_runs


@dataclass(frozen=True)
class PostProcessConfig:
    """Section 6.5's parameters. Tune on dev, freeze, never touch again."""

    threshold: float = 0.5
    median_kernel: int = 5
    closing: int = 3
    min_duration_s: float = 0.4
    merge_gap_s: float = 0.3
    merge_first: bool = False
    refine_radius: int = 0  # frames; 0 = no boundary refinement
    fps: float = LAVDF_FPS

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.median_kernel < 1 or self.median_kernel % 2 == 0:
            raise ValueError(f"median_kernel must be a positive odd int, got {self.median_kernel}")
        if self.closing < 1:
            raise ValueError(f"closing must be >= 1 (1 = off), got {self.closing}")
        if self.min_duration_s < 0 or self.merge_gap_s < 0:
            raise ValueError("min_duration_s and merge_gap_s must be >= 0")
        if self.refine_radius < 0:
            raise ValueError(f"refine_radius must be >= 0, got {self.refine_radius}")
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")

    def as_dict(self) -> dict:
        return asdict(self)


def smooth(scores: np.ndarray, kernel: int) -> np.ndarray:
    """Step 1. Median, not mean: it removes one-frame flicker without blurring edges."""
    scores = np.asarray(scores, dtype=np.float64)
    if kernel <= 1 or scores.size == 0:
        return scores.copy()
    return median_filter(scores, size=kernel, mode="nearest")


def close_holes(binary: np.ndarray, closing: int) -> np.ndarray:
    """Step 3. Fill interior runs of 0 shorter than `closing` frames (1 = off)."""
    out = np.asarray(binary).astype(bool).copy()
    if closing <= 1 or not out.any():
        return out
    for i0, i1 in frame_runs(~out):
        interior = i0 > 0 and i1 < len(out)
        if interior and (i1 - i0) < closing:
            out[i0:i1] = True
    return out


def _drop_short(runs: list[tuple[int, int]], min_frames: float) -> list[tuple[int, int]]:
    # A small tolerance so that e.g. 10 frames at 25 fps satisfies min_duration 0.4 s exactly.
    return [(a, b) for a, b in runs if (b - a) >= min_frames - 1e-9]


def _merge(runs: list[tuple[int, int]], gap_frames: float) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) < gap_frames - 1e-9:
            merged[-1][1] = max(merged[-1][1], b)  # refined runs may nest
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def refine_boundaries(
    runs: list[tuple[int, int]],
    start_scores: np.ndarray,
    end_scores: np.ndarray,
    radius: int,
) -> list[tuple[int, int]]:
    """Step 9. Snap each boundary to the boundary head's strongest response nearby.

    `start_scores[t]` is the start-ness of frame `t` (the first forged frame);
    `end_scores[t]` is the end-ness of frame `t` (the last forged frame). A segment is never
    allowed to invert or collapse: if refinement would make it empty it is left unchanged.
    """
    if radius <= 0:
        return list(runs)
    t = len(start_scores)
    out = []
    for a, b in runs:
        lo, hi = max(0, a - radius), min(t, a + radius + 1)
        new_a = lo + int(np.argmax(start_scores[lo:hi]))
        last = b - 1
        lo, hi = max(0, last - radius), min(t, last + radius + 1)
        new_b = lo + int(np.argmax(end_scores[lo:hi])) + 1
        out.append((new_a, new_b) if new_b > new_a else (a, b))
    return out


def scores_to_segments(
    scores: np.ndarray,
    cfg: PostProcessConfig | None = None,
    *,
    boundary_scores: np.ndarray | None = None,
    smoothed: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """⭐ The chain. `[T]` probabilities -> `[(start_s, end_s, confidence), ...]`.

    Output is sorted by start, non-overlapping, and within `[0, T / fps]` (P10-9 asserts
    all three as properties). A real video should come out as `[]` -- section 6.8.

    `smoothed` lets a grid search pass step 1's output in, computed once per kernel rather
    than once per configuration. It must be `smooth(scores, cfg.median_kernel)`.
    """
    cfg = cfg or PostProcessConfig()
    raw = np.asarray(scores, dtype=np.float64).ravel()
    if raw.size == 0:
        return []
    if np.isnan(raw).any():
        raise ValueError("scores contain NaN")

    if smoothed is None:
        smoothed = smooth(raw, cfg.median_kernel)
    elif len(smoothed) != raw.size:
        raise ValueError(f"smoothed has {len(smoothed)} frames, scores {raw.size}")
    binary = close_holes(smoothed > cfg.threshold, cfg.closing)
    runs = frame_runs(binary)

    min_frames = cfg.min_duration_s * cfg.fps
    gap_frames = cfg.merge_gap_s * cfg.fps
    if cfg.merge_first:
        runs = _drop_short(_merge(runs, gap_frames), min_frames)
    else:
        runs = _merge(_drop_short(runs, min_frames), gap_frames)

    if cfg.refine_radius and boundary_scores is not None:
        bnd = np.asarray(boundary_scores, dtype=np.float64)
        if bnd.shape != (raw.size, 2):
            raise ValueError(f"boundary_scores must be ({raw.size}, 2), got {bnd.shape}")
        runs = refine_boundaries(runs, bnd[:, 0], bnd[:, 1], cfg.refine_radius)
        # Refinement can make neighbours touch or overlap; re-merge so the output contract
        # (sorted, non-overlapping) holds regardless.
        runs = _merge(sorted(runs), 1e-6)

    return [(a / cfg.fps, b / cfg.fps, float(raw[a:b].mean())) for a, b in runs]


def segments_for_split(
    frame_scores: dict[str, np.ndarray],
    cfg: PostProcessConfig,
    boundary_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    """Apply the chain to every clip. Pure function of saved predictions (section L)."""
    return {
        vid: scores_to_segments(
            s, cfg, boundary_scores=None if boundary_scores is None else boundary_scores.get(vid)
        )
        for vid, s in frame_scores.items()
    }


def assert_segment_contract(
    segments: Sequence[Sequence[float]], duration_s: float, *, tol: float = 1e-9
) -> None:
    """⛔ P10-9: sorted by start, non-overlapping, inside `[0, duration]`."""
    prev_end = -np.inf
    for i, seg in enumerate(segments):
        s, e = float(seg[0]), float(seg[1])
        if not e > s:
            raise AssertionError(f"segment {i} is empty or reversed: {seg}")
        if s < -tol or e > duration_s + tol:
            raise AssertionError(f"segment {i} {seg} outside [0, {duration_s}]")
        if s < prev_end - tol:
            raise AssertionError(f"segment {i} {seg} overlaps or is out of order")
        prev_end = e
