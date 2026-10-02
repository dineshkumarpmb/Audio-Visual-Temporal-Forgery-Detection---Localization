"""Long-video chunking (P10-11, section 6.9).

    750-frame chunks, 125-frame overlap, average per-frame scores where chunks overlap,
    then post-process ONCE on the stitched full-length sequence.

**Why the stitching is separate from post-processing.** Running section 6.4's chain per
chunk would put an artificial segment boundary at every chunk edge -- a span crossing the
30 s mark would come out as two segments, each with half the IoU. So this module returns
*scores*, never segments, and the caller post-processes the result once.

⚠️ **On LAV-DF this path is never exercised by real data.** Phase 1 measured the longest clip (`reports/dataset_statistics.md`)
at 19.88 s = 497 frames, under one 750-frame chunk. It exists for the API (Phase 13), where
uploads can be any length, and it is verified on synthetic sequences instead: a stitched run
must agree with a single full-length pass wherever the model is local, and must never leave
a frame uncovered.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

CHUNK_FRAMES = 750
OVERLAP_FRAMES = 125


def chunk_windows(
    n_frames: int, chunk: int = CHUNK_FRAMES, overlap: int = OVERLAP_FRAMES
) -> list[tuple[int, int]]:
    """Half-open `[start, end)` windows covering `[0, n_frames)`.

    Windows step by `chunk - overlap`. The last one is pulled back to end exactly at
    `n_frames`, so it is always a full chunk (when the clip allows) rather than a sliver
    whose scores would come from almost no context.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk}")
    if not 0 <= overlap < chunk:
        raise ValueError(f"overlap must be in [0, chunk), got {overlap}")
    if n_frames <= 0:
        return []
    if n_frames <= chunk:
        return [(0, n_frames)]
    step = chunk - overlap
    windows = []
    start = 0
    while start + chunk < n_frames:
        windows.append((start, start + chunk))
        start += step
    windows.append((n_frames - chunk, n_frames))
    return windows


def stitch_scores(
    n_frames: int,
    predict: Callable[[int, int], np.ndarray],
    chunk: int = CHUNK_FRAMES,
    overlap: int = OVERLAP_FRAMES,
) -> np.ndarray:
    """Run `predict(start, end) -> [end - start] scores` per window and average overlaps.

    `predict` may return `(end - start,)` or `(end - start, k)` -- the boundary head's two
    channels stitch the same way as the frame head's one.
    """
    windows = chunk_windows(n_frames, chunk, overlap)
    total: np.ndarray | None = None
    count = np.zeros(n_frames, dtype=np.float64)
    for start, end in windows:
        part = np.asarray(predict(start, end), dtype=np.float64)
        if part.shape[0] != end - start:
            raise ValueError(f"predict({start}, {end}) returned {part.shape[0]} frames")
        if total is None:
            total = np.zeros((n_frames, *part.shape[1:]), dtype=np.float64)
        total[start:end] += part
        count[start:end] += 1
    if total is None:
        return np.zeros(0)
    if (count == 0).any():  # cannot happen by construction; asserted because it is silent
        raise AssertionError("a frame was covered by no chunk")
    return total / count.reshape(-1, *([1] * (total.ndim - 1)))
