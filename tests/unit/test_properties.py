"""⛔ P15-5: property tests -- what must hold for *every* input, not just the ones we thought of.

Hypothesis generates the inputs and shrinks any failure to a minimal case. The headline
property is section 6.4's output contract, which the API, the frontend's timeline and every
AP number rely on: **segments are sorted, non-overlapping, and inside `[0, T / fps]`**, for
any score trace and any legal post-processing configuration -- including boundary refinement,
which can move neighbours into each other. (GT-alignment and AP properties live in
`test_gt_alignment.py` and `test_ap_metric.py`.)
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.localization.chunking import chunk_windows, stitch_scores
from src.localization.postprocess import (
    PostProcessConfig,
    assert_segment_contract,
    scores_to_segments,
)

FPS = 25.0

scores_st = st.integers(1, 900).flatmap(
    lambda n: arrays(np.float64, n, elements=st.floats(0.0, 1.0, allow_nan=False, width=32))
)
config_st = st.builds(
    PostProcessConfig,
    threshold=st.floats(0.05, 0.95),
    median_kernel=st.sampled_from([1, 3, 5, 9, 15]),
    closing=st.integers(1, 9),
    min_duration_s=st.floats(0.0, 1.5),
    merge_gap_s=st.floats(0.0, 1.0),
    merge_first=st.booleans(),
    refine_radius=st.integers(0, 6),
)


@settings(max_examples=500, deadline=None)
@given(scores=scores_st, cfg=config_st, data=st.data())
def test_segments_are_sorted_disjoint_and_in_bounds(scores, cfg, data):
    n = len(scores)
    boundary = None
    if cfg.refine_radius:
        boundary = data.draw(arrays(np.float64, (n, 2), elements=st.floats(0.0, 1.0, width=32)))
    segs = scores_to_segments(scores, cfg, boundary_scores=boundary)
    assert_segment_contract(segs, n / FPS)
    for s, e, conf in segs:
        assert 0.0 <= conf <= 1.0
        # Every edge is on the frame grid: no sub-frame boundaries can appear.
        assert abs(s * FPS - round(s * FPS)) < 1e-6 and abs(e * FPS - round(e * FPS)) < 1e-6


@settings(max_examples=300, deadline=None)
@given(scores=scores_st, cfg=config_st)
def test_min_duration_holds_without_refinement(scores, cfg):
    cfg = PostProcessConfig(**{**cfg.as_dict(), "refine_radius": 0})
    for s, e, _ in scores_to_segments(scores, cfg):
        assert (e - s) * FPS >= cfg.min_duration_s * FPS - 1e-9


@settings(max_examples=300, deadline=None)
@given(n=st.integers(1, 900), low=st.floats(0.0, 0.49), cfg=config_st)
def test_all_low_scores_give_no_segments(n, low, cfg):
    """Section 6.8: a real video should come out as []. Strictly below threshold, nothing can."""
    cfg = PostProcessConfig(**{**cfg.as_dict(), "threshold": max(cfg.threshold, low + 1e-3)})
    assert scores_to_segments(np.full(n, low), cfg) == []


@settings(max_examples=200, deadline=None)
@given(scores=scores_st, cfg=config_st)
def test_post_processing_is_deterministic(scores, cfg):
    assert scores_to_segments(scores, cfg) == scores_to_segments(scores.copy(), cfg)


@settings(max_examples=300, deadline=None)
@given(
    n=st.integers(1, 5000),
    chunk=st.integers(20, 800),
    overlap_frac=st.floats(0.0, 0.45),
)
def test_chunk_windows_cover_every_frame_in_order(n, chunk, overlap_frac):
    overlap = int(chunk * overlap_frac)
    windows = chunk_windows(n, chunk, overlap)
    covered = np.zeros(n, dtype=bool)
    prev_start = -1
    for a, b in windows:
        assert 0 <= a < b <= n and b - a <= chunk
        assert a > prev_start
        prev_start = a
        covered[a:b] = True
    assert covered.all()


@settings(max_examples=100, deadline=None)
@given(n=st.integers(1, 2500), seed=st.integers(0, 2**31 - 1))
def test_stitching_a_per_frame_model_is_exact(n, seed):
    """A model whose output at frame t depends only on frame t must stitch to a single pass."""
    x = np.random.default_rng(seed).random(n)
    stitched = stitch_scores(n, lambda a, b: x[a:b] ** 2)
    assert stitched.shape == (n,)
    np.testing.assert_allclose(stitched, x**2, atol=1e-12)
