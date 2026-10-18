"""⛔ P15-2: ground-truth alignment -- `fake_periods` (seconds) <-> per-frame targets.

Section 6.3 calls this transform "the highest-risk step in the project": an off-by-one is
invisible in training and shifts every localization number by 40 ms. This file is the one the
plan says "protects every localization number", so it does not trust the implementation to
test itself:

* **An independent exact reference.** Frame `t` covers `[t/fps, (t+1)/fps)` and is forged iff
  that interval overlaps a span. The reference evaluates exactly that inequality in rational
  arithmetic (`fractions.Fraction` from the label's millisecond value), so float error -- the
  `1.16 * 25 = 28.999999999999996` class of bug P10-1 found on 2,938 boundaries -- cannot
  hide in it. `frame_targets` must agree with it on *every* millisecond-grid label, which is
  the resolution LAV-DF labels are stored at.
* **Real labels.** `tests/fixtures/lavdf_labels_sample.json` holds 450 real manifest entries,
  150 of them chosen *because* a boundary sits within float error of a frame edge. Both the
  reference and the round trip run on those, so this works without the dataset (CI).
* **Both directions, and both consumers.** Prediction and label must meet on the same frame
  grid: post-processing a perfect score trace must return exactly the labelled frames, and
  `evaluation.runs.ground_truth` must clip to what the model saw the same way.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.localization.postprocess import PostProcessConfig, scores_to_segments
from src.localization.targets import (
    check_roundtrip,
    frame_range,
    frame_targets,
    normalise_periods,
    targets_to_intervals,
)
from tests.regression import LABELS_SAMPLE

FPS = 25


def reference_targets(spans_ms: list[tuple[int, int]], n_frames: int, fps: int = FPS) -> np.ndarray:
    """Exact: frame t is forged iff t/fps < end and (t+1)/fps > start. No floats anywhere."""
    out = np.zeros(n_frames, dtype=np.float32)
    for s_ms, e_ms in spans_ms:
        s, e = Fraction(s_ms, 1000), Fraction(e_ms, 1000)
        for t in range(n_frames):
            if Fraction(t, fps) < e and Fraction(t + 1, fps) > s:
                out[t] = 1.0
    return out


def to_ms(x: float) -> int:
    ms = round(x * 1000)
    assert abs(ms / 1000 - x) < 1e-9, f"{x} is not on the millisecond grid"
    return ms


# ---------------------------------------------------------------- the frame rule, by hand


class TestFrameRule:
    def test_frame_aligned_span_marks_exactly_its_frames(self):
        t = frame_targets([(1.0, 2.0)], 100)
        assert np.flatnonzero(t).tolist() == list(range(25, 50))

    def test_overlap_not_containment(self):
        """A span touching 1 ms of a frame forged that frame's pixels (section 6.3)."""
        t = frame_targets([(0.039, 0.041)], 10)  # straddles the 0/1 frame edge at 40 ms
        assert np.flatnonzero(t).tolist() == [0, 1]

    def test_span_ending_on_a_frame_edge_does_not_mark_the_next_frame(self):
        t = frame_targets([(0.0, 0.08)], 10)  # exactly two frames
        assert np.flatnonzero(t).tolist() == [0, 1]

    def test_the_p10_1_float_case(self):
        """`1.16 * 25` is 28.999999999999996 in floats; frame 28 must stay real."""
        assert 1.16 * 25 != 29.0  # the hazard is real on this platform
        assert frame_range(1.16, 1.6, FPS, 100) == (29, 40)
        t = frame_targets([(1.16, 1.6)], 100)
        assert t[28] == 0.0 and t[29] == 1.0 and t[39] == 1.0 and t[40] == 0.0

    def test_real_video_is_exactly_zero(self):
        assert frame_targets([], 300).sum() == 0.0
        assert frame_targets(np.array([], dtype=object), 300).sum() == 0.0  # parquet's empty cell

    def test_spans_past_the_feature_grid_are_clipped_not_dropped(self):
        t = frame_targets([(1.9, 2.5)], 50)  # the grid ends at 2.0 s
        assert np.flatnonzero(t).tolist() == [47, 48, 49]
        assert frame_targets([(3.0, 4.0)], 50).sum() == 0.0  # wholly past it

    def test_overlapping_spans_union(self):
        t = frame_targets([(0.0, 0.2), (0.1, 0.4)], 20)
        assert np.flatnonzero(t).tolist() == list(range(10))

    @pytest.mark.parametrize(
        "bad", [[(1.0, 1.0)], [(2.0, 1.0)], [(-0.1, 1.0)], [(0.0, float("nan"))], [(1.0,)]]
    )
    def test_corrupt_labels_fail_loudly(self, bad):
        """A reversed or empty span would silently label a fake video real."""
        with pytest.raises((ValueError, TypeError)):
            frame_targets(bad, 100)


# ---------------------------------------------------------------- against the exact reference


class TestAgainstExactReference:
    def test_every_millisecond_start_lands_on_the_right_frame(self):
        """Exhaustive over 0-12 s: the first forged frame for every possible label start."""
        for s_ms in range(0, 12_000):
            lo, _ = frame_range(s_ms / 1000, s_ms / 1000 + 0.5, FPS, 10**6)
            assert lo == (s_ms * FPS) // 1000, s_ms  # floor(s * fps), exactly

    def test_every_millisecond_end_lands_on_the_right_frame(self):
        for e_ms in range(1, 12_000):
            _, hi = frame_range(max(e_ms / 1000 - 0.5, 0.0), e_ms / 1000, FPS, 10**6)
            assert hi == -((-e_ms * FPS) // 1000), e_ms  # ceil(e * fps), exactly

    @settings(max_examples=300, deadline=None)
    @given(
        spans=st.lists(
            st.tuples(st.integers(0, 9_000), st.integers(1, 2_000)).map(
                lambda p: (p[0], p[0] + p[1])
            ),
            min_size=0,
            max_size=3,
        ),
        n_frames=st.integers(1, 260),
    )
    def test_random_ms_labels_match_the_reference(self, spans, n_frames):
        floats = [(s / 1000, e / 1000) for s, e in spans]
        np.testing.assert_array_equal(
            frame_targets(floats, n_frames), reference_targets(spans, n_frames)
        )


# ---------------------------------------------------------------- real LAV-DF labels


@pytest.fixture(scope="module")
def sample() -> dict:
    return json.loads(LABELS_SAMPLE.read_text(encoding="utf-8"))


class TestRealLabels:
    def test_sample_is_what_it_claims(self, sample):
        entries = sample["entries"]
        assert len(entries) >= 400
        assert {e["split"] for e in entries} == {"train", "dev", "test"}
        assert sum(e["label"] == 0 for e in entries) > 50  # real videos are in it
        assert sum(len(e["fake_periods"]) == 2 for e in entries) > 5  # and two-span fakes
        assert sample["n_float_edge_in_sample"] >= 100

    def test_targets_match_the_exact_reference_on_real_labels(self, sample):
        for e in sample["entries"]:
            spans_ms = [(to_ms(s), to_ms(t)) for s, t in e["fake_periods"]]
            got = frame_targets(e["fake_periods"], e["n_frames"], e["fps"])
            np.testing.assert_array_equal(
                got, reference_targets(spans_ms, e["n_frames"], int(e["fps"])), e["video_id"]
            )

    def test_round_trip_within_one_frame_on_real_labels(self, sample):
        """⛔ P10-1 on real values; P10-2 (real target exactly 0) on every real video."""
        for e in sample["entries"]:
            out = check_roundtrip(e["fake_periods"], e["n_frames"], e["fps"])
            assert out["within_one_frame"], (e["video_id"], out)
            assert out["real_target_exactly_zero"], e["video_id"]

    def test_float_edge_labels_are_not_shifted(self, sample):
        """The 150 labels picked because `x * 25` lands within 1e-6 of an integer."""
        checked = 0
        for e in sample["entries"]:
            for s, t in e["fake_periods"]:
                lo, hi = frame_range(s, t, e["fps"], 10**6)
                assert lo == math.floor(Fraction(to_ms(s), 1000) * 25), (e["video_id"], s)
                assert hi == math.ceil(Fraction(to_ms(t), 1000) * 25), (e["video_id"], t)
                checked += 1
        assert checked > 300


# ---------------------------------------------------------------- prediction meets label


class TestPredictionAndLabelShareTheGrid:
    @settings(max_examples=150, deadline=None)
    @given(
        spans=st.lists(
            st.tuples(st.integers(0, 7_000), st.integers(41, 1_500)).map(
                lambda p: (p[0], p[0] + p[1])
            ),
            min_size=0,
            max_size=3,
        ),
        n_frames=st.integers(5, 250),
    )
    def test_perfect_scores_post_process_to_exactly_the_labelled_frames(self, spans, n_frames):
        """With smoothing, closing, min-duration and merging switched off, a score trace equal
        to the target must come back as exactly `targets_to_intervals(target)` -- so a perfect
        model scores IoU 1, and any boundary error is the model's, never the plumbing's."""
        target = frame_targets([(s / 1000, e / 1000) for s, e in spans], n_frames)
        identity = PostProcessConfig(
            threshold=0.5, median_kernel=1, closing=1, min_duration_s=0.0, merge_gap_s=0.0
        )
        segs = [(s, e) for s, e, _ in scores_to_segments(target, identity)]
        assert segs == targets_to_intervals(target)

    @settings(max_examples=200, deadline=None)
    @given(binary=st.lists(st.booleans(), min_size=1, max_size=300))
    def test_intervals_back_to_targets_is_the_identity(self, binary):
        t = np.asarray(binary, dtype=np.float32)
        np.testing.assert_array_equal(frame_targets(targets_to_intervals(t), len(t)), t)

    def test_evaluation_ground_truth_clips_to_what_the_model_saw(self):
        """`runs.ground_truth` must clip at `T / fps`, the same grid `frame_targets` clips at."""
        import pandas as pd

        from src.evaluation.runs import ground_truth

        manifest = pd.DataFrame(
            {"fake_periods": [[[1.5, 2.6]], [[3.0, 3.5]], []], "fps": [25.0] * 3},
            index=["a", "b", "c"],
        )
        gt = ground_truth(manifest, {"a": 50, "b": 50, "c": 50})  # the model saw 2.0 s
        assert gt == {"a": [(1.5, 2.0)], "b": [], "c": []}
        # Training targets widen to whole frames (1.5 s is mid-frame 37, which starts at
        # 1.48 s); evaluation keeps exact seconds. They must agree to within a frame, and on
        # exactly where the clip ends.
        for vid in ("a", "b"):
            intervals = targets_to_intervals(
                frame_targets(normalise_periods(manifest.loc[vid, "fake_periods"]), 50)
            )
            assert len(intervals) == len(gt[vid])
            for (ts, te), (gs, ge) in zip(intervals, gt[vid], strict=True):
                assert abs(ts - gs) < 1 / FPS and abs(te - ge) < 1 / FPS
                assert te <= 50 / FPS and ge <= 50 / FPS
