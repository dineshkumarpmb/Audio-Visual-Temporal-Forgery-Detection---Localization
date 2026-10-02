"""P10-1 / P10-2 / P10-6 / P10-8 / P10-9 / P10-11 and the Phase 10 losses.

Ground-truth round trip, the section 6.4 chain step by step, the synthetic known-span
recovery test (IoU > 0.9), the segment contract as a property test, chunk stitching, and
the focal / boundary losses.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.localization import interval_iou
from src.localization.chunking import chunk_windows, stitch_scores
from src.localization.postprocess import (
    PostProcessConfig,
    assert_segment_contract,
    close_holes,
    scores_to_segments,
    smooth,
)
from src.localization.targets import (
    check_roundtrip,
    frame_runs,
    frame_targets,
    targets_to_intervals,
)
from src.losses.localization import boundary_loss, boundary_targets, focal_loss_with_logits

FPS = 25.0
PLAIN = PostProcessConfig(median_kernel=1, closing=1, min_duration_s=0.0, merge_gap_s=0.0)


# ---------------------------------------------------------------- ⛔ P10-1 / P10-2


def test_frame_runs():
    assert frame_runs(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    assert frame_runs(np.array([1, 1])) == [(0, 2)]
    assert frame_runs(np.array([], dtype=bool)) == []


def test_inverse_transform_on_frame_aligned_span_is_exact():
    target = frame_targets([(1.0, 2.0)], 100, FPS)  # frames 25..49
    assert targets_to_intervals(target, FPS) == [(1.0, 2.0)]


def test_inverse_widens_to_whole_frames_by_less_than_one_frame():
    target = frame_targets([(1.01, 1.99)], 100, FPS)  # floor(25.25)=25, ceil(49.75)=50
    ((s, e),) = targets_to_intervals(target, FPS)
    assert (s, e) == (1.0, 2.0)
    assert abs(s - 1.01) * FPS < 1 and abs(e - 1.99) * FPS < 1


@pytest.mark.parametrize(
    "periods, n",
    [
        ([(0.0, 0.164)], 200),  # the dataset's shortest span, at the clip start
        ([(3.333, 4.9)], 200),
        ([(1.0, 1.5), (1.5, 2.0)], 200),  # touching spans merge
        ([(1.0, 1.52), (1.5, 2.0)], 200),  # overlapping at frame resolution
        ([(7.5, 9.0)], 200),  # past the truncated grid -> clipped
        ([(0.2, 0.6), (5.0, 6.6)], 200),
    ],
)
def test_check_roundtrip_within_one_frame(periods, n):
    out = check_roundtrip(periods, n, FPS)
    assert out["within_one_frame"], out


def test_real_video_target_is_exactly_zero():
    out = check_roundtrip([], 300, FPS)
    assert out["real_target_exactly_zero"] and out["n_runs"] == 0
    assert frame_targets([], 300, FPS).sum() == 0.0


def test_roundtrip_random_spans():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(50, 500))
        k = int(rng.integers(0, 3))
        periods = []
        for _ in range(k):
            length = float(rng.uniform(0.164, 1.6))
            start = float(rng.uniform(0, n / FPS - length))
            periods.append((start, start + length))
        assert check_roundtrip(periods, n, FPS)["within_one_frame"], periods


# ---------------------------------------------------------------- P10-6, step by step


def test_median_smoothing_removes_single_frame_flicker():
    x = np.zeros(20)
    x[10] = 1.0
    assert smooth(x, 5).max() == 0.0
    assert smooth(x, 1)[10] == 1.0


def test_closing_fills_short_interior_holes_only():
    b = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0], dtype=bool)
    out = close_holes(b, 3)  # holes of length 1 and 3: only the first is < 3
    assert out.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
    edge = np.array([0, 1, 1, 0], dtype=bool)  # edge zeros are not interior holes
    assert close_holes(edge, 5).tolist() == edge.tolist()


def test_threshold_extract_and_seconds():
    x = np.zeros(100)
    x[25:50] = 0.9
    ((s, e, c),) = scores_to_segments(x, PLAIN)
    assert (s, e) == (1.0, 2.0) and c == pytest.approx(0.9)


def test_min_duration_and_merge_gap():
    x = np.zeros(200)
    x[10:14] = 0.9  # 4 frames = 0.16 s
    x[50:70] = 0.9  # 20 frames
    x[74:90] = 0.9  # gap of 4 frames = 0.16 s
    cfg = PostProcessConfig(median_kernel=1, closing=1, min_duration_s=0.4, merge_gap_s=0.3)
    segs = scores_to_segments(x, cfg)
    assert [(round(s, 2), round(e, 2)) for s, e, _ in segs] == [(2.0, 3.6)]


def test_merge_first_keeps_fragments_of_one_span():
    # Two 8-frame fragments (0.32 s each) with a 2-frame gap. Section 6.4's order drops
    # both before merging; merge_first joins them into one 18-frame segment.
    x = np.zeros(100)
    x[20:28] = 0.9
    x[30:38] = 0.9
    base = dict(median_kernel=1, closing=1, min_duration_s=0.4, merge_gap_s=0.3)
    assert scores_to_segments(x, PostProcessConfig(**base)) == []
    ((s, e, _),) = scores_to_segments(x, PostProcessConfig(**base, merge_first=True))
    assert (s, e) == (0.8, 1.52)


def test_confidence_is_mean_raw_score():
    x = np.zeros(50)
    x[10:20] = np.linspace(0.6, 1.0, 10)
    ((_, _, c),) = scores_to_segments(x, PLAIN)
    assert c == pytest.approx(0.8)


def test_real_video_gives_no_segments():
    rng = np.random.default_rng(1)
    assert scores_to_segments(rng.uniform(0, 0.3, 300)) == []


def test_boundary_refinement_snaps_to_peaks():
    x = np.zeros(100)
    x[22:52] = 0.9  # the frame head is 3 frames early at the start, 2 late at the end
    bnd = np.zeros((100, 2))
    bnd[25, 0] = 1.0  # true first frame
    bnd[49, 1] = 1.0  # true last frame -> end = 50
    cfg = PostProcessConfig(
        median_kernel=1, closing=1, min_duration_s=0.0, merge_gap_s=0.0, refine_radius=4
    )
    ((s, e, _),) = scores_to_segments(x, cfg, boundary_scores=bnd)
    assert (s, e) == (1.0, 2.0)
    with pytest.raises(ValueError):
        scores_to_segments(x, cfg, boundary_scores=np.zeros((10, 2)))


def test_config_validation():
    with pytest.raises(ValueError):
        PostProcessConfig(median_kernel=4)
    with pytest.raises(ValueError):
        PostProcessConfig(threshold=1.5)


# ---------------------------------------------------------------- ⛔ P10-8 synthetic span


def test_synthetic_known_span_recovered_with_iou_above_0_9():
    """⛔ P10-8, under the plan's *default* chain and noisy scores."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        t = int(rng.integers(100, 500))
        length = int(rng.integers(12, 41))  # 0.48-1.6 s, the dataset's upper range
        start = int(rng.integers(0, t - length))
        scores = rng.uniform(0.0, 0.35, t)
        scores[start : start + length] = rng.uniform(0.65, 1.0, length)
        segs = scores_to_segments(scores, PostProcessConfig())
        assert len(segs) == 1
        truth = (start / FPS, (start + length) / FPS)
        assert interval_iou(segs[0][:2], truth) > 0.9


# ---------------------------------------------------------------- ⛔ P10-9 properties


def test_segment_contract_holds_on_random_scores():
    rng = np.random.default_rng(7)
    for _ in range(500):
        t = int(rng.integers(1, 400))
        # Blocky random scores, so there are many candidate segments to break the contract.
        scores = np.repeat(rng.uniform(0, 1, t // 5 + 1), 5)[:t]
        bnd = rng.uniform(0, 1, (t, 2))
        cfg = PostProcessConfig(
            threshold=float(rng.uniform(0.1, 0.9)),
            median_kernel=int(rng.choice([1, 3, 5, 9])),
            closing=int(rng.integers(1, 6)),
            min_duration_s=float(rng.uniform(0, 1)),
            merge_gap_s=float(rng.uniform(0, 1)),
            merge_first=bool(rng.integers(0, 2)),
            refine_radius=int(rng.integers(0, 6)),
        )
        segs = scores_to_segments(scores, cfg, boundary_scores=bnd)
        assert_segment_contract(segs, t / FPS)


def test_contract_checker_rejects_violations():
    with pytest.raises(AssertionError):
        assert_segment_contract([(1.0, 2.0), (1.5, 3.0)], 10)  # overlap
    with pytest.raises(AssertionError):
        assert_segment_contract([(2.0, 3.0), (0.0, 1.0)], 10)  # unsorted
    with pytest.raises(AssertionError):
        assert_segment_contract([(9.0, 10.5)], 10)  # past the end


# ---------------------------------------------------------------- P10-11 chunking


def test_chunk_windows_cover_everything():
    assert chunk_windows(300) == [(0, 300)]
    for n in (751, 1000, 1375, 2000, 5003):
        w = chunk_windows(n)
        covered = np.zeros(n, dtype=int)
        for a, b in w:
            assert b - a == 750
            covered[a:b] += 1
        assert covered.min() >= 1 and w[-1][1] == n and w[0][0] == 0
        # consecutive windows overlap by at least the configured 125 frames
        assert all(w[i][1] - w[i + 1][0] >= 125 for i in range(len(w) - 1))


def test_stitched_scores_match_a_single_pass_for_a_local_model():
    rng = np.random.default_rng(3)
    truth = rng.uniform(0, 1, 2000)
    stitched = stitch_scores(2000, lambda a, b: truth[a:b])
    np.testing.assert_allclose(stitched, truth)
    two = stitch_scores(2000, lambda a, b: np.stack([truth[a:b], 1 - truth[a:b]], 1))
    assert two.shape == (2000, 2)


def test_overlaps_are_averaged_and_post_processed_once():
    # A model whose output depends on which chunk it sees: +0.1 in even-numbered chunks.
    # Frames covered by one even and one odd chunk must land exactly halfway (+0.05).
    n = 1600  # windows [0,750) [625,1375) [850,1600)
    base = np.zeros(n)
    base[720:780] = 0.8  # straddles the 750-frame edge of the first window
    windows = chunk_windows(n)

    def predict(a, b):
        k = windows.index((a, b))
        return base[a:b] + (0.1 if k % 2 == 0 else 0.0)

    stitched = stitch_scores(n, predict)
    assert stitched[100] == pytest.approx(0.1)  # only window 0
    assert stitched[700] == pytest.approx(0.05)  # windows 0 and 1
    assert stitched[1000] == pytest.approx(0.05)  # windows 1 and 2
    assert stitched[1500] == pytest.approx(0.1)  # only window 2
    # Post-processed once on the stitched sequence: the straddling span stays ONE segment.
    segs = scores_to_segments(stitched, PostProcessConfig())
    assert [(round(s, 2), round(e, 2)) for s, e, _ in segs] == [(28.8, 31.2)]


# ---------------------------------------------------------------- P10-3 / P10-4 losses


def test_focal_loss_masks_padding_and_normalises_by_positives():
    logits = torch.tensor([[2.0, -2.0, 5.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    base = focal_loss_with_logits(logits, targets, mask)
    padded = logits.clone()
    padded[0, 3] = 50.0  # garbage in a padded slot must not move the loss
    assert torch.isclose(focal_loss_with_logits(padded, targets, mask), base)
    # Hand value for frame 0: p = sigmoid(2), alpha 0.25, (1-p)^2 * -log p, / 1 positive.
    p = torch.sigmoid(torch.tensor(2.0))
    f0 = 0.25 * (1 - p) ** 2 * -torch.log(p)
    q = torch.sigmoid(torch.tensor(-2.0))
    f1 = 0.75 * q**2 * -torch.log(1 - q)
    r = torch.sigmoid(torch.tensor(5.0))
    f2 = 0.75 * r**2 * -torch.log(1 - r)
    assert torch.isclose(base, f0 + f1 + f2, atol=1e-6)


def test_focal_reduces_to_weighted_bce_at_gamma_zero():
    torch.manual_seed(0)
    logits, targets = torch.randn(2, 30), (torch.rand(2, 30) > 0.8).float()
    mask = torch.ones(2, 30, dtype=torch.bool)
    fl = focal_loss_with_logits(logits, targets, mask, alpha=0.5, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    assert torch.isclose(fl, 0.5 * bce / targets.sum(), atol=1e-5)


def test_boundary_targets_peak_on_first_and_last_forged_frame():
    frames = torch.zeros(1, 40)
    frames[0, 10:20] = 1.0
    mask = torch.ones(1, 40, dtype=torch.bool)
    heat = boundary_targets(frames, mask, sigma=2.0)
    assert heat.shape == (1, 40, 2)
    assert int(heat[0, :, 0].argmax()) == 10 and heat[0, 10, 0] == 1.0
    assert int(heat[0, :, 1].argmax()) == 19 and heat[0, 19, 1] == 1.0
    real = boundary_targets(torch.zeros(1, 40), mask)
    assert real.sum() == 0.0


def test_boundary_loss_prefers_peaks_in_the_right_place():
    frames = torch.zeros(1, 40)
    frames[0, 10:20] = 1.0
    mask = torch.ones(1, 40, dtype=torch.bool)
    heat = boundary_targets(frames, mask)
    good = torch.full((1, 40, 2), -6.0)
    good[0, 10, 0] = good[0, 19, 1] = 6.0
    bad = torch.full((1, 40, 2), -6.0)
    bad[0, 30, 0] = bad[0, 5, 1] = 6.0
    assert boundary_loss(good, heat, mask) < boundary_loss(bad, heat, mask)


def test_float_error_does_not_move_a_boundary_by_a_frame():
    """⛔ P10-1's finding: 1.16 * 25 == 28.999999999999996, and plain floor() made frame 28
    forged. 2,938 of the manifest's 228,506 span boundaries were one frame off this way."""
    assert 1.16 * 25 < 29
    t = frame_targets([(1.16, 1.6)], 100, FPS)
    assert t[28] == 0.0 and t[29] == 1.0 and t[39] == 1.0 and t[40] == 0.0
    assert check_roundtrip([(1.16, 1.6)], 100, FPS)["worst_error_frames"] < 1e-6
