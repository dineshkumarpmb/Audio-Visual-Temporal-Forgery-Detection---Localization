"""⛔ P10-7 / P15-3: interval AP@IoU and AR@N against hand-computed cases.

Every expected value below is worked out by hand in the comment beside it, from the
definitions in `src/evaluation/localization.py` -- not by running the implementation and
pasting its output. Section 6.7's three mandatory cases come first.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.localization import (
    average_precision_at_iou,
    average_recall_at_n,
    boundary_errors,
    false_positive_rate_on_real,
    interval_iou,
    localization_report,
)

GT1 = {"a": [(1.0, 2.0)]}


def ap(preds, gt, t=0.5, conv="standard"):
    return average_precision_at_iou(preds, gt, t, convention=conv)


# ---------------------------------------------------------------- IoU


def test_iou_hand_values():
    assert interval_iou((0, 1), (0, 1)) == 1.0
    assert interval_iou((0, 1), (1, 2)) == 0.0  # touching, no overlap
    assert interval_iou((0, 2), (1, 3)) == pytest.approx(1 / 3)  # 1 / 3
    assert interval_iou((0, 4), (1, 2)) == pytest.approx(0.25)  # containment


# ---------------------------------------------------------------- section 6.7's three cases


def test_one_perfect_prediction_is_one():
    assert ap({"a": [(1.0, 2.0, 0.9)]}, GT1) == 1.0


def test_iou_0_4_at_threshold_0_5_is_zero():
    # [1.0, 1.4] vs [1.0, 2.0]: overlap 0.4, union 1.0 -> IoU 0.4 < 0.5 -> FP, AP 0.
    assert interval_iou((1.0, 1.4), (1.0, 2.0)) == pytest.approx(0.4)
    assert ap({"a": [(1.0, 1.4, 0.9)]}, GT1) == 0.0


def test_one_correct_one_spurious_ranked_correct_first():
    # ranks: TP (0.9), FP (0.8). P = [1, 1/2], R = [1, 1].
    # Recall steps: +1 at rank 1 with interpolated precision 1 -> AP = 1.0.
    assert ap({"a": [(1.0, 2.0, 0.9), (5.0, 6.0, 0.8)]}, GT1) == pytest.approx(1.0)


def test_one_correct_one_spurious_ranked_spurious_first():
    # ranks: FP (0.9), TP (0.8). P = [0, 1/2], R = [0, 1].
    # Recall step +1 at rank 2, precision 1/2 -> AP = 0.5.
    assert ap({"a": [(5.0, 6.0, 0.9), (1.0, 2.0, 0.8)]}, GT1) == pytest.approx(0.5)


# ---------------------------------------------------------------- matching rules


def test_interpolation_uses_the_envelope():
    # 2 GT. ranks: TP .9, FP .8, TP .7. P = [1, 1/2, 2/3], R = [.5, .5, 1].
    # Envelope (max to the right): [1, 2/3, 2/3]. Steps +.5 @ rank1 (1), +.5 @ rank3 (2/3).
    # AP = .5 * 1 + .5 * 2/3 = 5/6.
    gt = {"a": [(1.0, 2.0), (4.0, 5.0)]}
    preds = {"a": [(1.0, 2.0, 0.9), (7.0, 8.0, 0.8), (4.0, 5.0, 0.7)]}
    assert ap(preds, gt) == pytest.approx(5 / 6)


def test_duplicate_detection_is_a_false_positive():
    # Both overlap the single GT; only the higher-confidence one counts, the other is FP.
    # ranks: TP .9, FP .8 -> AP 1.0 (the FP comes after all recall is reached) ...
    preds = {"a": [(1.0, 2.0, 0.9), (1.05, 2.0, 0.8)]}
    assert ap(preds, GT1) == pytest.approx(1.0)
    # ... but at AR it cannot count twice: recall is capped at 1.
    assert average_recall_at_n(preds, GT1, 10, [0.5]) == 1.0


def test_duplicate_ranked_above_the_better_match_takes_the_span():
    # The .9 prediction (IoU .5) claims the span; the exact .8 one is then a duplicate FP.
    # At t=0.5: TP, FP -> AP 1.0. At t=0.75: the .9 one fails, the .8 one matches:
    # ranks FP, TP -> P = [0, 1/2], AP = 0.5.
    preds = {"a": [(1.0, 1.5, 0.9), (1.0, 2.0, 0.8)]}
    assert ap(preds, GT1, 0.5) == pytest.approx(1.0)
    assert ap(preds, GT1, 0.75) == pytest.approx(0.5)


def test_best_iou_match_among_unmatched_spans():
    # One prediction overlapping two GT spans must take the one with higher IoU,
    # leaving the other free for the next prediction.
    gt = {"a": [(0.0, 1.0), (1.0, 3.0)]}
    # p1 [0.9, 3.0]: IoU with span0 = 0.1/3.0, with span1 = 2/2.1 -> takes span1.
    # p2 [0.0, 1.0]: exact on span0 -> TP. Both TP -> AP 1.0.
    preds = {"a": [(0.9, 3.0, 0.9), (0.0, 1.0, 0.8)]}
    assert ap(preds, gt) == pytest.approx(1.0)


def test_iou_exactly_at_threshold_counts_under_standard_not_lavdf():
    # [1, 2] vs [1, 3]: IoU = 1/2 exactly. standard uses >=, lavdf uses > (measured).
    gt = {"a": [(1.0, 3.0)]}
    preds = {"a": [(1.0, 2.0, 0.9), (5.0, 6.0, 0.1)]}
    assert ap(preds, gt, 0.5) == 1.0
    assert ap(preds, gt, 0.5, "lavdf") == 0.0


def test_false_positives_on_real_videos_cost_precision():
    # A real video's segment outranks the true detection: ranks FP, TP -> AP 0.5.
    gt = {"a": [(1.0, 2.0)], "real": []}
    preds = {"a": [(1.0, 2.0, 0.8)], "real": [(3.0, 4.0, 0.95)]}
    assert ap(preds, gt) == pytest.approx(0.5)


def test_matches_never_cross_videos():
    # Identical interval, wrong video -> FP, and the GT in "a" is never recalled.
    gt = {"a": [(1.0, 2.0)], "b": []}
    assert ap({"b": [(1.0, 2.0, 0.9)]}, gt) == 0.0


def test_edge_cases():
    assert ap({}, GT1) == 0.0  # nothing predicted, one span to find
    assert ap({"a": []}, GT1) == 0.0
    assert math.isnan(ap({"r": [(1.0, 2.0, 0.9)]}, {"r": []}))  # no GT anywhere
    with pytest.raises(KeyError):
        ap({"unknown": [(1.0, 2.0, 0.5)]}, GT1)  # a real video must be listed explicitly
    with pytest.raises(ValueError):
        ap({"a": [(2.0, 1.0, 0.5)]}, GT1)  # reversed
    with pytest.raises(ValueError):
        ap({"a": [(1.0, 2.0, 0.5)]}, GT1, conv="coco")


# ---------------------------------------------------------------- the lavdf convention


def test_lavdf_convention_drops_the_first_recall_step():
    # Measured on the authors' evaluator (scripts/33_crosscheck_ap.py): one perfect
    # prediction scores 0.0, TP-then-FP 0.0, FP-then-TP 0.5.
    assert ap({"a": [(1.0, 2.0, 0.9)]}, GT1, conv="lavdf") == 0.0
    assert ap({"a": [(1.0, 2.0, 0.9), (5.0, 6.0, 0.8)]}, GT1, conv="lavdf") == 0.0
    assert ap({"a": [(5.0, 6.0, 0.9), (1.0, 2.0, 0.8)]}, GT1, conv="lavdf") == pytest.approx(0.5)


# ---------------------------------------------------------------- AR@N


def test_average_recall_hand_values():
    gt = {"a": [(1.0, 2.0), (4.0, 5.0)]}
    # top-1 is the exact match on span 0 -> recall 1/2 at every threshold.
    preds = {"a": [(1.0, 2.0, 0.9), (4.0, 5.0, 0.8)]}
    assert average_recall_at_n(preds, gt, 1, [0.5, 0.9]) == pytest.approx(0.5)
    assert average_recall_at_n(preds, gt, 2, [0.5, 0.9]) == pytest.approx(1.0)
    # IoU 0.6 on a single span: recalled at t=0.5 and 0.6, not at 0.7 or 0.9 -> 2/4.
    near = {"a": [(1.0, 1.6, 0.9)]}
    assert average_recall_at_n(near, GT1, 10, [0.5, 0.6, 0.7, 0.9]) == pytest.approx(0.5)


def test_average_recall_is_proposal_recall_not_matching():
    # One wide proposal can recall two spans at a low threshold.
    gt = {"a": [(0.0, 1.0), (1.0, 2.0)]}
    wide = {"a": [(0.0, 2.0, 0.9)]}
    assert average_recall_at_n(wide, gt, 1, [0.5]) == pytest.approx(1.0)  # IoU .5 each


# ---------------------------------------------------------------- boundary error / FP rate


def test_boundary_error_in_milliseconds():
    # start off by 40 ms, end off by 80 ms -> mean 60 ms.
    gt = {"a": [(1.0, 2.08)], "b": [(3.0, 4.0)]}
    preds = {"a": [(1.04, 2.0, 0.9)], "b": []}
    out = boundary_errors(preds, gt)
    assert out["start_error_ms"] == pytest.approx(40.0)
    assert out["end_error_ms"] == pytest.approx(80.0)
    assert out["boundary_error_ms"] == pytest.approx(60.0)
    assert out["n_matched"] == 1 and out["n_missed"] == 1


def test_false_positive_rate_on_real():
    gt = {"r1": [], "r2": [], "f": [(1.0, 2.0)]}
    preds = {"r1": [(0.0, 1.0, 0.7), (2.0, 3.0, 0.6)], "r2": [], "f": [(1.0, 2.0, 0.9)]}
    out = false_positive_rate_on_real(preds, gt)
    assert out["fp_video_rate"] == 0.5
    assert out["n_real"] == 2 and out["n_real_flagged"] == 1
    assert out["fp_segments_per_real_video"] == 1.0


def test_report_carries_both_conventions():
    rep = localization_report({"a": [(1.0, 2.0, 0.9)]}, GT1)
    assert rep["ap@0.5"] == 1.0 and rep["ap@0.5_lavdf"] == 0.0
    assert rep["ar@10"] == 1.0
    assert rep["n_gt_segments"] == 1 and rep["segment_count_error"] == 0.0
