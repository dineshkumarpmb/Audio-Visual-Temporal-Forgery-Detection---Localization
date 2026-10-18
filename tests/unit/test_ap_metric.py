"""⛔ P15-3: the AP metrics, checked against implementations that are not this project's.

Section 6.7: "a wrong AP implementation invalidates every localization number while looking
entirely plausible". `test_localization_metrics.py` pins hand-computed cases and
`scripts/33_crosscheck_ap.py` matched the LAV-DF convention to the dataset authors' evaluator
-- but that needs an external wheel and never runs in CI. This file adds checks that do:

* **Frame / clip AP and ROC-AUC = scikit-learn**, on random inputs including heavy ties --
  the case PF-23 found broken (ties were resolved by row order) and fixed.
* **Interval AP@IoU (standard convention) = a from-the-definition reference** written here in
  plain Python: rank every prediction by confidence, match each to the best unmatched
  same-video span at IoU >= t, then sum recall steps times the interpolated precision. No
  numpy, no shared helpers, so a bug in `src/evaluation/localization.py` cannot be copied in.
* **Invariants any AP must satisfy**, as property tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.localization import average_precision_at_iou, interval_iou
from src.evaluation.metrics import average_precision, roc_auc

# ---------------------------------------------------------------- frame / clip level


binary_problems = st.integers(2, 300).flatmap(
    lambda n: st.tuples(
        # Scores on a coarse grid make ties common, which is where implementations differ.
        st.lists(st.integers(0, 10).map(lambda k: k / 10), min_size=n, max_size=n),
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
    )
)


class TestAgainstScikitLearn:
    @settings(max_examples=400, deadline=None)
    @given(binary_problems)
    def test_average_precision_matches_sklearn(self, problem):
        scores, labels = (np.asarray(x, dtype=np.float64) for x in problem)
        assume(labels.sum() > 0)
        assert average_precision(scores, labels) == pytest.approx(
            average_precision_score(labels, scores), abs=1e-12
        )

    @settings(max_examples=400, deadline=None)
    @given(binary_problems)
    def test_roc_auc_matches_sklearn(self, problem):
        scores, labels = (np.asarray(x, dtype=np.float64) for x in problem)
        assume(0 < labels.sum() < len(labels))
        assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)

    def test_constant_scorer_ap_is_the_positive_rate(self):
        """PF-23's case: a constant scorer is one tie group; AP = fraction of positives."""
        labels = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        assert average_precision(np.full(10, 0.7), labels) == pytest.approx(0.3)
        assert average_precision_score(labels, np.full(10, 0.7)) == pytest.approx(0.3)

    def test_frame_level_on_realistic_scale(self):
        """A dev-sized frame problem (~80k frames, ~7% positive), continuous scores."""
        rng = np.random.default_rng(0)
        labels = (rng.random(80_000) < 0.07).astype(np.int64)
        scores = np.clip(0.3 * labels + rng.normal(0.35, 0.2, labels.size), 0, 1)
        assert average_precision(scores, labels) == pytest.approx(
            average_precision_score(labels, scores), abs=1e-10
        )
        assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-10)


# ---------------------------------------------------------------- interval AP@IoU


def ref_iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def reference_ap_at_iou(predictions: dict, ground_truth: dict, t: float) -> float:
    """Textbook interpolated AP, written from the definition with no shared code."""
    n_gt = sum(len(v) for v in ground_truth.values())
    if n_gt == 0:
        return float("nan")
    flat = [(seg[2], i, vid, seg[0], seg[1]) for vid, segs in predictions.items()
            for i, seg in enumerate(segs)]  # fmt: skip
    # Highest confidence first; equal confidences keep input order (stable sort).
    flat = sorted(enumerate(flat), key=lambda p: (-p[1][0], p[0]))
    taken = {vid: [False] * len(v) for vid, v in ground_truth.items()}
    hits = []
    for _, (_, _, vid, s, e) in flat:
        best, best_j = -1.0, -1
        for j, g in enumerate(ground_truth[vid]):
            if taken[vid][j]:
                continue
            iou = ref_iou((s, e), g)
            if iou > best:
                best, best_j = iou, j
        if best_j >= 0 and best >= t:
            taken[vid][best_j] = True
            hits.append(True)
        else:
            hits.append(False)
    precisions, recalls, tp = [], [], 0
    for k, hit in enumerate(hits, start=1):
        tp += hit
        precisions.append(tp / k)
        recalls.append(tp / n_gt)
    ap, prev_r = 0.0, 0.0
    for k in range(len(hits)):
        if recalls[k] > prev_r:
            ap += (recalls[k] - prev_r) * max(precisions[k:])
            prev_r = recalls[k]
    return ap


def localization_problems():
    span = st.tuples(st.integers(0, 90), st.integers(1, 30)).map(
        lambda p: (p[0] / 10, (p[0] + p[1]) / 10)
    )
    pred = st.tuples(span, st.integers(1, 99).map(lambda c: c / 100)).map(
        lambda p: (p[0][0], p[0][1], p[1])
    )
    video = st.tuples(st.lists(span, max_size=2), st.lists(pred, max_size=4))
    return st.lists(video, min_size=1, max_size=6)


class TestIntervalAP:
    @settings(max_examples=400, deadline=None)
    @given(videos=localization_problems(), t=st.sampled_from([0.1, 0.3, 0.5, 0.75, 0.95]))
    def test_matches_the_reference(self, videos, t):
        gt = {f"v{i}": [tuple(g) for g in v[0]] for i, v in enumerate(videos)}
        preds = {f"v{i}": [tuple(p) for p in v[1]] for i, v in enumerate(videos)}
        ref = reference_ap_at_iou(preds, gt, t)
        got = average_precision_at_iou(preds, gt, t)
        if math.isnan(ref):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(ref, abs=1e-12)

    def test_hand_case_two_videos(self):
        """Ranked: v1 hit (P=1, R=1/3), v2 miss, v2 hit (P=2/3, R=2/3), v1 dup -> AP = 5/9."""
        gt = {"v1": [(0.0, 1.0)], "v2": [(2.0, 3.0), (5.0, 6.0)]}
        preds = {
            "v1": [(0.0, 1.0, 0.9), (0.1, 1.0, 0.6)],
            "v2": [(8.0, 9.0, 0.8), (2.0, 3.1, 0.7)],
        }
        assert average_precision_at_iou(preds, gt, 0.5) == pytest.approx(1 / 3 + (1 / 3) * (2 / 3))
        assert reference_ap_at_iou(preds, gt, 0.5) == pytest.approx(5 / 9)


# ---------------------------------------------------------------- invariants


class TestInvariants:
    @settings(max_examples=200, deadline=None)
    @given(videos=localization_problems(), t=st.sampled_from([0.5, 0.75]))
    def test_ap_is_a_probability_and_order_free(self, videos, t):
        gt = {f"v{i}": v[0] for i, v in enumerate(videos)}
        preds = {f"v{i}": v[1] for i, v in enumerate(videos)}
        ap = average_precision_at_iou(preds, gt, t)
        if math.isnan(ap):
            assert sum(len(g) for g in gt.values()) == 0
            return
        assert 0.0 <= ap <= 1.0
        # Order-free only when confidences are distinct: exact ties rank by input order, by
        # design (see `test_exact_confidence_ties_rank_by_input_order`).
        confs = [p[2] for segs in preds.values() for p in segs]
        assume(len(set(confs)) == len(confs))
        reversed_videos = dict(reversed(list(preds.items())))
        assert average_precision_at_iou(reversed_videos, gt, t) == pytest.approx(ap)

    def test_exact_confidence_ties_rank_by_input_order(self):
        """Documented, not a bug (module docstring): with two predictions at the *same*
        confidence, the one listed first ranks first -- as in the published evaluators.
        Found by the property test above. Real segment confidences are means over frames,
        so exact ties are rare in practice; a constant scorer produces no segments at all."""
        gt = {"real": [], "fake": [(0.0, 0.1)]}
        miss_first = {"real": [(0.0, 0.1, 0.01)], "fake": [(0.0, 0.1, 0.01)]}
        hit_first = {"fake": [(0.0, 0.1, 0.01)], "real": [(0.0, 0.1, 0.01)]}
        assert average_precision_at_iou(miss_first, gt, 0.5) == pytest.approx(0.5)
        assert average_precision_at_iou(hit_first, gt, 0.5) == pytest.approx(1.0)

    @settings(max_examples=200, deadline=None)
    @given(videos=localization_problems())
    def test_perfect_predictions_score_one(self, videos):
        gt = {f"v{i}": v[0] for i, v in enumerate(videos)}
        assume(sum(len(g) for g in gt.values()) > 0)
        perfect = {vid: [(s, e, 0.9) for s, e in g] for vid, g in gt.items()}
        # Overlapping labelled spans can steal each other's match below IoU 1; require disjoint.
        for g in gt.values():
            assume(all(ref_iou(a, b) == 0 for i, a in enumerate(g) for b in g[i + 1 :]))
        assert average_precision_at_iou(perfect, gt, 0.95) == pytest.approx(1.0)

    @settings(max_examples=200, deadline=None)
    @given(videos=localization_problems(), conf=st.floats(0.01, 0.99))
    def test_a_false_accusation_on_a_real_video_never_helps(self, videos, conf):
        gt = {f"v{i}": v[0] for i, v in enumerate(videos)}
        preds = {f"v{i}": v[1] for i, v in enumerate(videos)}
        assume(sum(len(g) for g in gt.values()) > 0)
        before = average_precision_at_iou(preds, gt, 0.5)
        gt2, preds2 = {**gt, "real": []}, {**preds, "real": [(0.0, 1.0, conf)]}
        assert average_precision_at_iou(preds2, gt2, 0.5) <= before + 1e-12

    @settings(max_examples=300, deadline=None)
    @given(
        a=st.tuples(st.floats(0, 10), st.floats(0.01, 5)).map(lambda p: (p[0], p[0] + p[1])),
        b=st.tuples(st.floats(0, 10), st.floats(0.01, 5)).map(lambda p: (p[0], p[0] + p[1])),
    )
    def test_iou_is_symmetric_bounded_and_one_only_on_identity(self, a, b):
        ab, ba = interval_iou(a, b), interval_iou(b, a)
        assert ab == pytest.approx(ba)
        assert 0.0 <= ab <= 1.0
        assert interval_iou(a, a) == pytest.approx(1.0)
        if ab == pytest.approx(1.0, abs=1e-12):
            assert a == pytest.approx(b, abs=1e-6)

    @settings(max_examples=200, deadline=None)
    @given(binary_problems, st.sampled_from(["square", "shift", "logit"]))
    def test_frame_ap_is_invariant_to_monotone_rescaling(self, problem, how):
        scores, labels = (np.asarray(x, dtype=np.float64) for x in problem)
        assume(labels.sum() > 0)
        f = {
            "square": lambda x: x**2,
            "shift": lambda x: 3 * x + 7,
            "logit": lambda x: np.log((x + 1e-3) / (1 - x + 1e-3)),
        }[how]
        assert average_precision(f(scores), labels) == pytest.approx(
            average_precision(scores, labels), abs=1e-12
        )
