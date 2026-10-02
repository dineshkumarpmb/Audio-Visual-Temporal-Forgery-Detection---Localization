"""Interval AP@IoU and AR@N for temporal forgery localization (P10-7, section 6.6-6.7).

    predictions : {video_id: [(start_s, end_s, confidence), ...]}
    ground_truth: {video_id: [(start_s, end_s), ...]}          # [] for a real video

**Section 6.7 is blunt about why this module exists: a wrong AP implementation invalidates
every localization number while looking entirely plausible.** So it is a pure function of
two dictionaries -- no model, no loader, no torch -- tested against hand-computed cases in
`tests/unit/test_localization_metrics.py` and cross-checked against the LAV-DF authors' own
evaluator (`scripts/33_crosscheck_ap.py`).

Two conventions, because the published one is not the textbook one
----------------------------------------------------------------------
``convention="standard"`` (the default, and every headline in this project):

* All predictions across the split are ranked by confidence, highest first (stable sort,
  so ties resolve by input order -- see the note below).
* Each prediction matches the **unmatched** ground-truth span in its own video with the
  highest IoU, if that IoU is ``>= threshold``. A second prediction on an already-matched
  span is a false positive -- section 6.7's "only the highest-confidence counts".
* AP is the area under the **interpolated** precision-recall curve (precision replaced by
  its running maximum from the right), summed over every recall step *including the
  first*. One perfect prediction scores exactly 1.0.

``convention="lavdf"`` reproduces the evaluator shipped by the dataset's authors
(``avdeepfake1m.evaluation.ap_1d``), measured rather than read -- the package is a compiled
Rust extension. Two differences, each confirmed against it on random cases
(`scripts/33_crosscheck_ap.py`, `reports/ap_crosscheck.json`):

1. **The first recall step is dropped.** The sum runs over ``recall[1:] - recall[:-1]``, so
   one perfect prediction scores **0.0**, and at dev scale every AP is low by up to
   ``precision[0] / n_ground_truth``.
2. **Matching is by list order, not by confidence.** Each ground-truth span is claimed by
   the *first* prediction in that video's list whose IoU is **strictly** above the
   threshold. Predictions are sorted by confidence within each video first, which is the
   order the authors' submission format expects.

Real videos contribute no ground truth. Every segment predicted on one is a false positive,
so AP already punishes false accusations -- the separate FP-segment rate (section 6.8)
states the same failure per video, where a reader can see it.

⚠️ **Tied confidences** are ranked by input order, as in every published implementation.
This matters much less than it did for frame AP (PF-23): a segment's confidence is a mean
over its frames, so exact ties between segments are rare, and a constant scorer cannot
produce segments at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

AP_IOU_THRESHOLDS: tuple[float, ...] = (0.5, 0.75, 0.95)
AR_N_PROPOSALS: tuple[int, ...] = (100, 50, 20, 10)
# Section 6.6 names AR@N without its IoU set; LAV-DF's benchmark averages recall over
# 0.5:0.05:0.95, and matching it keeps the numbers comparable to the literature.
AR_IOU_THRESHOLDS: tuple[float, ...] = tuple(np.round(np.arange(0.5, 0.951, 0.05), 2).tolist())

Predictions = Mapping[str, Sequence[Sequence[float]]]
GroundTruth = Mapping[str, Sequence[Sequence[float]]]


def interval_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two 1-D intervals ``[start, end]``. Disjoint or touching intervals score 0."""
    a0, a1 = float(a[0]), float(a[1])
    b0, b1 = float(b[0]), float(b[1])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """``(P, G)`` IoU between every prediction and every ground-truth span."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 2)
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)))
    inter = np.clip(
        np.minimum(pred[:, None, 1], gt[None, :, 1]) - np.maximum(pred[:, None, 0], gt[None, :, 0]),
        0.0,
        None,
    )
    union = np.maximum(pred[:, None, 1], gt[None, :, 1]) - np.minimum(
        pred[:, None, 0], gt[None, :, 0]
    )
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def _validate(predictions: Predictions, ground_truth: GroundTruth) -> None:
    unknown = set(predictions) - set(ground_truth)
    if unknown:
        raise KeyError(
            f"{len(unknown)} predicted video(s) have no ground-truth entry, e.g. "
            f"{sorted(unknown)[:3]}. A real video needs an explicit empty list, otherwise its "
            "false positives would silently vanish from AP."
        )
    for vid, segs in predictions.items():
        for seg in segs:
            if len(seg) != 3:
                raise ValueError(f"{vid}: prediction {seg!r} is not (start, end, confidence)")
            if not seg[1] > seg[0]:
                raise ValueError(f"{vid}: prediction {seg!r} is empty or reversed")
    for vid, segs in ground_truth.items():
        for seg in segs:
            if len(seg) != 2 or not seg[1] > seg[0]:
                raise ValueError(f"{vid}: ground-truth span {seg!r} is not a valid (start, end)")


def _hits_standard(
    predictions: Predictions, ground_truth: GroundTruth, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, is_tp) per prediction, greedy by global confidence."""
    rows = [
        (float(seg[2]), vid, float(seg[0]), float(seg[1]))
        for vid, segs in predictions.items()
        for seg in segs
    ]
    order = sorted(range(len(rows)), key=lambda i: -rows[i][0])  # stable
    used = {vid: np.zeros(len(g), dtype=bool) for vid, g in ground_truth.items()}
    gt = {vid: np.asarray(g, dtype=np.float64).reshape(-1, 2) for vid, g in ground_truth.items()}

    conf = np.empty(len(rows))
    tp = np.zeros(len(rows), dtype=bool)
    for rank, i in enumerate(order):
        c, vid, s, e = rows[i]
        conf[rank] = c
        g = gt[vid]
        if not len(g):
            continue
        ious = iou_matrix(np.array([[s, e]]), g)[0]
        ious[used[vid]] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= threshold:
            used[vid][j] = True
            tp[rank] = True
    return conf, tp


def _hits_lavdf(
    predictions: Predictions, ground_truth: GroundTruth, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """The authors' matching: per ground-truth span, the first prediction with IoU > thr."""
    confs, tps = [], []
    for vid, segs in predictions.items():
        if not len(segs):
            continue
        arr = np.asarray(segs, dtype=np.float64).reshape(-1, 3)
        arr = arr[np.argsort(-arr[:, 2], kind="stable")]
        is_tp = np.zeros(len(arr), dtype=bool)
        g = np.asarray(ground_truth[vid], dtype=np.float64).reshape(-1, 2)
        if len(g):
            ious = iou_matrix(arr[:, :2], g)
            taken: set[int] = set()
            for j in range(len(g)):
                for i in np.flatnonzero(ious[:, j] > threshold):
                    if int(i) not in taken:
                        taken.add(int(i))
                        break
            is_tp[list(taken)] = True
        confs.append(arr[:, 2])
        tps.append(is_tp)
    if not confs:
        return np.empty(0), np.empty(0, dtype=bool)
    conf, tp = np.concatenate(confs), np.concatenate(tps)
    order = np.argsort(-conf, kind="stable")
    return conf[order], tp[order]


def _ap_from_hits(tp: np.ndarray, n_gt: int, *, drop_first: bool) -> float:
    if n_gt == 0:
        return float("nan")
    if not len(tp):
        return 0.0
    ctp = np.cumsum(tp)
    precision = ctp / np.arange(1, len(tp) + 1)
    recall = ctp / n_gt
    # Interpolate: each precision becomes the best precision at any equal-or-higher recall.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    if drop_first:
        return float(np.sum((recall[1:] - recall[:-1]) * envelope[1:]))
    steps = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(steps * envelope))


def average_precision_at_iou(
    predictions: Predictions,
    ground_truth: GroundTruth,
    threshold: float,
    *,
    convention: str = "standard",
) -> float:
    """⛔ P10-7: interval AP at one IoU threshold. NaN when there is no ground truth at all."""
    if convention not in ("standard", "lavdf"):
        raise ValueError(f"unknown convention {convention!r}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"IoU threshold must be in (0, 1], got {threshold}")
    _validate(predictions, ground_truth)
    n_gt = sum(len(g) for g in ground_truth.values())
    if convention == "standard":
        _, tp = _hits_standard(predictions, ground_truth, threshold)
    else:
        _, tp = _hits_lavdf(predictions, ground_truth, threshold)
    return _ap_from_hits(tp, n_gt, drop_first=convention == "lavdf")


def average_recall_at_n(
    predictions: Predictions,
    ground_truth: GroundTruth,
    n_proposals: int,
    iou_thresholds: Iterable[float] = AR_IOU_THRESHOLDS,
    *,
    convention: str = "standard",
) -> float:
    """AR@N: recall using each video's top-N proposals, averaged over IoU thresholds.

    A ground-truth span counts as recalled at threshold ``t`` if **any** of its video's top-N
    proposals overlaps it by ``>= t`` (``> t`` under ``lavdf``). Proposals are not consumed
    -- this is proposal recall, not detection matching, and it is what AR@N means in the
    temporal-localization literature.

    ⚠️ Section 6.4's chain emits a handful of segments per video, not 100, so AR@100, @50
    and @20 are expected to coincide. That is a property of the post-processing, not a bug
    here -- it is stated beside the numbers when they are reported.
    """
    if convention not in ("standard", "lavdf"):
        raise ValueError(f"unknown convention {convention!r}")
    _validate(predictions, ground_truth)
    thresholds = np.asarray(list(iou_thresholds), dtype=np.float64)
    recalled = np.zeros(len(thresholds))
    n_gt = 0
    for vid, g in ground_truth.items():
        g = np.asarray(g, dtype=np.float64).reshape(-1, 2)
        if not len(g):
            continue
        n_gt += len(g)
        segs = predictions.get(vid, [])
        if not len(segs):
            continue
        arr = np.asarray(segs, dtype=np.float64).reshape(-1, 3)
        top = arr[np.argsort(-arr[:, 2], kind="stable")][:n_proposals]
        best = iou_matrix(top[:, :2], g).max(axis=0)  # (G,)
        if convention == "lavdf":
            recalled += (best[None, :] > thresholds[:, None]).sum(axis=1)
        else:
            recalled += (best[None, :] >= thresholds[:, None]).sum(axis=1)
    if n_gt == 0:
        return float("nan")
    return float((recalled / n_gt).mean())


def boundary_errors(predictions: Predictions, ground_truth: GroundTruth) -> dict[str, float]:
    """Section 6.6's interpretable metric: how far off are the boundaries, in milliseconds?

    Each ground-truth span is paired with the predicted segment in its video that overlaps it
    most (IoU > 0); spans with no overlapping prediction are counted as missed rather than
    given an arbitrary error, so the error describes only what was found and ``n_missed``
    says how much was not.
    """
    _validate(predictions, ground_truth)
    start_err, end_err, missed = [], [], 0
    for vid, g in ground_truth.items():
        g = np.asarray(g, dtype=np.float64).reshape(-1, 2)
        if not len(g):
            continue
        segs = predictions.get(vid, [])
        if not len(segs):
            missed += len(g)
            continue
        arr = np.asarray(segs, dtype=np.float64).reshape(-1, 3)
        ious = iou_matrix(arr[:, :2], g)
        for j in range(len(g)):
            i = int(np.argmax(ious[:, j]))
            if ious[i, j] <= 0:
                missed += 1
                continue
            start_err.append(abs(arr[i, 0] - g[j, 0]))
            end_err.append(abs(arr[i, 1] - g[j, 1]))
    both = start_err + end_err
    return {
        "start_error_ms": float(np.mean(start_err) * 1000) if start_err else float("nan"),
        "end_error_ms": float(np.mean(end_err) * 1000) if end_err else float("nan"),
        "boundary_error_ms": float(np.mean(both) * 1000) if both else float("nan"),
        "boundary_error_median_ms": float(np.median(both) * 1000) if both else float("nan"),
        "n_matched": len(start_err),
        "n_missed": missed,
    }


def false_positive_rate_on_real(predictions: Predictions, ground_truth: GroundTruth) -> dict:
    """Section 6.8: the fraction of genuine videos on which the system flags any span.

    For a tool a journalist might use this is arguably the metric that matters most --
    falsely accusing a real video is worse than missing a fake one.
    """
    _validate(predictions, ground_truth)
    real = [vid for vid, g in ground_truth.items() if not len(g)]
    flagged = [vid for vid in real if len(predictions.get(vid, []))]
    return {
        "fp_video_rate": len(flagged) / len(real) if real else float("nan"),
        "n_real": len(real),
        "n_real_flagged": len(flagged),
        "fp_segments_per_real_video": (
            sum(len(predictions.get(v, [])) for v in real) / len(real) if real else float("nan")
        ),
    }


def localization_report(
    predictions: Predictions,
    ground_truth: GroundTruth,
    *,
    ap_thresholds: Iterable[float] = AP_IOU_THRESHOLDS,
    ar_n: Iterable[int] = AR_N_PROPOSALS,
    ar_thresholds: Iterable[float] = AR_IOU_THRESHOLDS,
) -> dict:
    """Every section 6.6 number in one dict, under both conventions for AP and AR."""
    ap_thresholds, ar_n, ar_thresholds = list(ap_thresholds), list(ar_n), list(ar_thresholds)
    out: dict = {}
    for conv, suffix in (("standard", ""), ("lavdf", "_lavdf")):
        for t in ap_thresholds:
            out[f"ap@{t:g}{suffix}"] = average_precision_at_iou(
                predictions, ground_truth, t, convention=conv
            )
        for n in ar_n:
            out[f"ar@{n}{suffix}"] = average_recall_at_n(
                predictions, ground_truth, n, ar_thresholds, convention=conv
            )
    out.update(boundary_errors(predictions, ground_truth))
    out.update(false_positive_rate_on_real(predictions, ground_truth))
    n_pred = [len(predictions.get(v, [])) for v, g in ground_truth.items() if len(g)]
    n_true = [len(g) for g in ground_truth.values() if len(g)]
    out["segment_count_error"] = (
        float(np.mean(np.abs(np.array(n_pred) - np.array(n_true)))) if n_true else float("nan")
    )
    out["n_videos"] = len(ground_truth)
    out["n_gt_segments"] = int(sum(n_true))
    out["n_pred_segments"] = int(sum(len(v) for v in predictions.values()))
    return out
