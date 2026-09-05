"""Classification metrics for Phase 4 (section 8.1).

AUC is the headline because it is threshold-free: Phase 4's job is to show the features
carry signal, and a threshold chosen at this stage would only have to be re-chosen on dev
later (section 3.5 RULE 4). Accuracy is reported alongside but is **not** the number to
judge on — LAV-DF is 73.3% fake at video level, so a model that always says "fake" scores
73.3% accuracy and 0.5 AUC.

Everything here is computed with numpy from raw scores so it can be unit-tested against
hand-worked examples, rather than trusting a library call whose conventions (which class
is positive, how ties break) are easy to get subtly wrong.
"""

from __future__ import annotations

import numpy as np


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(np.int64)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: {scores.shape} scores vs {labels.shape} labels")
    if scores.size == 0:
        raise ValueError("no samples")
    bad = set(np.unique(labels)) - {0, 1}
    if bad:
        raise ValueError(f"labels must be 0/1, found {sorted(bad)}")
    return scores, labels


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    Uses average ranks so tied scores are handled correctly — a model that outputs the
    same value for everything must score exactly 0.5, not 1.0. Returns NaN when only one
    class is present, because AUC is undefined there and returning 0.5 would silently
    look like a legitimately uninformative model.
    """
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation).

    **Tied scores are resolved as a group, not by input order** — the same discipline
    `roc_auc` applies via average ranks. Ranking tied items arbitrarily lets a model that
    emits one constant value score anywhere from 0 to 1 depending only on how the rows
    happened to be sorted; resolving the group to the precision at its end gives it exactly
    the positive rate, which is the correct AP for a scorer that has ranked nothing.

    That mattered little at clip level, where sigmoid outputs are never exactly equal —
    but it made the `majority_class` floor (a constant by construction) report 0.7383
    instead of its true 0.7200 on the Phase 4-6 dev split. At **frame** level it matters a
    great deal: a saturated frame head is the specific degenerate mode `frame_pos_weight`
    defends against, and under order-dependent ties it would score near-perfect AP.
    """
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    ranked, hits = scores[order], labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)

    # Every position takes the precision at the end of its tie group, so all items sharing
    # a score are credited identically regardless of where sorting happened to place them.
    group_end = np.empty(len(ranked), dtype=bool)
    group_end[-1] = True
    group_end[:-1] = ranked[:-1] != ranked[1:]
    ends = np.flatnonzero(group_end)
    precision = precision[ends][np.searchsorted(ends, np.arange(len(ranked)))]

    return float((precision * hits).sum() / n_pos)


def accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    scores, labels = _check(scores, labels)
    return float(((scores >= threshold).astype(np.int64) == labels).mean())


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """The rate where false accepts equal false rejects. Threshold-free, like AUC."""
    scores, labels = _check(scores, labels)
    if labels.sum() in (0, labels.size):
        return float("nan")

    thresholds = np.unique(scores)
    best, best_gap = 0.5, np.inf
    for t in thresholds:
        pred = scores >= t
        far = float((pred & (labels == 0)).sum() / max((labels == 0).sum(), 1))
        frr = float(((~pred) & (labels == 1)).sum() / max((labels == 1).sum(), 1))
        gap = abs(far - frr)
        if gap < best_gap:
            best, best_gap = 0.5 * (far + frr), gap
    return float(best)


def summary(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    scores, labels = _check(scores, labels)
    pred = (scores >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auc": roc_auc(scores, labels),
        "ap": average_precision(scores, labels),
        "accuracy": accuracy(scores, labels, threshold),
        "eer": equal_error_rate(scores, labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "n": int(labels.size),
        "positive_rate": float(labels.mean()),
    }


def frame_metrics(
    scores: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """P7-4: frame-level AP — the first localization-relevant metric in the project.

    Every valid frame in the split is one sample, pooled across clips: `(B, T)` scores
    against `(B, T)` targets from `src.localization.targets`, keeping only positions where
    `mask` is True. Padded positions are dropped rather than scored, because a padded frame
    is trivially "real" and including them would inflate AP by adding free true negatives
    in proportion to how ragged the batch happened to be.

    **AP rather than AUC is the headline here, and the imbalance is why.** The median
    forged span is ~16 frames against clips averaging ~200 (P1-11), so roughly 5-10% of
    frames are positive — an ROC curve is dominated by the vast negative majority and looks
    good for a model that has found nothing. AP is computed against the positive class
    only. AUC is reported alongside for continuity with the clip-level tables, not to be
    judged on.

    ⚠️ This is *frame* AP, not the *interval* AP@IoU that Phase 10's gate requires (P10-7).
    They are different quantities: this one never forms a segment, so it cannot punish a
    prediction that is right about which frames are forged and wrong about where the
    boundaries fall. It is the cheap tracked metric Phase 7 asks for, not the headline
    localization number.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    if scores.shape != targets.shape:
        raise ValueError(f"shape mismatch: {scores.shape} scores vs {targets.shape} targets")
    if mask is not None:
        keep = np.asarray(mask).ravel().astype(bool)
        if keep.shape != scores.shape:
            raise ValueError(f"mask shape {keep.shape} != scores {scores.shape}")
        scores, targets = scores[keep], targets[keep]

    labels = (targets > 0.5).astype(np.int64)
    if labels.size == 0:
        return {
            "frame_ap": float("nan"),
            "frame_auc": float("nan"),
            "n_frames": 0,
            "frame_positive_rate": float("nan"),
        }

    n_pos = int(labels.sum())
    return {
        "frame_ap": average_precision(scores, labels) if n_pos else float("nan"),
        "frame_auc": roc_auc(scores, labels),
        "n_frames": int(labels.size),
        "frame_positive_rate": float(labels.mean()),
    }


def per_class_breakdown(
    scores: np.ndarray, labels: np.ndarray, class_names: list[str], threshold: float = 0.5
) -> dict[str, dict]:
    """Section 8.2.4's collapse detector.

    Each fake class is scored against **all** the real videos, so the numbers are
    comparable across classes. This is the diagnostic that catches a model ignoring one
    modality: if `visual_only` sits near chance while `audio_only` is near-perfect, no
    aggregate figure will tell you, and the aggregate will look fine.
    """
    scores, labels = _check(scores, labels)
    names = np.asarray(class_names)
    real = labels == 0

    out: dict[str, dict] = {}
    for cls in ("visual_only", "audio_only", "both"):
        sel = (names == cls) & (labels == 1)
        if not sel.any():
            continue
        keep = sel | real
        out[cls] = {
            "n_fake": int(sel.sum()),
            "auc": roc_auc(scores[keep], labels[keep]),
            "recall": float((scores[sel] >= threshold).mean()),
        }
    if real.any():
        out["real"] = {
            "n": int(real.sum()),
            "specificity": float((scores[real] < threshold).mean()),
        }
    return out
