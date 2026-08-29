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
    """Area under the precision-recall curve (step interpolation)."""
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
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
