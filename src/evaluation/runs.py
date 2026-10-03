"""P11-1: evaluation as a pure function of saved predictions -- no model, no loader, no GPU.

    predictions .npz + manifest  ->  metrics dict

Training scripts save what a model *said* (`*_predictions.npz`); everything a report claims
is computed here from those files. A metric bug is then fixed by re-running evaluation, never
by retraining, and the dev numbers of Experiment G and the single test run (P11-8) go through
exactly the same code.

The `.npz` layout is the one `scripts/34_train_localization.py` writes: `video_ids`,
`labels`, `clip_scores`, `lengths`, and the per-frame `frame_scores` (plus optional
`boundary_scores`) flattened over clips, with `lengths` giving each clip's slice.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.evaluation.localization import average_precision_at_iou, localization_report
from src.evaluation.metrics import frame_metrics, per_class_breakdown, summary
from src.localization.postprocess import (
    PostProcessConfig,
    assert_segment_contract,
    scores_to_segments,
    smooth,
)
from src.localization.targets import frame_targets, normalise_periods

FPS = 25.0

# ⛔ P10-10's grid. Shared so any arm tuned later is tuned exactly as Experiment G's were.
GRID = {
    "threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "median_kernel": [1, 3, 5, 9],
    "min_duration_s": [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0],
    "merge_gap_s": [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
    "merge_first": [False, True],
}
REFINE = [0, 2, 4]


# ---------------------------------------------------------------- loading


def load_predictions(path: str | Path) -> dict:
    """One saved `.npz` -> per-clip frame (and boundary) scores, keyed by video id."""
    d = np.load(path, allow_pickle=False)
    offsets = np.concatenate(([0], np.cumsum(d["lengths"])))
    ids = [str(v) for v in d["video_ids"]]
    frames = {v: d["frame_scores"][offsets[i] : offsets[i + 1]] for i, v in enumerate(ids)}
    bnd = d["boundary_scores"] if "boundary_scores" in d.files else np.zeros((0, 2))
    bounds = (
        {v: bnd[offsets[i] : offsets[i + 1]] for i, v in enumerate(ids)}
        if len(bnd) == len(d["frame_scores"])
        else None
    )
    clip = d["clip_scores"] if "clip_scores" in d.files else None
    return {
        "frames": frames,
        "bounds": bounds,
        "lengths": dict(zip(ids, d["lengths"].tolist(), strict=True)),
        "labels": dict(zip(ids, d["labels"].tolist(), strict=True)),
        "clip_scores": (
            None if clip is None else dict(zip(ids, clip.astype(float).tolist(), strict=True))
        ),
    }


def ground_truth(manifest, lengths: Mapping[str, int]) -> dict[str, list[tuple[float, float]]]:
    """Label seconds, clipped to what the model saw (`T / fps`). Real videos map to `[]`."""
    gt = {}
    for vid, t in lengths.items():
        end_of_clip = t / FPS
        spans = []
        for s, e in normalise_periods(manifest.loc[vid, "fake_periods"]):
            if s < end_of_clip:
                spans.append((s, min(e, end_of_clip)))
        gt[vid] = spans
    return gt


# ---------------------------------------------------------------- post-processing


def segments(pred: dict, cfg: PostProcessConfig, smoothed: dict | None = None) -> dict:
    bounds = pred["bounds"] if cfg.refine_radius else None
    return {
        vid: scores_to_segments(
            s,
            cfg,
            boundary_scores=None if bounds is None else bounds[vid],
            smoothed=None if smoothed is None else smoothed[vid],
        )
        for vid, s in pred["frames"].items()
    }


def _grid_for_kernel(job: tuple) -> list[dict]:
    """One median kernel's slice of the grid. Top-level so worker processes can import it."""
    kernel, preds, gt, refine = job
    smoothed = [{vid: smooth(s, kernel) for vid, s in p["frames"].items()} for p in preds]
    rows = []
    for tau, mind, gap, mf, rad in itertools.product(
        GRID["threshold"],
        GRID["min_duration_s"],
        GRID["merge_gap_s"],
        GRID["merge_first"],
        refine,
    ):
        # With no min-duration filter or no merging, the two step orders are identical --
        # score the configuration once rather than twice.
        if mf and (mind == 0.0 or gap == 0.0):
            continue
        cfg = PostProcessConfig(
            threshold=tau,
            median_kernel=kernel,
            min_duration_s=mind,
            merge_gap_s=gap,
            merge_first=mf,
            refine_radius=rad,
        )
        aps = [
            average_precision_at_iou(segments(p, cfg, sm), gt, 0.5)
            for p, sm in zip(preds, smoothed, strict=True)
        ]
        rows.append({"cfg": cfg.as_dict(), "ap50": float(np.mean(aps))})
    return rows


def grid_search(
    preds: Sequence[dict], gt: dict, has_boundary: bool, workers: int = 4
) -> tuple[dict, list[dict]]:
    """⛔ P10-10: exhaustive search maximising the mean AP@0.5 over `preds` (one per seed).

    Split by median kernel across processes; the result is identical to a serial sweep
    because the rows are re-assembled in grid order before the (stable) sort.
    """
    refine = REFINE if has_boundary else [0]
    jobs = [(k, list(preds), gt, refine) for k in GRID["median_kernel"]]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            parts = list(pool.map(_grid_for_kernel, jobs))
    else:
        parts = [_grid_for_kernel(j) for j in jobs]
    table = [row for part in parts for row in part]
    table.sort(key=lambda row: -row["ap50"])  # stable: grid order breaks exact ties
    return table[0]["cfg"], table


# ---------------------------------------------------------------- scoring


def localization_metrics(
    pred: dict, gt: dict, cfg: PostProcessConfig, pred_drop_visual: dict | None = None
) -> dict:
    """Every section 6.6 number for one run under one post-processing config."""
    segs = segments(pred, cfg)
    for vid, s in segs.items():  # ⛔ P10-9 on real predictions, not just synthetic ones
        assert_segment_contract(s, pred["lengths"][vid] / FPS)
    rep = localization_report(segs, gt)
    # PF-21 for localization. Refinement is off on *both* sides of this pair: the saved
    # drop-visual predictions carry no boundary channel, and a like-for-like pair matters
    # more than the absolute number.
    if pred_drop_visual is not None:
        plain = PostProcessConfig(**{**cfg.as_dict(), "refine_radius": 0})
        rep["ap@0.5_no_refine"] = average_precision_at_iou(segments(pred, plain), gt, 0.5)
        rep["ap@0.5_drop_visual"] = average_precision_at_iou(
            segments(pred_drop_visual, plain), gt, 0.5
        )
    return rep


def classification_metrics(pred: dict, manifest) -> dict:
    """Table 8.2.1 + 8.2.4 from the saved clip scores, plus frame AP from the saved frames."""
    if pred["clip_scores"] is None:
        raise ValueError("these predictions carry no clip scores")
    ids = list(pred["lengths"])
    scores = np.array([pred["clip_scores"][v] for v in ids])
    labels = np.array([pred["labels"][v] for v in ids])
    out = summary(scores, labels)
    out["per_class"] = per_class_breakdown(
        scores, labels, [str(manifest.loc[v, "class_name"]) for v in ids]
    )
    targets = [
        frame_targets(
            manifest.loc[v, "fake_periods"], pred["lengths"][v], float(manifest.loc[v, "fps"])
        )
        for v in ids
    ]
    out.update(
        frame_metrics(
            np.concatenate([pred["frames"][v] for v in ids]),
            np.concatenate(targets),
        )
    )
    out["confusion"] = confusion(scores, labels)
    return out


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    pred = np.asarray(scores) >= threshold
    labels = np.asarray(labels).astype(bool)
    return {
        "tp": int((pred & labels).sum()),
        "fp": int((pred & ~labels).sum()),
        "tn": int((~pred & ~labels).sum()),
        "fn": int((~pred & labels).sum()),
    }


def _targets(pred: dict, ids, manifest) -> np.ndarray:
    return np.concatenate(
        [
            frame_targets(
                manifest.loc[v, "fake_periods"], pred["lengths"][v], float(manifest.loc[v, "fps"])
            )
            for v in ids
        ]
    )


def per_class_frame_ap(pred: dict, manifest) -> dict[str, float]:
    """Table 8.2.4's frame-AP column: each fake class's frames plus every real clip's."""
    ids = list(pred["lengths"])
    cls = {v: str(manifest.loc[v, "class_name"]) for v in ids}
    out = {}
    for name in ("visual_only", "audio_only", "both"):
        keep = [v for v in ids if cls[v] in (name, "real")]
        if not any(cls[v] == name for v in keep):
            continue
        scores = np.concatenate([pred["frames"][v] for v in keep])
        out[name] = frame_metrics(scores, _targets(pred, keep, manifest))["frame_ap"]
    return out


def per_class_localization(pred: dict, gt: dict, cfg: PostProcessConfig, manifest) -> dict:
    """Section 8.2.4's collapse detector, for localization: AP@0.5 per fake class.

    Each class is scored on its own fakes plus every real video, so a false segment on a
    real clip costs every class equally -- the same convention as `per_class_breakdown`.
    """
    segs = segments(pred, cfg)
    real = [v for v, g in gt.items() if not g]
    out = {}
    for cls in ("visual_only", "audio_only", "both"):
        fakes = [v for v in gt if gt[v] and str(manifest.loc[v, "class_name"]) == cls]
        if not fakes:
            continue
        keep = fakes + real
        out[cls] = {
            "n_fake": len(fakes),
            "ap@0.5": average_precision_at_iou(
                {v: segs[v] for v in keep}, {v: gt[v] for v in keep}, 0.5
            ),
        }
    return out


# ---------------------------------------------------------------- aggregation


def summarise(reports: Sequence[dict]) -> dict:
    """Mean ± std (population, as every earlier phase reported) of each numeric field."""
    keys = [k for k, v in reports[0].items() if isinstance(v, int | float)]
    return {
        k: {
            "mean": float(np.nanmean([r[k] for r in reports])),
            "std": float(np.nanstd([r[k] for r in reports])),
        }
        for k in keys
    }


def compare(a: Sequence[dict], b: Sequence[dict], seeds_a, seeds_b, key: str) -> dict:
    """`b - a`, paired by seed: section 7.3 criteria 1 (magnitude) and 2 (consistency)."""
    pa = dict(zip(seeds_a, [r[key] for r in a], strict=True))
    pb = dict(zip(seeds_b, [r[key] for r in b], strict=True))
    shared = sorted(set(pa) & set(pb))
    diffs = np.array([pb[s] - pa[s] for s in shared])
    delta = float(np.mean([pb[s] for s in shared]) - np.mean([pa[s] for s in shared]))
    spread = max(float(np.std(list(pa.values()))), float(np.std(list(pb.values()))))
    consistent = bool(diffs.size and ((diffs > 0).all() or (diffs < 0).all()))
    return {
        "delta": delta,
        "seed_spread": spread,
        "per_seed": {int(s): float(pb[s] - pa[s]) for s in shared},
        "consistent": consistent,
        "significant": bool(abs(delta) > spread and consistent and len(shared) >= 3),
        "n_paired": len(shared),
    }
