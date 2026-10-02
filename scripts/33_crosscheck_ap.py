"""⛔ P10-7: cross-check interval AP/AR against the LAV-DF authors' own evaluator.

    python scripts/33_crosscheck_ap.py --reference <dir containing avdeepfake1m/>

Section 6.7: "where possible, cross-check against an established implementation". The
established one here is the dataset authors' ``avdeepfake1m.evaluation`` (the evaluator
behind the published LAV-DF / AV-Deepfake1M numbers). It is a compiled Rust extension, so
its conventions were *measured*, not read -- see ``src/evaluation/localization.py``.

The reference is **not** a project dependency: its wheel pins numpy<2 and pandas~=2.0 and
pulls in lightning and torchvision. Fetch it into a scratch directory and point this script
at it::

    pip download avdeepfake1m==0.0.4 --no-deps -d <tmp>
    python -m zipfile -e <tmp>/avdeepfake1m-0.0.4-*.whl <tmp>/avdf

Only the extension module is loaded (by path), so none of the package's heavy imports run.

**What is asserted.** On ``--cases`` random problems (several videos, 0-2 true spans each,
some real videos, predictions ranging from near-perfect to spurious, thresholds spanning
0.1-0.95), ``convention="lavdf"`` must equal the reference to 1e-6 for AP *and* AR. The
``standard`` convention's gap to the reference is reported alongside but not asserted: the
dropped first recall step makes the reference lower, while matching by list order rather
than by confidence can move it either way.

**Float32 threshold ties.** The reference computes IoU in float32, so an IoU of *exactly*
0.75 comes out as 0.75000006 and passes its strict ``> 0.75``; in exact arithmetic it does
not. That only happens when inputs sit on a coarse grid. So the check runs twice: on
**unrounded** inputs, where exact ties have probability zero and agreement must be perfect,
and on inputs **rounded to 3 decimals**, where every disagreement must be traced to an IoU
within 1e-6 of a threshold -- anything else fails the check.

Two reference crashes, worked around rather than hidden. A video with an **empty
prediction list** panics inside the extension's JSON reader ("shape should be allocated by
now"), and a labelled video with **no prediction entry** panics on lookup ("no entry found
for key"). So a video without predictions is given one placeholder, ``(1000, 1001, 0.0)``:
outside every clip (IoU 0 with everything) and ranked below every real prediction. The
*same* augmented input goes to both implementations, so the comparison stays exact; the
placeholder is a trailing false positive, which moves no recall step and so no AP or AR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.localization import (  # noqa: E402
    AR_IOU_THRESHOLDS,
    average_precision_at_iou,
    average_recall_at_n,
    iou_matrix,
)
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def load_reference(root: Path):
    hits = sorted((root / "avdeepfake1m").glob("_evaluation*.pyd")) + sorted(
        (root / "avdeepfake1m").glob("_evaluation*.so")
    )
    if not hits:
        raise FileNotFoundError(f"no avdeepfake1m/_evaluation extension under {root}")
    spec = importlib.util.spec_from_file_location("_evaluation", hits[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_case(rng: np.random.Generator, decimals: int | None = None) -> tuple[dict, dict]:
    def r(x: float) -> float:
        return round(x, decimals) if decimals is not None else x

    preds, gt = {}, {}
    for v in range(int(rng.integers(1, 7))):
        vid = f"{v:03d}.mp4"
        duration = float(rng.uniform(4, 20))
        spans = []
        for _ in range(int(rng.integers(0, 3))):  # 0 -> a real video
            length = float(rng.uniform(0.2, 1.6))
            start = float(rng.uniform(0, duration - length))
            spans.append([r(start), r(start + length)])
        gt[vid] = spans
        segs = []
        for s in spans:  # a noisy detection of each true span, most of the time
            if rng.uniform() < 0.7:
                a = s[0] + float(rng.normal(0, 0.15))
                b = s[1] + float(rng.normal(0, 0.15))
                if b > a:
                    segs.append([r(a), r(b), float(rng.uniform())])
        for _ in range(int(rng.integers(0, 4))):  # spurious ones
            a = float(rng.uniform(0, duration - 0.3))
            segs.append([r(a), r(a + float(rng.uniform(0.2, 1.5))), float(rng.uniform())])
        preds[vid] = segs
    return preds, gt


PLACEHOLDER = [1000.0, 1001.0, 0.0]


def with_placeholders(preds: dict) -> dict:
    return {vid: (segs if segs else [PLACEHOLDER]) for vid, segs in preds.items()}


def reference_scores(ev, preds, gt, ap_thr, ar_n, ar_thr) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        p, g = Path(tmp) / "pred.json", Path(tmp) / "gt.json"
        # Reference format: {file: [[confidence, start, end], ...]}, sorted per video.
        ref_preds = {
            vid: [[s[2], s[0], s[1]] for s in sorted(segs, key=lambda x: -x[2])]
            for vid, segs in preds.items()
        }
        p.write_text(json.dumps(ref_preds), encoding="utf-8")
        g.write_text(
            json.dumps([{"file": vid, "fake_segments": spans} for vid, spans in gt.items()]),
            encoding="utf-8",
        )
        return ev.ap_ar_1d(str(p), str(g), "file", "fake_segments", 1.0, ap_thr, ar_n, ar_thr)


def has_threshold_tie(preds: dict, gt: dict, thresholds: list[float]) -> bool:
    """True if any prediction/ground-truth IoU sits within 1e-6 of a threshold."""
    th = np.asarray(thresholds)
    for vid, spans in gt.items():
        if not spans or not preds.get(vid):
            continue
        ious = iou_matrix(np.asarray(preds[vid])[:, :2], np.asarray(spans))
        if (np.abs(ious[..., None] - th) < 1e-6).any():
            return True
    return False


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(
        description="Cross-check interval AP/AR against LAV-DF's evaluator"
    )
    ap.add_argument("--reference", required=True, help="directory holding avdeepfake1m/")
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/ap_crosscheck.json")
    args = ap.parse_args()

    ev = load_reference(Path(args.reference))
    ap_thr = [0.1, 0.3, 0.5, 0.75, 0.95]
    ar_n = [1, 2, 5, 100]
    ar_thr = list(AR_IOU_THRESHOLDS)

    def sweep(decimals: int | None) -> dict:
        rng = np.random.default_rng(args.seed)
        worst_ap = worst_ar = 0.0
        std_gap, failures, tie_explained = [], [], 0
        n_checked = n_skipped = 0
        for case in range(args.cases):
            preds, gt = random_case(rng, decimals)
            preds = with_placeholders(preds)  # see module docstring
            if not any(gt.values()):
                n_skipped += 1  # both implementations are undefined with no ground truth
                continue
            ref = reference_scores(ev, preds, gt, ap_thr, ar_n, ar_thr)
            tied = has_threshold_tie(preds, gt, ap_thr + ar_thr)
            for t in ap_thr:
                ref_ap = [v for k, v in ref["ap"].items() if abs(float(k) - t) < 1e-5][0]
                mine = average_precision_at_iou(preds, gt, t, convention="lavdf")
                if abs(mine - ref_ap) > 1e-6:
                    if tied:
                        tie_explained += 1
                    else:
                        failures.append(
                            {"case": case, "metric": f"ap@{t}", "ref": ref_ap, "mine": mine}
                        )
                else:
                    worst_ap = max(worst_ap, abs(mine - ref_ap))
                std_gap.append(average_precision_at_iou(preds, gt, t) - ref_ap)
            for n in ar_n:
                ref_ar = [v for k, v in ref["ar"].items() if int(k) == n][0]
                mine = average_recall_at_n(preds, gt, n, ar_thr, convention="lavdf")
                if abs(mine - ref_ar) > 1e-6:
                    if tied:
                        tie_explained += 1
                    else:
                        failures.append(
                            {"case": case, "metric": f"ar@{n}", "ref": ref_ar, "mine": mine}
                        )
                else:
                    worst_ar = max(worst_ar, abs(mine - ref_ar))
            n_checked += 1
        return {
            "decimals": decimals,
            "cases_checked": n_checked,
            "cases_skipped_no_gt": n_skipped,
            "worst_abs_diff_ap_agreeing": worst_ap,
            "worst_abs_diff_ar_agreeing": worst_ar,
            "disagreements_explained_by_float32_threshold_ties": tie_explained,
            "unexplained_failures": failures[:50],
            "n_unexplained": len(failures),
            "standard_minus_reference_ap": {
                "mean": float(np.mean(std_gap)),
                "min": float(np.min(std_gap)),
                "max": float(np.max(std_gap)),
            },
        }

    exact = sweep(None)
    grid = sweep(3)
    # On unrounded inputs nothing may disagree at all, tie-explained or not.
    passed = (
        exact["n_unexplained"] == 0
        and exact["disagreements_explained_by_float32_threshold_ties"] == 0
        and grid["n_unexplained"] == 0
    )
    failures = exact["unexplained_failures"] + grid["unexplained_failures"]
    for sw, label in ((exact, "unrounded"), (grid, "3-decimal grid")):
        print(
            f"  {label:<15} cases {sw['cases_checked']:>4}  worst |dAP| "
            f"{sw['worst_abs_diff_ap_agreeing']:.1e}  |dAR| "
            f"{sw['worst_abs_diff_ar_agreeing']:.1e}  float32 ties "
            f"{sw['disagreements_explained_by_float32_threshold_ties']}  "
            f"unexplained {sw['n_unexplained']}"
        )
    gap = exact["standard_minus_reference_ap"]
    print(
        f"  standard - reference AP: mean {gap['mean']:+.4f}  min {gap['min']:+.4f}  "
        f"max {gap['max']:+.4f}  {DIM}(the reference drops the first recall step and "
        f"matches by list order){RESET}"
    )
    print(
        f"\n  {(GREEN + 'CROSS-CHECK PASSED') if passed else (RED + 'CROSS-CHECK FAILED')}{RESET}"
    )
    for f in failures[:10]:
        print(f"    {RED}{f}{RESET}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "reference": "avdeepfake1m 0.0.4 (_evaluation extension)",
                "seed": args.seed,
                "ap_thresholds": ap_thr,
                "ar_n": ar_n,
                "ar_iou_thresholds": ar_thr,
                "unrounded": exact,
                "grid_3_decimals": grid,
                "passed": passed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
