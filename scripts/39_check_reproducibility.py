"""P11-10: re-run one experiment from its logged config and confirm the metrics match.

    python scripts/34_train_localization.py --arms full --seeds 1 --out-root experiments/_repro --skip-overfit
    python scripts/39_check_reproducibility.py

Compares the re-run's `result.json` and `dev_predictions.npz` against the original's. The
plan asks for agreement to ~1e-4; with `seed_everything(deterministic=True)` on the same GPU
a deterministic re-run should agree far more closely than that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
KEYS = ("auc", "ap", "f1", "frame_ap", "frame_auc")


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P11-10 reproducibility check")
    ap.add_argument("--original", default="experiments/phase10_full/seed0")
    ap.add_argument("--rerun", default="experiments/_repro/phase10_full/seed0")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--out", default="reports/reproducibility.json")
    args = ap.parse_args()

    a = json.loads((Path(args.original) / "result.json").read_text(encoding="utf-8"))
    b = json.loads((Path(args.rerun) / "result.json").read_text(encoding="utf-8"))
    same_cfg = a["train_config"] == b["train_config"]
    rows = {k: {"original": a["dev"][k], "rerun": b["dev"][k]} for k in KEYS}
    for r in rows.values():
        r["abs_diff"] = abs(r["original"] - r["rerun"])
    pa = np.load(Path(args.original) / "dev_predictions.npz")
    pb = np.load(Path(args.rerun) / "dev_predictions.npz")
    frame_diff = float(np.abs(pa["frame_scores"] - pb["frame_scores"]).max())
    checks = {
        "identical logged train_config": same_cfg,
        "same best epoch": a["best_epoch"] == b["best_epoch"],
        f"every dev metric within {args.tol:g}": all(
            r["abs_diff"] <= args.tol for r in rows.values()
        ),
    }
    passed = all(checks.values())
    print(f"  original {args.original} ({a['git_sha'][:8]})  vs  re-run {args.rerun}")
    for k, r in rows.items():
        print(f"    {k:<10} {r['original']:.6f}  {r['rerun']:.6f}  |Δ| {r['abs_diff']:.2e}")
    print(f"    max |Δ frame score| {frame_diff:.2e}")
    for k, v in checks.items():
        print(f"    {k:<36}: {v}")
    print(f"\n  {(GREEN if passed else RED)}P11-10 {'PASSED' if passed else 'FAILED'}{RESET}")
    Path(args.out).write_text(
        json.dumps(
            {
                "original": args.original,
                "rerun": args.rerun,
                "original_git_sha": a["git_sha"],
                "rerun_git_sha": b["git_sha"],
                "checked_at_git_sha": git_sha(),
                "best_epoch": [a["best_epoch"], b["best_epoch"]],
                "metrics": rows,
                "max_frame_score_diff": frame_diff,
                "tolerance": args.tol,
                "checks": checks,
                "passed": passed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
