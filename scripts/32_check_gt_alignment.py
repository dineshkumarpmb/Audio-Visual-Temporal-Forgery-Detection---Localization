"""⛔ P10-1 / P10-2: the ground-truth round trip, over every entry in the manifest.

    python scripts/32_check_gt_alignment.py

Section 6.3 calls `fake_periods -> [T] target` "the highest-risk step in the project".
Phase 2 and 3 verified the *labels* against the media (7/7 and 12/12 pairs, median error
0.000 s); this checks the *transform*, on all 136,304 entries rather than a sample:

* **P10-1** -- spans -> target -> intervals reproduces every labelled span to within one
  frame (`src.localization.targets.check_roundtrip`);
* **P10-2** -- every real video's target sums to exactly 0.

`n_frames` is the manifest's video frame count. Training uses the feature array's length,
which can be up to 3 frames shorter (the common audio/video grid); spans past that grid are
clipped the same way in both, and `check_roundtrip` compares clipped spans to their clipped
extent -- so this is the stricter of the two lengths, not a looser one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.localization.targets import check_roundtrip  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P10-1/P10-2 ground-truth round trip")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--out", default="reports/gt_alignment.json")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest)
    t0 = time.time()
    worst, n_fail, n_real, n_real_bad, n_fake, n_spans, n_merged = 0.0, 0, 0, 0, 0, 0, 0
    failures = []
    for vid, periods, n_frames, fps, label in zip(
        manifest["video_id"],
        manifest["fake_periods"],
        manifest["n_frames"],
        manifest["fps"],
        manifest["label"],
        strict=True,
    ):
        out = check_roundtrip(periods, int(n_frames), float(fps))
        if int(label) == 0:
            n_real += 1
            if out["n_spans"] or not out["real_target_exactly_zero"]:
                n_real_bad += 1
                failures.append({"video_id": vid, "why": "real video with a non-zero target"})
            continue
        n_fake += 1
        n_spans += out["n_spans"]
        n_merged += out["n_spans"] - out["n_runs"]
        worst = max(worst, out["worst_error_frames"])
        if not out["within_one_frame"]:
            n_fail += 1
            failures.append({"video_id": vid, **out})

    passed = n_fail == 0 and n_real_bad == 0
    print(f"  entries        {len(manifest):,}  ({time.time() - t0:.1f}s)")
    print(f"  fake videos    {n_fake:,}  spans {n_spans:,}  worst error {worst:.3f} frames")
    print(f"  merged at frame resolution  {n_merged}  {DIM}(spans that touch/overlap){RESET}")
    print(f"  ⛔ P10-1 within one frame: {n_fake - n_fail:,}/{n_fake:,}")
    print(f"  ⛔ P10-2 real target == 0: {n_real - n_real_bad:,}/{n_real:,}")
    print(f"\n  {(GREEN + 'PASSED') if passed else (RED + 'FAILED')}{RESET}")
    for f in failures[:10]:
        print(f"    {RED}{f}{RESET}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "n_entries": int(len(manifest)),
                "n_fake": n_fake,
                "n_spans": n_spans,
                "n_spans_merged_at_frame_resolution": n_merged,
                "worst_error_frames": float(worst),
                "p10_1_failures": n_fail,
                "n_real": n_real,
                "p10_2_failures": n_real_bad,
                "failures": failures[:50],
                "passed": passed,
                "labels_are_seconds": bool(np.isfinite(worst)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
