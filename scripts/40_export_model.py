"""P12-2: export the deployable model -- one weights-only file, no ONNX (decision PF-29).

    python scripts/40_export_model.py

Writes `models/avtfd_g_seed0.pt`: the headline arm's Stage-B weights, the frozen MobileNetV2
backbone and the frozen section 6.4 post-processing config, i.e. everything
`src.inference.pipeline.VideoAnalyzer` needs and nothing it does not. `best.pt` is ~3x larger
because it carries Adam's two moment buffers per parameter.

**Which seed ships.** Experiment G trained 3 seeds; one model must serve. The rule, fixed
before any Phase 12 number existed: the seed with the **median dev AP@0.5** -- representative
of the reported mean rather than the best draw, so the demo neither flatters nor undersells
the results table. Computed from `reports/experiment_g.json`, never chosen by hand.

**Why not ONNX (PF-29).** The plan's P12-2 names ONNX as a speed lever. It is not one here:
the model is <2% of end-to-end time (P12-1), MediaPipe detection is already a CPU runtime,
the size gate is met in plain PyTorch, and Phases 13-14 run locally in the same Python
environment. ONNX would add a second copy of the model to keep in parity for no gain.

The file is also re-loaded here and checked tensor-for-tensor against `best.pt`, so an
export bug cannot reach P12-5's accuracy check disguised as an optimisation effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.inference.pipeline import build_bundle, bundle_parameter_count  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
SIZE_LIMIT_MB = 50.0


def median_seed(experiment_g: dict, arm: str) -> tuple[int, list[float]]:
    """Index of the run whose dev AP@0.5 is the median of the arm's seeds."""
    aps = [float(r["ap@0.5"]) for r in experiment_g["arms"][arm]["runs"]]
    order = np.argsort(aps, kind="stable")
    return int(order[len(aps) // 2]), aps


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P12-2: export the deployable model bundle")
    ap.add_argument("--experiment-g", default="reports/experiment_g.json")
    ap.add_argument("--frozen", default="configs/postprocess_frozen.json")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--out", default=None, help="default: models/avtfd_g_seed<idx>.pt")
    ap.add_argument("--facts", default="reports/model_export.json")
    args = ap.parse_args()

    g = json.loads(Path(args.experiment_g).read_text(encoding="utf-8"))
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    arm = frozen["arm"]
    idx, aps = median_seed(g, arm)
    run_dir = Path(args.exp_root) / f"phase10_{arm}" / f"seed{idx}"
    out = Path(args.out or f"models/avtfd_g_seed{idx}.pt")

    print(f"{DIM}{'=' * 72}{RESET}\n  P12-2  Export the deployable model\n{DIM}{'=' * 72}{RESET}")
    print(f"  arm '{arm}', dev AP@0.5 per seed {[round(a, 4) for a in aps]}")
    print(f"  median-AP@0.5 rule -> seed{idx} ({run_dir})")

    bundle = build_bundle(
        run_dir, frozen["config"], source={"arm": arm, "rule": "median dev AP@0.5"}
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)

    # Round-trip: what was written must be exactly what was trained.
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    identical = loaded["model"].keys() == ckpt["model"].keys() and all(
        torch.equal(loaded["model"][k], ckpt["model"][k]) for k in ckpt["model"]
    )

    size_mb = out.stat().st_size / 1024**2
    best_mb = (run_dir / "best.pt").stat().st_size / 1024**2
    landmarker = Path("models/face_landmarker.task")
    lm_mb = landmarker.stat().st_size / 1024**2 if landmarker.exists() else float("nan")
    params = bundle_parameter_count(bundle)

    print(
        f"  wrote {out}  {size_mb:.1f} MB  {DIM}(best.pt {best_mb:.1f} MB incl. optimizer){RESET}"
    )
    print(
        f"  params: Stage-B {params['model'] / 1e6:.2f}M, backbone {params['backbone'] / 1e6:.2f}M"
        f"  {DIM}+ face_landmarker.task {lm_mb:.1f} MB{RESET}"
    )
    print(
        f"  round-trip vs best.pt: {GREEN + 'identical' if identical else RED + 'MISMATCH'}{RESET}"
    )
    ok = identical and size_mb < SIZE_LIMIT_MB
    print(
        f"  size < {SIZE_LIMIT_MB:.0f} MB: {GREEN + 'yes' if size_mb < SIZE_LIMIT_MB else RED + 'NO'}{RESET}"
    )

    facts = {
        "path": out.as_posix(),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "size_mb": round(size_mb, 2),
        "best_pt_size_mb": round(best_mb, 2),
        "face_landmarker_mb": round(lm_mb, 2),
        "total_with_landmarker_mb": round(size_mb + lm_mb, 2),
        "size_limit_mb": SIZE_LIMIT_MB,
        "params_stage_b": params["model"],
        "params_backbone": params["backbone"],
        "dtype": "float32",
        "arm": arm,
        "seed_index": idx,
        "seed": bundle["source"].get("seed"),
        "dev_ap50_per_seed": aps,
        "selection_rule": "median dev AP@0.5 (reports/experiment_g.json)",
        "source_run": run_dir.as_posix(),
        "postprocess": frozen["config"],
        "round_trip_identical": identical,
        "git_sha": git_sha(),
    }
    Path(args.facts).write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {args.facts}")
    print(f"\n{GREEN if ok else RED}{'EXPORT OK' if ok else 'EXPORT FAILED'}{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
