"""P11-1: save dev predictions for trained runs that never wrote them.

    python scripts/36_dump_dev_predictions.py --arms phase8_full phase9_F0 phase9_F1 phase9_F2
    python scripts/36_dump_dev_predictions.py --arms phase10_full --verify

Phases 8 and 9 recorded metrics, not predictions, so Table 8.2.2 has no localization row for
E or F2. This re-runs each seed's `best.pt` on dev -- inference only, no training -- and
writes `dev_predictions.npz` beside it, in the layout `src/evaluation/runs.py` reads.

`--verify` re-predicts arms that *already* have the file and compares, so the rebuilt model
is shown to be the trained one before any new file is trusted. Dev only: the test split is
P11-8's, and `scripts/37_test_once.py` is the only thing that reads it.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioPreprocessConfig, VisualFeatureConfig  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.inference.predict import collect_predictions, load_attention_run  # noqa: E402
from src.utils.console import init_console  # noqa: E402


def _load_phase8():
    path = REPO_ROOT / "scripts" / "27_train_attention.py"
    spec = importlib.util.spec_from_file_location("phase8_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase8_sweep"] = module
    spec.loader.exec_module(module)
    return module


P8 = _load_phase8()
GREEN, RED, YELLOW, DIM, RESET = P8.GREEN, P8.RED, P8.YELLOW, P8.DIM, P8.RESET


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Save dev predictions from trained checkpoints")
    ap.add_argument("--arms", nargs="+", required=True, help="experiment dirs, e.g. phase8_full")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--verify", action="store_true", help="compare against existing files")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    manifest = read_manifest(args.manifest).set_index("video_id")
    loaders: dict[tuple[str, str], object] = {}
    failed = 0
    for arm in args.arms:
        for seed_dir in sorted((Path(args.exp_root) / arm).glob("seed*")):
            if not (seed_dir / "best.pt").exists():
                continue
            out = seed_dir / "dev_predictions.npz"
            if out.exists() and not (args.verify or args.force):
                print(f"  {arm}/{seed_dir.name}  {GREEN}already saved{RESET}")
                continue
            model, result = load_attention_run(seed_dir, args.device)
            key = (result["backbone"], result["feature"])
            if key not in loaders:
                args.backbone, args.feature = key
                _, datasets = P8.build_datasets(
                    args,
                    manifest,
                    VisualFeatureConfig(backbone=key[0]),
                    AudioPreprocessConfig(feature=key[1]),
                )
                loaders[key] = P8.paired_loader(
                    datasets["dev"], args.batch_size, shuffle=False, seed=0
                )
            preds = collect_predictions(model, loaders[key], args.device)
            preds_nv = collect_predictions(model, loaders[key], args.device, drop_visual=True)

            if args.verify and out.exists():
                old = np.load(out)
                same_ids = (old["video_ids"] == preds["video_ids"]).all()
                diff = float(np.abs(old["frame_scores"] - preds["frame_scores"]).max())
                ok = bool(same_ids and diff < 1e-5)
                failed += not ok
                print(
                    f"  {arm}/{seed_dir.name}  verify: max |Δ frame score| {diff:.2e}  "
                    f"[{GREEN + 'PASS' if ok else RED + 'FAIL'}{RESET}]"
                )
            else:
                np.savez_compressed(out, **preds)
                np.savez_compressed(
                    seed_dir / "dev_predictions_drop_visual.npz",
                    **{k: v for k, v in preds_nv.items() if k != "boundary_scores"},
                )
                print(
                    f"  {arm}/{seed_dir.name}  {GREEN}saved{RESET} {len(preds['video_ids'])} clips"
                )
            del model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
