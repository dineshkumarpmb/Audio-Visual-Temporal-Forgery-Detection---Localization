"""⛔ P11-8: score the test split exactly once, with everything frozen.

    python scripts/38_test_once.py --final

Section 7.1 rule 4 / section 3.5 RULE 4: dev for all decisions, test once at the end. This
script is the only code in the project that reads the test split, and it refuses to run
twice. Before it predicts anything it requires:

1. no `reports/test_once.lock` and no `test_predictions*.npz` anywhere under `experiments/`;
2. `configs/postprocess_frozen.json` exists and records that no test prediction existed
   when it was frozen;
3. every Phase 11 ablation (H, I) finished on 3 seeds -- no decision is still open;
4. at least `--min-test-clips` test clips with features, so the one run is not spent on
   a handful of clips by accident;
5. `--final` on the command line.

**The lock is written before the first prediction**, so a crash halfway still counts as the
test having been touched. Scoring re-uses `src/evaluation/runs.py` -- the same functions as
every dev number -- with the frozen post-processing config, unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioPreprocessConfig, VisualFeatureConfig  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.metrics import roc_auc  # noqa: E402
from src.evaluation.runs import (  # noqa: E402
    classification_metrics,
    ground_truth,
    load_predictions,
    localization_metrics,
    per_class_frame_ap,
    per_class_localization,
    summarise,
)
from src.inference.predict import collect_predictions, load_attention_run  # noqa: E402
from src.localization.postprocess import PostProcessConfig  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
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
LOCK = Path("reports/test_once.lock")
ABLATIONS = ("phase11_H_nosync", "phase11_I_nomd")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preconditions(args, root: Path, frozen: dict | None) -> dict[str, bool]:
    done = [len(list((root / a).glob("seed*/result.json"))) >= 3 for a in ABLATIONS]
    return {
        "test never scored before (no lock, no test predictions)": not LOCK.exists()
        and not list(root.glob("*/seed*/test_predictions*.npz")),
        "post-processing frozen on dev before any test prediction": bool(
            frozen
            and frozen.get("tuned_on") == "dev"
            and not frozen.get("test_split_scored_before_freeze", True)
        ),
        "every Phase 11 ablation finished on 3 seeds": all(done),
        "--final given": bool(args.final),
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P11-8: the single test-split run")
    ap.add_argument("--final", action="store_true", help="required: this can only happen once")
    ap.add_argument("--exp-root", default="experiments")
    ap.add_argument("--frozen", default="configs/postprocess_frozen.json")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-test-clips", type=int, default=300)
    ap.add_argument("--out", default="reports/test_once.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.exp_root)
    fz_path = Path(args.frozen)
    frozen = json.loads(fz_path.read_text(encoding="utf-8")) if fz_path.exists() else None
    print(f"{DIM}{'=' * 72}{RESET}\n  ⛔ P11-8  The single test run\n{DIM}{'=' * 72}{RESET}")
    pre = preconditions(args, root, frozen)
    for k, v in pre.items():
        print(f"    [{GREEN + 'ok' if v else RED + 'NO'}{RESET}] {k}")
    if not all(pre.values()):
        print(f"\n{RED}refusing to touch the test split{RESET}")
        return 1

    arm = frozen["arm"]
    arm_dir = root / f"phase10_{arm}"
    seed_dirs = sorted(d for d in arm_dir.glob("seed*") if (d / "best.pt").exists())
    cfg = PostProcessConfig(**frozen["config"])

    manifest = read_manifest(args.manifest).set_index("video_id")
    first = json.loads((seed_dirs[0] / "result.json").read_text(encoding="utf-8"))
    args.backbone, args.feature = first["backbone"], first["feature"]
    _, datasets = P8.build_datasets(
        args,
        manifest,
        VisualFeatureConfig(backbone=args.backbone),
        AudioPreprocessConfig(feature=args.feature),
    )
    test = datasets.get("test")
    n_test = 0 if test is None else len(test)
    print(f"\n  arm '{arm}', {len(seed_dirs)} seeds, {n_test} test clips with features")
    if n_test < args.min_test_clips:
        print(
            f"{RED}only {n_test} test clips (< {args.min_test_clips}); extract the test "
            f"split's features first -- the one test run is not spent on this{RESET}"
        )
        return 1

    # ⛔ Lock first. From here on the test split counts as touched, whatever happens next.
    sha = git_sha()
    scored_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    LOCK.write_text(
        json.dumps(
            {
                "scored_at": scored_at,
                "git_sha": sha,
                "arm": arm,
                "frozen": str(fz_path),
                "frozen_sha256": _sha256(fz_path),
                "n_test_clips": n_test,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  {YELLOW}lock written -> {LOCK}{RESET}")

    loader = P8.paired_loader(test, args.batch_size, shuffle=False, seed=0)
    per_seed, preds_paths = [], {}
    for seed_dir in seed_dirs:
        model, result = load_attention_run(seed_dir, args.device)
        preds = collect_predictions(model, loader, args.device)
        preds_nv = collect_predictions(model, loader, args.device, drop_visual=True)
        p = seed_dir / "test_predictions.npz"
        np.savez_compressed(p, **preds)
        np.savez_compressed(
            seed_dir / "test_predictions_drop_visual.npz",
            **{k: v for k, v in preds_nv.items() if k != "boundary_scores"},
        )
        preds_paths[seed_dir.name] = {"path": p.as_posix(), "sha256": _sha256(p)}
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

        pred = load_predictions(p)
        pred_nv = load_predictions(seed_dir / "test_predictions_drop_visual.npz")
        gt = ground_truth(manifest, pred["lengths"])
        cls = classification_metrics(pred, manifest)
        loc = localization_metrics(pred, gt, cfg, pred_nv)
        ids = list(pred["lengths"])
        labels = np.array([pred["labels"][v] for v in ids])
        auc_nv = roc_auc(np.array([pred_nv["clip_scores"][v] for v in ids]), labels)
        row = {
            "seed": seed_dir.name,
            **{k: v for k, v in cls.items() if isinstance(v, int | float)},
            **{k: v for k, v in loc.items() if isinstance(v, int | float)},
            "visual_reliance_auc": cls["auc"] - auc_nv,
            "visual_reliance_ap50": loc["ap@0.5_no_refine"] - loc["ap@0.5_drop_visual"],
            "per_class": cls["per_class"],
            "confusion": cls["confusion"],
            "per_class_frame_ap": per_class_frame_ap(pred, manifest),
            "per_class_localization": per_class_localization(pred, gt, cfg, manifest),
            "dev_auc": result["dev"]["auc"],
        }
        per_seed.append(row)
        print(
            f"    {seed_dir.name}  AUC {row['auc']:.4f}  AP@0.5 {row['ap@0.5']:.4f}  "
            f"AP@0.75 {row['ap@0.75']:.4f}  frame AP {row['frame_ap']:.4f}"
        )

    summ = summarise(per_seed)
    four = {}
    for name in ("visual_only", "audio_only", "both"):
        if all(name in r["per_class"] for r in per_seed):
            vals = {
                "auc": [r["per_class"][name]["auc"] for r in per_seed],
                "frame_ap": [r["per_class_frame_ap"][name] for r in per_seed],
                "ap@0.5": [r["per_class_localization"][name]["ap@0.5"] for r in per_seed],
            }
            four[name] = {
                "n": per_seed[0]["per_class"][name]["n_fake"],
                **{
                    k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in vals.items()
                },
            }
    labels_n = per_seed[0]["n"]
    n_fake = int(round(per_seed[0]["positive_rate"] * labels_n))
    facts = {
        "scored_at": scored_at,
        "git_sha": sha,
        "n_test_runs": 1,
        "arm": arm,
        "n_seeds": len(per_seed),
        "n_clips": labels_n,
        "n_fake": n_fake,
        "n_real": labels_n - n_fake,
        "frozen_config": frozen["config"],
        "frozen_config_source": str(fz_path),
        "frozen_at": frozen["frozen_at"],
        "predictions": preds_paths,
        "summary": summ,
        "four_class": four,
        "per_seed": per_seed,
    }
    out = Path(args.out)
    out.with_suffix(".json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )
    write_markdown(out, facts)
    print(f"\n  {GREEN}done{RESET} -- wrote {out} and {out.with_suffix('.json')}")
    return 0


def _pm(s: dict, k: str, fmt: str = ".4f") -> str:
    return f"{s[k]['mean']:{fmt}} ± {s[k]['std']:{fmt}}"


def write_markdown(path: Path, f: dict) -> None:
    s = f["summary"]
    lines = [
        "# ⛔ P11-8 — the single test run",
        "",
        f"**Test split, scored once.** Arm `{f['arm']}` (Experiment G's headline), "
        f"{f['n_seeds']} seeds, {f['n_clips']} clips ({f['n_fake']} fake / {f['n_real']} real).",
        "",
        f"- scored at {f['scored_at']}, git `{f['git_sha'][:12]}`",
        f"- post-processing: `{f['frozen_config_source']}`, frozen on dev at {f['frozen_at']} — "
        f"`{json.dumps(f['frozen_config'])}`",
        "- model selection, λ, post-processing and every Phase 11 decision were made on dev",
        "  before this ran; nothing was changed after it.",
        "",
        "| metric | test (mean ± std over seeds) |",
        "|---|---|",
    ]
    for k, fmt in (
        ("auc", ".4f"),
        ("ap", ".4f"),
        ("accuracy", ".4f"),
        ("precision", ".4f"),
        ("recall", ".4f"),
        ("f1", ".4f"),
        ("frame_ap", ".4f"),
        ("ap@0.5", ".4f"),
        ("ap@0.75", ".4f"),
        ("ap@0.95", ".4f"),
        ("ar@100", ".4f"),
        ("ar@10", ".4f"),
        ("boundary_error_ms", ".1f"),
        ("fp_video_rate", ".3f"),
        ("visual_reliance_auc", ".4f"),
        ("visual_reliance_ap50", ".4f"),
    ):
        lines.append(f"| {k} | {_pm(s, k, fmt)} |")
    lines += [
        "",
        "## 4-class breakdown (test)",
        "",
        "| class | N | video AUC | frame AP | AP@0.5 |",
        "|---|---|---|---|---|",
    ]
    for name, d in f["four_class"].items():
        lines.append(
            f"| {name} | {d['n']} | {_pm(d, 'auc')} | {_pm(d, 'frame_ap')} | {_pm(d, 'ap@0.5')} |"
        )
    lines += [
        "",
        "## Prediction files (sha256)",
        "",
        "| seed | file | sha256 |",
        "|---|---|---|",
    ]
    for seed, p in f["predictions"].items():
        lines.append(f"| {seed} | `{p['path']}` | `{p['sha256'][:16]}…` |")
    lines += [
        "",
        "> ⚠️ PF-17 / PF-19 / PF-22 apply to test as to dev: the absolute numbers are upper",
        "> bounds on this data. The test split is drawn from the same LAV-DF release, so this",
        "> measures held-out *clips*, not generalisation to unseen generators.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
