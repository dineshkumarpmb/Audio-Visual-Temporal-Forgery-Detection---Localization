"""Phase 4 Stage-B: train the visual baseline, resolve D-1 (P4-6 .. P4-11).

    python scripts/14_train.py --overfit-only                 # P4-7, run this first
    python scripts/14_train.py --backbone resnet18 --seeds 3
    python scripts/14_train.py --backbone mobilenet_v2 --seeds 3

**⛔ P4-7 runs before every training run and aborts it on failure.** Driving 10 samples to
near-zero loss is the cheapest possible proof that gradients actually reach the weights.
If it fails the model is broken — a detached graph, a frozen parameter, a learning rate
that cannot move anything — and real training would only produce a confusing bad number
hours later instead of a clear failure in ten seconds.

**Baseline 0 (P4-10) is reported alongside every run**, because "0.82 AUC" means nothing
without knowing that always-guessing-fake scores 0.500. LAV-DF is 73.3% fake at video
level, so accuracy in particular is close to meaningless on its own.

**Test is never touched** (section 3.5 RULE 4). Early stopping, model selection and every
number printed here come from dev.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import VisualFeatureConfig, config_hash  # noqa: E402
from src.data.dataset import VisualFeatureDataset, collate  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.models.heads.classification import MeanPoolBaseline, VisualBaseline  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.training.reporting import baseline_zero, git_sha, loader_for  # noqa: E402
from src.training.trainer import TrainConfig, Trainer, evaluate_split  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def build_loaders(args, manifest, cfg: VisualFeatureConfig):
    """Datasets for the official train/dev splits. Test is deliberately not built."""
    ids: list[str] = []
    for name in args.subsets:
        ids.extend(load_subset(name, args.subset_dir))
    ids = sorted(set(ids))

    full = VisualFeatureDataset(
        ids, manifest, cfg, args.feature_root, require_face=not args.ignore_face_mask
    )
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for vid, split in zip(full.video_ids, full.splits, strict=True):
        by_split[split].append(vid)

    datasets = {
        s: VisualFeatureDataset(
            v, manifest, cfg, args.feature_root, require_face=not args.ignore_face_mask
        )
        for s, v in by_split.items()
        if v
    }
    return full, datasets


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Train the Phase 4 visual baseline")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--feature-root", default="data/features/visual")
    ap.add_argument("--backbone", choices=("resnet18", "mobilenet_v2"), default="resnet18")
    ap.add_argument("--pooling", choices=("attention", "mean"), default="attention")
    ap.add_argument("--seeds", type=int, default=3, help="P4-9: independent runs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--amp", action="store_true", help="PF-4: off locally, on for Kaggle")
    ap.add_argument("--ignore-face-mask", action="store_true", help="ablation")
    ap.add_argument("--overfit-only", action="store_true", help="run P4-7 and stop")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = VisualFeatureConfig(backbone=args.backbone)
    manifest = read_manifest(args.manifest).set_index("video_id")

    try:
        full, datasets = build_loaders(args, manifest, cfg)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}")
        return 1

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 4  Train -- {args.backbone} / {args.pooling}-pool"
        f"\n{DIM}{'=' * 72}{RESET}"
    )
    print(
        f"  features : {config_hash(cfg)}  dim={cfg.feature_dim}  {DIM}{args.feature_root}{RESET}"
    )
    print(
        f"  available: {len(full)} clips  {DIM}(of {len(full) + len(full.missing)} requested; "
        f"{len(full.missing)} not extracted){RESET}"
    )
    for s, ds in sorted(datasets.items()):
        b = ds.label_balance
        print(
            f"    {s:<6} {b['n']:>5} clips  {b['positive']:>4} fake / {b['negative']:>4} real "
            f"({b['positive_rate']:.1%})"
        )

    if "train" not in datasets or "dev" not in datasets:
        print(f"\n{RED}need both a train and a dev split{RESET}")
        return 1

    make_model = VisualBaseline if args.pooling == "attention" else MeanPoolBaseline

    # ---- ⛔ P4-7 overfit-a-batch ------------------------------------------------
    print("\n  ⛔ P4-7 overfit-a-batch (10 samples)")
    seed_everything(1337)
    tiny = [datasets["train"][i] for i in range(min(10, len(datasets["train"])))]
    batch = collate(tiny)
    probe = Trainer(
        make_model(cfg.feature_dim, dropout=0.0),
        TrainConfig(lr=1e-2, seed=1337, amp=False),
        args.device,
    )
    losses = probe.overfit_batch(batch, steps=300)
    passed = losses[-1] < 0.01
    print(
        f"  [{GREEN if passed else RED}{'PASS' if passed else 'FAIL'}{RESET}] "
        f"loss {losses[0]:.4f} -> {losses[-1]:.6f}"
    )
    if not passed:
        print(
            f"  {RED}The model cannot fit 10 samples. Check requires_grad, the graph, and lr "
            f"before training anything.{RESET}"
        )
        return 1
    if args.overfit_only:
        return 0

    # ---- Baseline 0 (P4-10) ------------------------------------------------------
    dev_labels = datasets["dev"].labels
    floors = baseline_zero(dev_labels)
    print(f"\n  P4-10 Baseline 0 on dev ({len(dev_labels)} clips)")
    for name, m in floors.items():
        print(f"    {name:<16} AUC {m['auc']:.4f}  acc {m['accuracy']:.4f}")

    # ---- P4-9 train, one run per seed --------------------------------------------
    train_loader = loader_for(datasets["train"], args.batch_size, shuffle=True, seed=0)
    dev_loader = loader_for(datasets["dev"], args.batch_size, shuffle=False, seed=0)
    pos_rate = datasets["train"].label_balance["positive_rate"]
    pos_weight = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else None

    runs = []
    sha = git_sha()
    for seed in range(args.seeds):
        print(f"\n  seed {seed}")
        seed_everything(1337 + seed)
        tcfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            pos_weight=pos_weight,
            patience=args.patience,
            amp=args.amp,
            seed=1337 + seed,
        )
        run_dir = Path(args.out_root) / f"phase4_{args.backbone}_{args.pooling}" / f"seed{seed}"
        model = make_model(cfg.feature_dim, dropout=args.dropout)
        trainer = Trainer(model, tcfg, args.device, run_dir)

        t0 = time.time()
        trainer.fit(train_loader, dev_loader)
        elapsed = time.time() - t0

        trainer.resume(run_dir / "best.pt")  # evaluate the selected model, not the last
        dev_metrics = evaluate_split(trainer, dev_loader)
        trainer.write_history(run_dir / "history.json")

        record = {
            "seed": 1337 + seed,
            "backbone": args.backbone,
            "pooling": args.pooling,
            "git_sha": sha,
            "feature_config": config_hash(cfg),
            "n_params": model.n_parameters,
            "epochs_run": len(trainer.history),
            "best_epoch": trainer.best_epoch,
            "wall_clock_s": round(elapsed, 2),
            "n_clips_train": len(datasets["train"]),
            "dev": dev_metrics,
            "train_config": asdict(tcfg),
        }
        (run_dir / "result.json").write_text(
            json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8"
        )
        runs.append(record)
        print(
            f"    best epoch {trainer.best_epoch}  dev AUC {dev_metrics['auc']:.4f}  "
            f"acc {dev_metrics['accuracy']:.4f}  {elapsed:.1f}s"
        )

        if not args.no_mlflow:
            _log_mlflow(record, run_dir, args)

    # ---- summary -----------------------------------------------------------------
    aucs = np.array([r["dev"]["auc"] for r in runs])
    accs = np.array([r["dev"]["accuracy"] for r in runs])
    print(f"\n{DIM}{'-' * 72}{RESET}")
    print(f"  {args.backbone} / {args.pooling}-pool over {len(runs)} seed(s)")
    print(
        f"    dev AUC  {aucs.mean():.4f} +/- {aucs.std():.4f}   "
        f"{DIM}[{aucs.min():.4f}, {aucs.max():.4f}]{RESET}"
    )
    print(f"    dev acc  {accs.mean():.4f} +/- {accs.std():.4f}")
    print(
        f"    vs Baseline 0 majority AUC {floors['majority_class']['auc']:.4f} -> "
        f"{GREEN if aucs.mean() > 0.6 else YELLOW}"
        f"+{aucs.mean() - floors['majority_class']['auc']:.4f}{RESET}"
    )

    print("\n  section 8.2.4 four-class breakdown (best seed)")
    best = max(runs, key=lambda r: r["dev"]["auc"])
    for cls, m in best["dev"]["per_class"].items():
        if "auc" in m:
            print(f"    {cls:<12} n={m['n_fake']:>4}  AUC {m['auc']:.4f}  recall {m['recall']:.3f}")
        else:
            print(f"    {cls:<12} n={m['n']:>4}  specificity {m['specificity']:.3f}")

    out = Path(f"reports/phase4_{args.backbone}_{args.pooling}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "pooling": args.pooling,
                "git_sha": sha,
                "n_clips": len(full),
                "splits": {s: ds.label_balance for s, ds in datasets.items()},
                "baseline_zero": floors,
                "dev_auc_mean": float(aucs.mean()),
                "dev_auc_std": float(aucs.std()),
                "dev_acc_mean": float(accs.mean()),
                "runs": runs,
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  wrote {out}")
    print(f"\n{GREEN}TRAINING DONE{RESET}")
    return 0


def _log_mlflow(record: dict, run_dir: Path, args) -> None:
    """P4-8: local MLflow tracking. Never fatal -- tracking must not break training.

    The plan says "local, file-backed". MLflow 3.x **deprecated the bare filesystem store**
    and raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set. SQLite is the migration path
    MLflow itself recommends and is still a single local file with no server, so it keeps
    the plan's intent (nothing to run, nothing to host) while actually working. Recorded as
    decision PF-14.
    """
    try:
        import mlflow

        store = Path("experiments/mlflow.db").resolve()
        store.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")
        mlflow.set_experiment("phase4_visual_baseline")
        with mlflow.start_run(run_name=f"{args.backbone}_{args.pooling}_seed{record['seed']}"):
            mlflow.log_params(
                {
                    "backbone": record["backbone"],
                    "pooling": record["pooling"],
                    "seed": record["seed"],
                    "git_sha": record["git_sha"],
                    "feature_config": record["feature_config"],
                    "n_params": record["n_params"],
                    "n_clips_train": record.get("n_clips_train"),
                    **{f"train_{k}": v for k, v in record["train_config"].items()},
                }
            )
            mlflow.log_metrics(
                {
                    f"dev_{k}": float(v)
                    for k, v in record["dev"].items()
                    if isinstance(v, int | float)
                }
            )
            mlflow.log_metrics(
                {
                    "wall_clock_s": record["wall_clock_s"],
                    "best_epoch": record["best_epoch"],
                    "epochs_run": record["epochs_run"],
                }
            )
            for name in ("best.pt", "history.json", "result.json"):
                path = run_dir / name
                if path.exists():
                    mlflow.log_artifact(str(path))
    except Exception as exc:  # noqa: BLE001
        print(
            f"    {YELLOW}mlflow logging skipped: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:120]}{RESET}"
        )


if __name__ == "__main__":
    sys.exit(main())
