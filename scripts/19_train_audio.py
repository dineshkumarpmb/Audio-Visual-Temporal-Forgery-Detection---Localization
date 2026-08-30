"""Phase 5 Stage-B: train the audio baseline, feed Decision C-1 (P5-4 .. P5-7).

    python scripts/19_train_audio.py --overfit-only              # P5-6, run this first
    python scripts/19_train_audio.py --feature mfcc   --seeds 3  # B2a
    python scripts/19_train_audio.py --feature logmel --seeds 3  # B2b

The plan calls this `scripts/05_extract_audio.py` + a trainer; extraction already happened
in Phase 3 (`scripts/12_extract_audio.py` writes both feature arms on the 40 ms video
grid), so this script is only Stage B.

**⛔ P5-6 runs before every training run and aborts it on failure.** It matters more here
than it did in Phase 4: the visual arm trained a small head on *frozen* features, while
this trains a four-block conv stack from scratch. Overfitting 10 samples is the cheapest
proof that gradients reach every block rather than dying in one.

**⛔ This is also the PF-17 control.** `audio_only` forgeries are pixel-identical to their
originals, so the visual baseline should have been at chance on them and instead scored
0.7551 off a clip-length shortcut. Audio is the modality that can genuinely see those
fakes. The per-class breakdown printed here is the comparison that says whether Phase 4's
number was skill or artefact — so `audio_only` is reported first, not buried in the table.

**Test is never touched** (section 3.5 RULE 4). Early stopping, model selection and every
number here come from dev.
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

from src.config import AudioPreprocessConfig, config_hash  # noqa: E402
from src.data.dataset import AudioFeatureDataset, collate  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.models.heads.classification import AudioBaseline  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.training.reporting import baseline_zero, git_sha, loader_for  # noqa: E402
from src.training.trainer import TrainConfig, Trainer, evaluate_split  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# What the visual arm scored on clips it provably cannot see the manipulation in (PF-17).
VISUAL_AUDIO_ONLY_AUC = 0.7551


def build_datasets(args, manifest, cfg: AudioPreprocessConfig):
    """Datasets for the official train/dev splits. Test is deliberately not built."""
    ids: list[str] = []
    for name in args.subsets:
        ids.extend(load_subset(name, args.subset_dir))
    ids = sorted(set(ids))

    full = AudioFeatureDataset(ids, manifest, cfg, args.feature_root)
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for vid, split in zip(full.video_ids, full.splits, strict=True):
        by_split[split].append(vid)

    datasets = {
        s: AudioFeatureDataset(v, manifest, cfg, args.feature_root)
        for s, v in by_split.items()
        if v
    }
    return full, datasets


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Train the Phase 5 audio baseline")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--feature-root", default="data/interim/audio")
    ap.add_argument("--feature", choices=("logmel", "mfcc"), default="logmel")
    ap.add_argument("--norm", choices=("batch", "group"), default="batch")
    ap.add_argument("--seeds", type=int, default=3, help="P5-4: independent runs")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--amp", action="store_true", help="PF-4: off locally, on for Kaggle")
    ap.add_argument("--overfit-only", action="store_true", help="run P5-6 and stop")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = AudioPreprocessConfig(feature=args.feature)
    manifest = read_manifest(args.manifest).set_index("video_id")

    try:
        full, datasets = build_datasets(args, manifest, cfg)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}")
        return 1

    tag = f"{args.feature}_{args.norm}" if args.norm != "batch" else args.feature
    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 5  Train audio baseline -- {args.feature}"
        f"\n{DIM}{'=' * 72}{RESET}"
    )
    probe = AudioBaseline(args.feature, dropout=args.dropout, norm=args.norm)
    einfo = probe.encoder.info(args.feature)
    print(f"  features : {config_hash(cfg)}  dim={einfo.in_dim}  {DIM}{args.feature_root}{RESET}")
    print(
        f"  encoder  : {einfo.feature_dim}-d, receptive field {einfo.receptive_field} frames "
        f"({einfo.receptive_field * 0.04:.2f} s), {DIM}norm={args.norm}, stride 1{RESET}"
    )
    print(f"  params   : {probe.n_parameters:,}")
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

    # ---- ⛔ P5-6 overfit-a-batch ------------------------------------------------
    print("\n  ⛔ P5-6 overfit-a-batch (10 samples)")
    seed_everything(1337)
    tiny = [datasets["train"][i] for i in range(min(10, len(datasets["train"])))]
    batch = collate(tiny)
    trainer = Trainer(
        AudioBaseline(args.feature, dropout=0.0, norm=args.norm),
        TrainConfig(lr=1e-2, seed=1337, amp=False),
        args.device,
    )
    losses = trainer.overfit_batch(batch, steps=300)
    passed = losses[-1] < 0.01
    print(
        f"  [{GREEN if passed else RED}{'PASS' if passed else 'FAIL'}{RESET}] "
        f"loss {losses[0]:.4f} -> {losses[-1]:.6f}"
    )
    if not passed:
        print(
            f"  {RED}The encoder cannot fit 10 samples. Check gradient flow through all four "
            f"conv blocks before training anything.{RESET}"
        )
        return 1
    if args.overfit_only:
        return 0

    # ---- Baseline 0 --------------------------------------------------------------
    dev_labels = datasets["dev"].labels
    floors = baseline_zero(dev_labels)
    print(f"\n  Baseline 0 on dev ({len(dev_labels)} clips)")
    for name, m in floors.items():
        print(f"    {name:<16} AUC {m['auc']:.4f}  acc {m['accuracy']:.4f}")

    # ---- P5-4 train, one run per seed --------------------------------------------
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
        run_dir = Path(args.out_root) / f"phase5_audio_{tag}" / f"seed{seed}"
        model = AudioBaseline(args.feature, dropout=args.dropout, norm=args.norm)
        runner = Trainer(model, tcfg, args.device, run_dir)

        t0 = time.time()
        runner.fit(train_loader, dev_loader)
        elapsed = time.time() - t0

        runner.resume(run_dir / "best.pt")  # evaluate the selected model, not the last
        dev_metrics = evaluate_split(runner, dev_loader)
        runner.write_history(run_dir / "history.json")

        record = {
            "seed": 1337 + seed,
            "feature": args.feature,
            "norm": args.norm,
            "git_sha": sha,
            "feature_config": config_hash(cfg),
            "n_params": model.n_parameters,
            "receptive_field": einfo.receptive_field,
            "epochs_run": len(runner.history),
            "best_epoch": runner.best_epoch,
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
            f"    best epoch {runner.best_epoch}  dev AUC {dev_metrics['auc']:.4f}  "
            f"acc {dev_metrics['accuracy']:.4f}  {elapsed:.1f}s"
        )

        if not args.no_mlflow:
            _log_mlflow(record, run_dir, args)

    # ---- summary -----------------------------------------------------------------
    aucs = np.array([r["dev"]["auc"] for r in runs])
    accs = np.array([r["dev"]["accuracy"] for r in runs])
    print(f"\n{DIM}{'-' * 72}{RESET}")
    print(f"  audio / {args.feature} over {len(runs)} seed(s)")
    print(
        f"    dev AUC  {aucs.mean():.4f} +/- {aucs.std():.4f}   "
        f"{DIM}[{aucs.min():.4f}, {aucs.max():.4f}]{RESET}"
    )
    print(f"    dev acc  {accs.mean():.4f} +/- {accs.std():.4f}")
    print(
        f"    vs Baseline 0 majority AUC {floors['majority_class']['auc']:.4f} -> "
        f"+{aucs.mean() - floors['majority_class']['auc']:.4f}"
    )

    # ---- ⛔ the PF-17 control ------------------------------------------------------
    audio_only = np.array([r["dev"]["per_class"]["audio_only"]["auc"] for r in runs])
    beats_visual = audio_only.mean() > VISUAL_AUDIO_ONLY_AUC
    print("\n  ⛔ PF-17 control -- audio_only (invisible to the visual arm)")
    print(
        f"    audio  {audio_only.mean():.4f} +/- {audio_only.std():.4f}   "
        f"vs visual {VISUAL_AUDIO_ONLY_AUC:.4f}   "
        f"{(GREEN if beats_visual else YELLOW)}"
        f"{audio_only.mean() - VISUAL_AUDIO_ONLY_AUC:+.4f}{RESET}"
    )
    if not beats_visual:
        print(
            f"    {YELLOW}Audio does not beat a visual model on fakes the visual model cannot\n"
            f"    see. That points at the shared clip-length shortcut, not at audio skill.{RESET}"
        )

    print("\n  section 8.2.4 four-class breakdown (best seed)")
    best = max(runs, key=lambda r: r["dev"]["auc"])
    for cls, m in best["dev"]["per_class"].items():
        if "auc" in m:
            print(f"    {cls:<12} n={m['n_fake']:>4}  AUC {m['auc']:.4f}  recall {m['recall']:.3f}")
        else:
            print(f"    {cls:<12} n={m['n']:>4}  specificity {m['specificity']:.3f}")

    out = Path(f"reports/phase5_audio_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "feature": args.feature,
                "norm": args.norm,
                "git_sha": sha,
                "n_clips": len(full),
                "n_params": runs[0]["n_params"],
                "receptive_field": einfo.receptive_field,
                "splits": {s: ds.label_balance for s, ds in datasets.items()},
                "baseline_zero": floors,
                "dev_auc_mean": float(aucs.mean()),
                "dev_auc_std": float(aucs.std()),
                "dev_acc_mean": float(accs.mean()),
                "audio_only_auc_mean": float(audio_only.mean()),
                "visual_audio_only_auc": VISUAL_AUDIO_ONLY_AUC,
                "beats_visual_on_audio_only": bool(beats_visual),
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
    """P4-8's tracking, reused for Phase 5. Never fatal -- tracking must not break training."""
    try:
        import mlflow

        store = Path("experiments/mlflow.db").resolve()
        store.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")
        mlflow.set_experiment("phase5_audio_baseline")
        with mlflow.start_run(run_name=f"audio_{args.feature}_seed{record['seed']}"):
            mlflow.log_params(
                {
                    "feature": record["feature"],
                    "norm": record["norm"],
                    "seed": record["seed"],
                    "git_sha": record["git_sha"],
                    "feature_config": record["feature_config"],
                    "n_params": record["n_params"],
                    "receptive_field": record["receptive_field"],
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
