"""Phase 7 Stage-B: train Baseline 4 and run Experiment D (P7-1 .. P7-5).

    python scripts/24_train_temporal.py --overfit-only     # gradient check first
    python scripts/24_train_temporal.py --seeds 3          # Experiment D
    python scripts/24_train_temporal.py --seeds 3 --frame-weight 0   # the ablation arm

Baseline 4 = Phase 6's fusion **plus a 2-layer BiLSTM** and the first per-frame head. Same
786 clips, same dev split, same MobileNetV2 (D-1) + log-Mel (C-1) arms, same defences, same
video-level head — so `D − C` is the recurrence and nothing else.

**Selection stays on dev *clip* AUC, deliberately.** Frame AP is the new metric but it is
not what early stopping watches, because Experiment D compares against Phase 6's C, which
was selected on clip AUC. Switching the selection criterion at the same time as adding the
BiLSTM would make the comparison measure two changes at once. Frame AP is *reported* at the
selected checkpoint, which is what P7-4 asks for.

**⛔ The collapse diagnostics come along unchanged.** X-7 says re-check R4 at every
scale-up, and adding a component that can carry information across time is exactly the kind
of change that can let one modality dominate anew. So every run still reports the modality
ablation and the 4-class breakdown, on the same tolerance Phase 6 used.

**Test is never touched** (section 3.5 RULE 4).
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

from src.config import AudioPreprocessConfig, VisualFeatureConfig, config_hash  # noqa: E402
from src.data.dataset import MultimodalFeatureDataset, collate_paired  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    frame_metrics,
    per_class_breakdown,
    summary,
)
from src.localization.targets import frame_targets  # noqa: E402
from src.models.backbones.audio import NATIVE_DIM  # noqa: E402
from src.models.temporal.lstm import TemporalFusionBaseline  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.training.reporting import baseline_zero, git_sha  # noqa: E402
from src.training.trainer import TrainConfig, Trainer, evaluate_split  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

VISUAL_DIM = {"resnet18": 512, "mobilenet_v2": 1280}


def paired_loader(ds, batch_size: int, *, shuffle: bool, seed: int):
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_paired,
        num_workers=0,
        generator=g if shuffle else None,
    )


@torch.no_grad()
def ablate(trainer, loader, *, drop_visual: bool = False, drop_audio: bool = False) -> dict:
    """Section 5.6's collapse probe, extended to the frame head.

    Identical to Phase 6's except that it also re-scores frames, because the question
    "which modality is the *localization* actually coming from?" is new at Phase 7 and has
    no reason to have the same answer as the clip-level one.
    """
    trainer.model.eval()
    scores, labels, classes = [], [], []
    fs, ft = [], []
    for batch in loader:
        out = trainer.model(
            batch["features"].to(trainer.device),
            batch["audio"].to(trainer.device),
            batch["mask"].to(trainer.device),
            batch["face"].to(trainer.device),
            drop_visual=drop_visual,
            drop_audio=drop_audio,
        )
        scores.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
        labels.append(batch["label"].numpy())
        classes.extend(batch["class_name"])
        if "frames" in batch:
            keep = batch["mask"].cpu().numpy().ravel().astype(bool)
            fs.append(torch.sigmoid(out["frame_logits"]).float().cpu().numpy().ravel()[keep])
            ft.append(batch["frames"].numpy().ravel()[keep])

    s, y = np.concatenate(scores), np.concatenate(labels)
    result = {**summary(s, y), "per_class": per_class_breakdown(s, y, classes)}
    if fs:
        result.update(frame_metrics(np.concatenate(fs), np.concatenate(ft)))
    return result


def build_datasets(args, manifest, vcfg, acfg):
    ids: list[str] = []
    for name in args.subsets:
        ids.extend(load_subset(name, args.subset_dir))
    ids = sorted(set(ids))

    def make(subset_ids):
        return MultimodalFeatureDataset(
            subset_ids,
            manifest,
            vcfg,
            acfg,
            args.visual_root,
            args.audio_root,
            max_frames=args.max_frames,
            with_frames=True,  # P7-3: Phase 7 is the first phase that needs them
        )

    full = make(ids)
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for vid, split in zip(full.video_ids, full.splits, strict=True):
        by_split[split].append(vid)
    return full, {s: make(v) for s, v in by_split.items() if v}


def frame_balance(ds) -> dict:
    """Frame-level positive rate over a split — the number that sets `frame_pos_weight`.

    Computed from the manifest spans and the *cached* clip lengths rather than assumed,
    because the clip-level fake rate (73%) is nowhere near the frame-level one and guessing
    would set the weight an order of magnitude wrong.
    """
    total, positive = 0, 0.0
    for i in range(len(ds)):
        t = len(np.load(ds.audio_root / f"{ds.video_ids[i]}.npy", mmap_mode="r"))
        if ds.max_frames is not None:
            t = min(t, ds.max_frames)
        target = frame_targets(ds.fake_periods[i], t, float(ds.fps[i]))
        total += target.size
        positive += float(target.sum())
    rate = positive / total if total else 0.0
    return {
        "n_frames": int(total),
        "positive": int(positive),
        "positive_rate": rate,
        # The standard BCE re-balancing weight: negatives per positive.
        "pos_weight": (1 - rate) / rate if 0 < rate < 1 else None,
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Train the Phase 7 temporal baseline")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--backbone", choices=tuple(VISUAL_DIM), default="mobilenet_v2")
    ap.add_argument("--feature", choices=tuple(NATIVE_DIM), default="logmel")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--lstm-hidden", type=int, default=256, help="P7-1")
    ap.add_argument("--lstm-layers", type=int, default=2, help="P7-1")
    ap.add_argument("--lstm-dropout", type=float, default=0.1)
    ap.add_argument("--modality-dropout", type=float, default=0.2, help="P6-4")
    ap.add_argument("--aux-weight", type=float, default=0.3, help="P6-5")
    ap.add_argument(
        "--frame-weight",
        type=float,
        default=1.0,
        help="P7-3 weight on the per-frame loss; 0 trains the clip head alone, which is "
        "the ablation showing whether the frame objective costs clip AUC",
    )
    ap.add_argument("--no-defences", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overfit-only", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.no_defences:
        args.modality_dropout, args.aux_weight = 0.0, 0.0
    tag = "bilstm_nodef" if args.no_defences else "bilstm"
    if args.frame_weight == 0:
        tag += "_noframe"

    vcfg = VisualFeatureConfig(backbone=args.backbone)
    acfg = AudioPreprocessConfig(feature=args.feature)
    manifest = read_manifest(args.manifest).set_index("video_id")

    try:
        full, datasets = build_datasets(args, manifest, vcfg, acfg)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}")
        return 1

    def make_model(**over):
        kw = {
            "visual_dim": VISUAL_DIM[args.backbone],
            "feature": args.feature,
            "dropout": args.dropout,
            "modality_dropout": args.modality_dropout,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "lstm_dropout": args.lstm_dropout,
        }
        return TemporalFusionBaseline(**{**kw, **over})

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 7  Temporal (Baseline 4) -- {args.backbone} + "
        f"{args.feature} + BiLSTM\n{DIM}{'=' * 72}{RESET}"
    )
    print(f"  visual   : {config_hash(vcfg)}  dim={VISUAL_DIM[args.backbone]}")
    print(f"  audio    : {config_hash(acfg)}  dim={NATIVE_DIM[args.feature]}")
    print(
        f"  temporal : BiLSTM {args.lstm_layers}x{args.lstm_hidden} bidirectional  "
        f"{DIM}(dropout {args.lstm_dropout}){RESET}"
    )
    print(
        f"  defences : modality-dropout {args.modality_dropout}  aux-weight {args.aux_weight}"
        + (f"  {YELLOW}(DISABLED){RESET}" if args.no_defences else "")
    )
    print(f"  frame    : weight {args.frame_weight}")
    print(f"  params   : {make_model().n_parameters:,}")
    print(
        f"  available: {len(full)} clips  {DIM}(of {len(full) + len(full.missing)} requested; "
        f"{len(full.missing)} missing a stream){RESET}"
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

    # ---- frame-level balance -------------------------------------------------------
    fb = {s: frame_balance(ds) for s, ds in sorted(datasets.items())}
    print("\n  per-frame target balance (P7-3)")
    for s, b in fb.items():
        pw = f"{b['pos_weight']:.2f}" if b["pos_weight"] else "n/a"
        print(
            f"    {s:<6} {b['n_frames']:>8,} frames  {b['positive']:>7,} forged "
            f"({b['positive_rate']:.2%})  {DIM}pos_weight {pw}{RESET}"
        )
    frame_pos_weight = fb["train"]["pos_weight"]

    # ---- overfit-a-batch -----------------------------------------------------------
    print("\n  ⛔ overfit-a-batch (10 samples)")
    seed_everything(1337)
    tiny = collate_paired([datasets["train"][i] for i in range(min(10, len(datasets["train"])))])
    probe = Trainer(
        make_model(dropout=0.0, modality_dropout=0.0, lstm_dropout=0.0),
        TrainConfig(lr=1e-2, seed=1337, amp=False, frame_weight=args.frame_weight),
        args.device,
    )
    losses = probe.overfit_batch(tiny, steps=300)
    passed = losses[-1] < 0.01
    print(
        f"  [{GREEN if passed else RED}{'PASS' if passed else 'FAIL'}{RESET}] "
        f"loss {losses[0]:.4f} -> {losses[-1]:.6f}"
    )
    if not passed:
        print(f"  {RED}Baseline 4 cannot fit 10 samples -- fix gradients before training.{RESET}")
        return 1
    if args.overfit_only:
        return 0

    dev_labels = datasets["dev"].labels
    floors = baseline_zero(dev_labels)
    print(f"\n  Baseline 0 on dev ({len(dev_labels)} clips)")
    for name, m in floors.items():
        print(f"    {name:<16} AUC {m['auc']:.4f}  acc {m['accuracy']:.4f}")

    train_loader = paired_loader(datasets["train"], args.batch_size, shuffle=True, seed=0)
    dev_loader = paired_loader(datasets["dev"], args.batch_size, shuffle=False, seed=0)
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
            aux_weight=args.aux_weight,
            frame_weight=args.frame_weight,
            frame_pos_weight=frame_pos_weight,
        )
        run_dir = Path(args.out_root) / f"phase7_{tag}" / f"seed{seed}"
        model = make_model()
        runner = Trainer(model, tcfg, args.device, run_dir)

        t0 = time.time()
        runner.fit(train_loader, dev_loader)
        elapsed = time.time() - t0

        runner.resume(run_dir / "best.pt")
        dev_metrics = evaluate_split(runner, dev_loader)
        no_visual = ablate(runner, dev_loader, drop_visual=True)
        no_audio = ablate(runner, dev_loader, drop_audio=True)
        runner.write_history(run_dir / "history.json")

        record = {
            "seed": 1337 + seed,
            "temporal": "bilstm",
            # ⛔ P7-6's own claim, recorded rather than asserted in prose: this arm has no
            # self- or cross-attention, so `D - C` cannot be attention's doing.
            "attention": False,
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "backbone": args.backbone,
            "feature": args.feature,
            "defences": not args.no_defences,
            "modality_dropout": args.modality_dropout,
            "aux_weight": args.aux_weight,
            "frame_weight": args.frame_weight,
            "frame_pos_weight": frame_pos_weight,
            "git_sha": sha,
            "n_params": model.n_parameters,
            "epochs_run": len(runner.history),
            "best_epoch": runner.best_epoch,
            "wall_clock_s": round(elapsed, 2),
            "n_clips_train": len(datasets["train"]),
            "dev": dev_metrics,
            "dev_drop_visual": no_visual,
            "dev_drop_audio": no_audio,
            "train_config": asdict(tcfg),
        }
        (run_dir / "result.json").write_text(
            json.dumps(record, indent=2, default=float) + "\n", encoding="utf-8"
        )
        runs.append(record)
        fap = dev_metrics.get("frame_ap", float("nan"))
        print(
            f"    best epoch {runner.best_epoch}  dev AUC {dev_metrics['auc']:.4f}  "
            f"frame AP {fap:.4f}  "
            f"{DIM}(-visual {no_visual['auc']:.4f}, -audio {no_audio['auc']:.4f}){RESET}  "
            f"{elapsed:.1f}s"
        )
        if not args.no_mlflow:
            _log_mlflow(record, run_dir, tag)

    # ---- Experiment D ---------------------------------------------------------------
    aucs = np.array([r["dev"]["auc"] for r in runs])
    faps = np.array([r["dev"].get("frame_ap", np.nan) for r in runs], dtype=float)
    dv = np.array([r["dev_drop_visual"]["auc"] for r in runs])
    da = np.array([r["dev_drop_audio"]["auc"] for r in runs])

    print(f"\n{DIM}{'-' * 72}{RESET}")
    print(f"  temporal / {tag} over {len(runs)} seed(s)")
    print(f"    dev AUC   {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"    frame AP  {np.nanmean(faps):.4f} +/- {np.nanstd(faps):.4f}   {DIM}(P7-4){RESET}")

    drop_v = aucs.mean() - dv.mean()
    drop_a = aucs.mean() - da.mean()
    collapsed = drop_v < 0.01
    print("\n  ⛔ section 5.6 modality ablation (dev AUC when a stream is zeroed)")
    print(f"    without visual: {dv.mean():.4f}   {DIM}drop {drop_v:+.4f}{RESET}")
    print(f"    without audio : {da.mean():.4f}   {DIM}drop {drop_a:+.4f}{RESET}")
    print(
        f"    {(RED if collapsed else GREEN)}"
        f"{'COLLAPSED -- video is unused' if collapsed else 'both modalities contribute'}{RESET}"
    )

    print("\n  four-class breakdown (best seed)")
    best = max(runs, key=lambda r: r["dev"]["auc"])
    for cls, m in best["dev"]["per_class"].items():
        if "auc" in m:
            print(f"    {cls:<12} n={m['n_fake']:>4}  AUC {m['auc']:.4f}  recall {m['recall']:.3f}")
        else:
            print(f"    {cls:<12} n={m['n']:>4}  specificity {m['specificity']:.3f}")

    out = Path(f"reports/phase7_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "temporal": "bilstm",
                "attention": False,
                "lstm_hidden": args.lstm_hidden,
                "lstm_layers": args.lstm_layers,
                "defences": not args.no_defences,
                "backbone": args.backbone,
                "feature": args.feature,
                "frame_weight": args.frame_weight,
                "frame_pos_weight": frame_pos_weight,
                "git_sha": sha,
                "n_clips": len(full),
                "n_params": runs[0]["n_params"],
                "splits": {s: ds.label_balance for s, ds in datasets.items()},
                "frame_balance": fb,
                "baseline_zero": floors,
                "dev_auc_mean": float(aucs.mean()),
                "dev_auc_std": float(aucs.std()),
                "dev_frame_ap_mean": float(np.nanmean(faps)),
                "dev_frame_ap_std": float(np.nanstd(faps)),
                "drop_visual_auc_mean": float(dv.mean()),
                "drop_audio_auc_mean": float(da.mean()),
                "drop_visual_delta": float(drop_v),
                "drop_audio_delta": float(drop_a),
                "collapsed": bool(collapsed),
                "runs": runs,
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  wrote {out}")
    print(f"\n{GREEN}TRAINING DONE{RESET}  {DIM}run scripts/25_experiment_d.py for the gate{RESET}")
    return 0


def _log_mlflow(record: dict, run_dir: Path, tag: str) -> None:
    """Never fatal -- tracking must not break training (PF-14)."""
    try:
        import mlflow

        store = Path("experiments/mlflow.db").resolve()
        store.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{store.as_posix()}")
        mlflow.set_experiment("phase7_temporal")
        with mlflow.start_run(run_name=f"temporal_{tag}_seed{record['seed']}"):
            mlflow.log_params(
                {
                    "temporal": record["temporal"],
                    "lstm_hidden": record["lstm_hidden"],
                    "lstm_layers": record["lstm_layers"],
                    "defences": record["defences"],
                    "backbone": record["backbone"],
                    "feature": record["feature"],
                    "seed": record["seed"],
                    "git_sha": record["git_sha"],
                    "n_params": record["n_params"],
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
                    "dev_drop_visual_auc": float(record["dev_drop_visual"]["auc"]),
                    "dev_drop_audio_auc": float(record["dev_drop_audio"]["auc"]),
                    "wall_clock_s": record["wall_clock_s"],
                    "best_epoch": record["best_epoch"],
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
