"""Phase 8 Stage-B: train the section 5.1 model and run Experiment E (P8-1 .. P8-9).

    python scripts/27_train_attention.py --overfit-only     # gradient check first
    python scripts/27_train_attention.py --seeds 3          # all four arms
    python scripts/27_train_attention.py --arms full        # just the headline

**Experiment E changes two components at once, so it is run as four arms, not one.**
Swapping concat→cross-attention *and* BiLSTM→Transformer together would make `E − D`
uninterpretable — precisely the failure P7-6 exists to prevent. Every arm below shares the
Phase 6 stream encoders, the same 786 clips, the same dev split and the same sync head, so
each pair differs by exactly one component:

| arm | cross-attn | temporal | isolates |
|---|---|---|---|
| `d_sync` | ✗ concat | BiLSTM | the **sync loss** (vs Phase 7's D, which has none) |
| `cross` | ✓ | BiLSTM | **P8-3** cross-attention, against `d_sync` |
| `self` | ✗ concat | Transformer | **P8-1** self-attention, against `d_sync` |
| `full` | ✓ | Transformer | the section 5.1 model — both, against each single-component arm |

**⛔ The collapse question is the real one here.** PF-21 measured the section 5.6 collapse
surviving the BiLSTM: zeroing video costs 0.0020 AUC and barely moves frame AP. Modality
dropout backfired (PF-20) and the auxiliary heads did not save it. Cross-attention is the
last structurally different candidate — its V→A queries come *from* video, so a dead visual
stream cannot route attention. The `cross` arm is what tests that, and the modality ablation
is reported on every arm.

**P8-9's "attention elevates on forged spans" is measured, not eyeballed.** `attention_lift`
is the ratio of mean attention-pool weight on forged frames to mean weight on genuine frames,
within fake clips only. >1 means the pooling operator concentrates where the forgery is. The
attention *maps* (P8-8) are a figure; this is the number that can enter a table.

**Test is never touched** (section 3.5 RULE 4).
"""

from __future__ import annotations

import argparse
import gc
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
from src.evaluation.metrics import frame_metrics, per_class_breakdown, summary  # noqa: E402
from src.localization.targets import frame_targets  # noqa: E402
from src.models.attention_model import AttentionFusionModel  # noqa: E402
from src.models.backbones.audio import NATIVE_DIM  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.training.reporting import baseline_zero, git_sha  # noqa: E402
from src.training.trainer import TrainConfig, Trainer, evaluate_split  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

VISUAL_DIM = {"resnet18": 512, "mobilenet_v2": 1280}

# (cross_attention, temporal). Order matters: the headline arm runs first so a killed sweep
# still leaves the number Experiment E is actually about.
ARMS = {
    "full": (True, "transformer"),
    "cross": (True, "lstm"),
    "self": (False, "transformer"),
    "d_sync": (False, "lstm"),
}


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
def diagnostics(trainer, loader, *, drop_visual=False, drop_audio=False) -> dict:
    """Modality ablation, frame AP, sync separation and P8-9's attention lift, in one pass."""
    trainer.model.eval()
    scores, labels, classes = [], [], []
    fs, ft = [], []
    sync_real, sync_fake = [], []
    lift_num, lift_den = [], []
    entropies = []

    for batch in loader:
        mask = batch["mask"].to(trainer.device)
        out = trainer.model(
            batch["features"].to(trainer.device),
            batch["audio"].to(trainer.device),
            mask,
            batch["face"].to(trainer.device),
            drop_visual=drop_visual,
            drop_audio=drop_audio,
            compute_entropy=True,
        )
        y = batch["label"].numpy()
        scores.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
        labels.append(y)
        classes.extend(batch["class_name"])

        m = batch["mask"].numpy().astype(bool)
        if "frames" in batch:
            keep = m.ravel()
            fs.append(torch.sigmoid(out["frame_logits"]).float().cpu().numpy().ravel()[keep])
            ft.append(batch["frames"].numpy().ravel()[keep])

        if "head_entropy" in out:
            entropies.append(out["head_entropy"].float().cpu().numpy())

        # Sync separation: real clips are in-sync by construction, so a working head should
        # score them above fakes. This is the trained analogue of the ⛔ P8-6 desync test.
        clip_sync = trainer.model.sync_head.clip_score(out["sync"], mask).float().cpu().numpy()
        sync_real.append(clip_sync[y < 0.5])
        sync_fake.append(clip_sync[y > 0.5])

        # ⛔ P8-9: does attention concentrate on forged frames? Fake clips only -- a real
        # clip has no forged span, so it can neither support nor refute the claim.
        if "frames" in batch:
            attn = out["attention"].float().cpu().numpy()
            tgt = batch["frames"].numpy()
            for i in range(len(y)):
                if y[i] < 0.5:
                    continue
                valid, forged = m[i], tgt[i] > 0.5
                on = attn[i][valid & forged]
                off = attn[i][valid & ~forged]
                if on.size and off.size:
                    lift_num.append(on.mean())
                    lift_den.append(off.mean())

    s, yy = np.concatenate(scores), np.concatenate(labels)
    result = {**summary(s, yy), "per_class": per_class_breakdown(s, yy, classes)}
    if fs:
        result.update(frame_metrics(np.concatenate(fs), np.concatenate(ft)))

    real = np.concatenate(sync_real) if sync_real else np.array([])
    fake = np.concatenate(sync_fake) if sync_fake else np.array([])
    result["sync_real_mean"] = float(real.mean()) if real.size else float("nan")
    result["sync_fake_mean"] = float(fake.mean()) if fake.size else float("nan")
    result["sync_separation"] = result["sync_real_mean"] - result["sync_fake_mean"]

    if lift_num:
        num, den = np.array(lift_num), np.array(lift_den)
        ratio = num / np.maximum(den, 1e-12)
        result["attention_lift"] = float(ratio.mean())
        result["attention_lift_median"] = float(np.median(ratio))
        result["attention_lift_n_clips"] = int(len(ratio))
    if entropies:
        result["head_entropy_mean"] = float(np.concatenate(entropies).mean())
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
            with_frames=True,
        )

    full = make(ids)
    by_split: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for vid, split in zip(full.video_ids, full.splits, strict=True):
        by_split[split].append(vid)
    return full, {s: make(v) for s, v in by_split.items() if v}


def frame_pos_weight_for(ds) -> float | None:
    total, positive = 0, 0.0
    for i in range(len(ds)):
        t = len(np.load(ds.audio_root / f"{ds.video_ids[i]}.npy", mmap_mode="r"))
        if ds.max_frames is not None:
            t = min(t, ds.max_frames)
        target = frame_targets(ds.fake_periods[i], t, float(ds.fps[i]))
        total += target.size
        positive += float(target.sum())
    rate = positive / total if total else 0.0
    return (1 - rate) / rate if 0 < rate < 1 else None


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Train the Phase 8 attention model")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--backbone", choices=tuple(VISUAL_DIM), default="mobilenet_v2")
    ap.add_argument("--feature", choices=tuple(NATIVE_DIM), default="logmel")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16, help="section H: batch 16 on 4 GB")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--attn-dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=4, help="P8-1")
    ap.add_argument("--n-heads", type=int, default=4, help="P8-1")
    ap.add_argument("--cross-layers", type=int, default=2, help="P8-3")
    ap.add_argument("--no-positional", action="store_true", help="⛔ P8-2 ablation")
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--frame-weight", type=float, default=1.0)
    ap.add_argument("--sync-weight", type=float, default=0.3, help="P8-5, section 5.4 λ_sync")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overfit-only", action="store_true")
    ap.add_argument(
        "--skip-overfit",
        action="store_true",
        help="skip the P4-7 gradient check. Only for resuming an interrupted sweep, where it "
        "has already passed for this arm and costs ~90s of every retry",
    )
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--out-root", default="experiments")
    # Redirecting --out-root alone is not enough to make a smoke run harmless: the
    # per-arm report path was hardcoded, so a 2-epoch trial silently overwrote the real
    # reports/phase8_full.json the gate reads. Both must move together.
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument(
        "--restart",
        action="store_true",
        help="retrain arms/seeds that already have results. Default is to resume: this "
        "sweep is 12 runs on a 6 GB machine and the first attempt was killed by the OS "
        "for low memory mid-arm (R1/X-4), so finished work must survive a restart",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vcfg = VisualFeatureConfig(backbone=args.backbone)
    acfg = AudioPreprocessConfig(feature=args.feature)
    manifest = read_manifest(args.manifest).set_index("video_id")

    try:
        full, datasets = build_datasets(args, manifest, vcfg, acfg)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    if "train" not in datasets or "dev" not in datasets:
        print(f"\n{RED}need both a train and a dev split{RESET}")
        return 1

    def make_model(arm: str, **over):
        cross, temporal = ARMS[arm]
        kw = {
            "visual_dim": VISUAL_DIM[args.backbone],
            "feature": args.feature,
            "dropout": args.dropout,
            "attn_dropout": args.attn_dropout,
            "modality_dropout": args.modality_dropout,
            "cross_attention": cross,
            "temporal": temporal,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "cross_layers": args.cross_layers,
            "positional": not args.no_positional,
        }
        return AttentionFusionModel(**{**kw, **over})

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 8  Attention (Experiment E) -- {args.backbone} + "
        f"{args.feature}\n{DIM}{'=' * 72}{RESET}"
    )
    print(f"  visual   : {config_hash(vcfg)}  audio: {config_hash(acfg)}")
    print(
        f"  encoder  : {args.n_layers}L x {args.n_heads}H transformer, cross {args.cross_layers}L"
    )
    print(f"  losses   : frame {args.frame_weight}  aux {args.aux_weight}  sync {args.sync_weight}")
    print(f"  arms     : {', '.join(args.arms)}")
    for arm in args.arms:
        cross, temporal = ARMS[arm]
        print(
            f"    {arm:<8} cross={str(cross):<5} temporal={temporal:<12} "
            f"{make_model(arm).n_parameters:>10,} params"
        )
    print(f"  available: {len(full)} clips")
    for s, ds in sorted(datasets.items()):
        b = ds.label_balance
        print(f"    {s:<6} {b['n']:>5} clips  {b['positive']:>4} fake / {b['negative']:>4} real")

    fpw = frame_pos_weight_for(datasets["train"])
    print(f"  frame pos_weight: {fpw:.2f}")

    # ---- overfit-a-batch, on the headline arm --------------------------------------
    if args.skip_overfit and not args.overfit_only:
        print(f"\n  {YELLOW}skipping overfit-a-batch (--skip-overfit){RESET}")
    else:
        print(f"\n  ⛔ overfit-a-batch (10 samples, arm '{args.arms[0]}')")
        seed_everything(1337)
        tiny = collate_paired(
            [datasets["train"][i] for i in range(min(10, len(datasets["train"])))]
        )
        probe = Trainer(
            make_model(args.arms[0], dropout=0.0, attn_dropout=0.0, modality_dropout=0.0),
            TrainConfig(lr=1e-3, seed=1337, amp=False, frame_weight=args.frame_weight),
            args.device,
        )
        losses = probe.overfit_batch(tiny, steps=400)
        passed = losses[-1] < 0.05
        print(
            f"  [{GREEN if passed else RED}{'PASS' if passed else 'FAIL'}{RESET}] "
            f"loss {losses[0]:.4f} -> {losses[-1]:.6f}"
        )
        if not passed:
            print(f"  {RED}Cannot fit 10 samples -- fix gradients before training.{RESET}")
            return 1
        # The probe holds a whole second model plus its optimiser state. On a machine with
        # ~2 GB free that is not a rounding error -- free it before real training starts.
        del probe
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if args.overfit_only:
        return 0

    floors = baseline_zero(datasets["dev"].labels)
    train_loader = paired_loader(datasets["train"], args.batch_size, shuffle=True, seed=0)
    dev_loader = paired_loader(datasets["dev"], args.batch_size, shuffle=False, seed=0)
    pos_rate = datasets["train"].label_balance["positive_rate"]
    pos_weight = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else None
    sha = git_sha()

    for arm in args.arms:
        cross, temporal = ARMS[arm]
        arm_report = Path(args.report_dir) / f"phase8_{arm}.json"
        if arm_report.exists() and not args.restart:
            print(
                f"\n{DIM}{'=' * 72}{RESET}\n  arm '{arm}': {GREEN}already complete{RESET} "
                f"{DIM}({arm_report}; --restart to redo){RESET}"
            )
            continue
        print(f"\n{DIM}{'=' * 72}{RESET}\n  arm '{arm}'  cross={cross}  temporal={temporal}")
        runs = []
        for seed in range(args.seeds):
            run_dir = Path(args.out_root) / f"phase8_{arm}" / f"seed{seed}"
            done = run_dir / "result.json"
            # ⛔ X-4 / R13: this sweep is 12 runs on a 6 GB machine and the first attempt was
            # killed by the OS mid-arm. Finished seeds are reloaded rather than retrained, so
            # an interruption costs the run in flight and nothing else.
            if done.exists() and not args.restart:
                runs.append(json.loads(done.read_text(encoding="utf-8")))
                print(
                    f"\n  seed {seed}  {GREEN}already done{RESET} "
                    f"{DIM}(AUC {runs[-1]['dev']['auc']:.4f}){RESET}"
                )
                continue
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
                frame_pos_weight=fpw,
                sync_weight=args.sync_weight,
            )
            model = make_model(arm)
            runner = Trainer(model, tcfg, args.device, run_dir)

            # ⛔ X-4/R13, mid-seed. The Trainer has checkpointed every epoch since Phase 4;
            # this is the caller that finally uses it. A seed here takes 13-18 minutes and
            # this 6 GB machine kills the process within a few, so seed-level resume alone
            # never converged -- every attempt restarted epoch 0 and died before finishing.
            # Resuming from `last.pt` makes each attempt cost at most one epoch.
            resumed = run_dir / "last.pt"
            if resumed.exists() and not args.restart:
                start = runner.resume(resumed)
                print(
                    f"    {GREEN}resuming from epoch {start}{RESET} "
                    f"{DIM}(best so far {runner.best_auc:.4f} @ {runner.best_epoch}){RESET}"
                )

            t0 = time.time()
            runner.fit(train_loader, dev_loader)
            elapsed = time.time() - t0

            runner.resume(run_dir / "best.pt")
            dev_metrics = evaluate_split(runner, dev_loader)
            diag = diagnostics(runner, dev_loader)
            dev_metrics.update({k: v for k, v in diag.items() if k not in dev_metrics})
            no_visual = diagnostics(runner, dev_loader, drop_visual=True)
            no_audio = diagnostics(runner, dev_loader, drop_audio=True)
            runner.write_history(run_dir / "history.json")

            record = {
                "seed": 1337 + seed,
                "arm": arm,
                "cross_attention": cross,
                "temporal": temporal,
                "attention": cross or temporal == "transformer",
                "positional": not args.no_positional,
                "backbone": args.backbone,
                "feature": args.feature,
                "defences": True,
                "sync_weight": args.sync_weight,
                "frame_weight": args.frame_weight,
                "aux_weight": args.aux_weight,
                "modality_dropout": args.modality_dropout,
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
            best_epoch = runner.best_epoch

            # R1: 6 GB of host RAM across 12 runs. Without this the optimiser state,
            # checkpoint tensors and CUDA cache of every finished run stay resident and the
            # OS eventually kills the sweep -- which is exactly what happened on attempt 1.
            del model, runner
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

            print(
                f"    best epoch {best_epoch}  AUC {dev_metrics['auc']:.4f}  "
                f"frame AP {dev_metrics.get('frame_ap', float('nan')):.4f}  "
                f"lift {dev_metrics.get('attention_lift', float('nan')):.2f}x  "
                f"{DIM}(-visual {no_visual['auc']:.4f}){RESET}  {elapsed:.0f}s"
            )

        aucs = np.array([r["dev"]["auc"] for r in runs])
        faps = np.array([r["dev"].get("frame_ap", np.nan) for r in runs], dtype=float)
        lifts = np.array([r["dev"].get("attention_lift", np.nan) for r in runs], dtype=float)
        dv = np.array([r["dev_drop_visual"]["auc"] for r in runs])
        da = np.array([r["dev_drop_audio"]["auc"] for r in runs])
        seps = np.array([r["dev"].get("sync_separation", np.nan) for r in runs], dtype=float)

        out = Path(args.report_dir) / f"phase8_{arm}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "cross_attention": cross,
                    "temporal": temporal,
                    "attention": cross or temporal == "transformer",
                    "positional": not args.no_positional,
                    "backbone": args.backbone,
                    "feature": args.feature,
                    "defences": True,
                    "sync_weight": args.sync_weight,
                    "git_sha": sha,
                    "n_clips": len(full),
                    "n_params": runs[0]["n_params"],
                    "splits": {s: ds.label_balance for s, ds in datasets.items()},
                    "baseline_zero": floors,
                    "dev_auc_mean": float(aucs.mean()),
                    "dev_auc_std": float(aucs.std()),
                    "dev_frame_ap_mean": float(np.nanmean(faps)),
                    "dev_frame_ap_std": float(np.nanstd(faps)),
                    "attention_lift_mean": float(np.nanmean(lifts)),
                    "attention_lift_std": float(np.nanstd(lifts)),
                    "sync_separation_mean": float(np.nanmean(seps)),
                    "drop_visual_auc_mean": float(dv.mean()),
                    "drop_audio_auc_mean": float(da.mean()),
                    "drop_visual_delta": float(aucs.mean() - dv.mean()),
                    "drop_audio_delta": float(aucs.mean() - da.mean()),
                    "collapsed": bool(aucs.mean() - dv.mean() < 0.01),
                    "runs": runs,
                },
                indent=2,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"\n  {arm}: AUC {aucs.mean():.4f} +/- {aucs.std():.4f}  "
            f"frame AP {np.nanmean(faps):.4f}  lift {np.nanmean(lifts):.2f}x  "
            f"drop-visual {aucs.mean() - dv.mean():+.4f}"
        )
        print(f"  wrote {out}")

    print(f"\n{GREEN}TRAINING DONE{RESET}  {DIM}run scripts/28_experiment_e.py for the gate{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
