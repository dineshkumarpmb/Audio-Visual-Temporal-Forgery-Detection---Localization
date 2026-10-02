"""Phase 10: train the localization arms and save their dev predictions (P10-3 .. P10-5).

    python scripts/34_train_localization.py --arms bce full focal --seeds 3
    python scripts/34_train_localization.py --arms full_l1 full_l4 --seeds 1   # λ tuning

**Training writes predictions, not metrics.** Each finished seed saves its dev per-frame
probabilities (and boundary probabilities, when the arm has the head) to
`dev_predictions.npz`. Post-processing, its grid search and every localization number are
then computed by `scripts/35_experiment_g.py` from those files alone -- section L's
"evaluation is a pure function of saved predictions", which means a metric bug can be
fixed, or a metric added, without retraining anything.

Every arm is Phase 9's F0 -- the section 5.1 model with no augmentation, PF-25's
recommendation -- plus one change to the objective:

| arm | frame loss | λ_loc | boundary head, λ_bnd | answers |
|---|---|---|---|---|
| `bce` | weighted BCE | 1.0 | -- | the control: F0's objective, selected on frame AP |
| `focal` | focal (P10-3) | 2.0 | -- | does the loss swap alone help? |
| `full` | focal | 2.0 | ✓ 0.5 | section 5.4's objective at its starting λ (P10-4/5) |
| `full_l1` / `full_l4` | focal | 1.0 / 4.0 | ✓ 0.5 | λ_loc tuning on dev (P10-5) |
| `full_b1` | focal | 2.0 | ✓ 1.0 | λ_bnd tuning on dev |

λ_cls = 1.0 and λ_sync = 0.3 throughout (section 5.4's starting values).

**⛔ Model selection is on dev frame AP, for every arm including the control.** Phases 4-9
picked `best.pt` by clip AUC; Phase 10's headline is localization, and frame AP is the
threshold-free localization signal available per epoch -- it cannot be the tuned
post-processing's own output, so selecting on it is not circular. The control arm is
retrained rather than borrowed from Phase 9 so that it differs from the others in its loss
alone, not in its loss *and* its selection rule.

**Test is never touched** (section 3.5 RULE 4).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
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
from src.data.dataset import collate_paired  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.models.attention_model import AttentionFusionModel  # noqa: E402
from src.models.backbones.audio import NATIVE_DIM  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.training.reporting import git_sha  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402
from src.utils.console import init_console  # noqa: E402


def _load_phase8():
    """Phase 8's helpers define frame AP and the modality ablation; share, never re-derive."""
    path = REPO_ROOT / "scripts" / "27_train_attention.py"
    spec = importlib.util.spec_from_file_location("phase8_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase8_sweep"] = module
    spec.loader.exec_module(module)
    return module


P8 = _load_phase8()
GREEN, RED, YELLOW, DIM, RESET = P8.GREEN, P8.RED, P8.YELLOW, P8.DIM, P8.RESET
VISUAL_DIM = P8.VISUAL_DIM

# (frame_loss, λ_loc, λ_bnd). Order matters: the control first, then the headline.
ARMS = {
    "bce": ("bce", 1.0, 0.0),
    "full": ("focal", 2.0, 0.5),
    "focal": ("focal", 2.0, 0.0),
    "full_l1": ("focal", 1.0, 0.5),
    "full_l4": ("focal", 4.0, 0.5),
    "full_b1": ("focal", 2.0, 1.0),
}


@torch.no_grad()
def collect_predictions(trainer: Trainer, loader, *, drop_visual: bool = False) -> dict:
    """Per-clip frame (and boundary) probabilities over valid frames, flattened + offsets."""
    trainer.model.eval()
    ids, labels, clip, frames, bounds, lengths = [], [], [], [], [], []
    for batch in loader:
        mask = batch["mask"].to(trainer.device)
        out = trainer.model(
            batch["features"].to(trainer.device),
            batch["audio"].to(trainer.device),
            mask,
            batch["face"].to(trainer.device),
            drop_visual=drop_visual,
        )
        fp = torch.sigmoid(out["frame_logits"]).float().cpu().numpy()
        bp = (
            torch.sigmoid(out["boundary_logits"]).float().cpu().numpy()
            if "boundary_logits" in out
            else None
        )
        cp = torch.sigmoid(out["logit"]).float().cpu().numpy()
        m = batch["mask"].numpy().astype(bool)
        for i, vid in enumerate(batch["video_id"]):
            t = int(m[i].sum())
            # Valid frames are a prefix: collate_paired pads at the end only.
            ids.append(vid)
            labels.append(int(batch["label"][i]))
            clip.append(float(cp[i]))
            frames.append(fp[i, :t])
            if bp is not None:
                bounds.append(bp[i, :t])
            lengths.append(t)
    return {
        "video_ids": np.array(ids),
        "labels": np.array(labels, dtype=np.int8),
        "clip_scores": np.array(clip, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "frame_scores": np.concatenate(frames).astype(np.float32),
        "boundary_scores": (
            np.concatenate(bounds).astype(np.float32) if bounds else np.zeros((0, 2), np.float32)
        ),
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Train the Phase 10 localization arms")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--backbone", choices=tuple(VISUAL_DIM), default="mobilenet_v2")
    ap.add_argument("--feature", choices=tuple(NATIVE_DIM), default="logmel")
    ap.add_argument("--arms", nargs="+", default=["bce", "full", "focal"], choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--attn-dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--cls-weight", type=float, default=1.0, help="section 5.4 λ_cls")
    ap.add_argument("--sync-weight", type=float, default=0.3, help="section 5.4 λ_sync")
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vcfg = VisualFeatureConfig(backbone=args.backbone)
    acfg = AudioPreprocessConfig(feature=args.feature)
    manifest = read_manifest(args.manifest).set_index("video_id")
    try:
        full, datasets = P8.build_datasets(args, manifest, vcfg, acfg)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}")
        return 1
    if "train" not in datasets or "dev" not in datasets:
        print(f"\n{RED}need both a train and a dev split{RESET}")
        return 1

    def make_model(arm: str, **over):
        _, _, bnd = ARMS[arm]
        kw = {
            "visual_dim": VISUAL_DIM[args.backbone],
            "feature": args.feature,
            "dropout": args.dropout,
            "attn_dropout": args.attn_dropout,
            "modality_dropout": args.modality_dropout,
            "cross_attention": True,
            "temporal": "transformer",
            "positional": True,
            "boundary": bnd > 0,
        }
        return AttentionFusionModel(**{**kw, **over})

    def make_cfg(arm: str, seed: int, frame_pw, pos_weight, **over) -> TrainConfig:
        loss, lam_loc, lam_bnd = ARMS[arm]
        kw = {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "pos_weight": pos_weight,
            "patience": args.patience,
            "seed": 1337 + seed,
            "aux_weight": args.aux_weight,
            "cls_weight": args.cls_weight,
            "frame_weight": lam_loc,
            "frame_loss": loss,
            "frame_pos_weight": frame_pw if loss == "bce" else None,
            "sync_weight": args.sync_weight,
            "boundary_weight": lam_bnd,
            "select_metric": "frame_ap",
        }
        return TrainConfig(**{**kw, **over})

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 10  Localization (Experiment G) -- {args.backbone} + "
        f"{args.feature}\n{DIM}{'=' * 72}{RESET}"
    )
    print(f"  visual   : {config_hash(vcfg)}  audio: {config_hash(acfg)}")
    print("  base     : Phase 9's F0 (the section 5.1 model, no augmentation -- PF-25)")
    print("  selection: dev frame AP, every arm")
    for arm in args.arms:
        loss, lam_loc, lam_bnd = ARMS[arm]
        print(f"    {arm:<8} frame={loss:<5} λ_loc={lam_loc}  λ_bnd={lam_bnd}")
    for split, ds in datasets.items():
        b = ds.label_balance
        print(
            f"    {split:<6} {b['n']:>5} clips  {b['positive']:>4} fake / {b['negative']:>4} real"
        )

    fpw = P8.frame_pos_weight_for(datasets["train"])
    pos_rate = datasets["train"].label_balance["positive_rate"]
    pos_weight = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else None

    # ⛔ Overfit-a-batch on the arm with the most new code: focal + the boundary head. A
    # detached boundary head would train "fine" and simply never learn anything.
    probe_arm = next((a for a in args.arms if ARMS[a][2] > 0), None)
    if probe_arm and not args.skip_overfit:
        print(f"\n  ⛔ overfit-a-batch (10 samples, arm '{probe_arm}')")
        seed_everything(1337)
        tiny = collate_paired(
            [datasets["train"][i] for i in range(min(10, len(datasets["train"])))]
        )
        probe = Trainer(
            make_model(probe_arm, dropout=0.0, attn_dropout=0.0, modality_dropout=0.0),
            make_cfg(probe_arm, 0, fpw, pos_weight, sync_weight=0.0, aux_weight=0.0),
            args.device,
        )
        losses = probe.overfit_batch(tiny, steps=400, verbose=False)
        passed = losses[-1] < 0.1 * losses[0]
        print(
            f"  [{GREEN if passed else RED}{'PASS' if passed else 'FAIL'}{RESET}] "
            f"loss {losses[0]:.4f} -> {losses[-1]:.4f}  {DIM}(must fall below 10%){RESET}"
        )
        del probe
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        if not passed:
            return 1

    dev_loader = P8.paired_loader(datasets["dev"], args.batch_size, shuffle=False, seed=0)
    sha = git_sha()

    for arm in args.arms:
        loss, lam_loc, lam_bnd = ARMS[arm]
        print(
            f"\n{DIM}{'=' * 72}{RESET}\n  arm '{arm}'  frame={loss} λ_loc={lam_loc} λ_bnd={lam_bnd}"
        )
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            run_dir = Path(args.out_root) / f"phase10_{arm}" / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            done = run_dir / "result.json"
            if done.exists() and not args.restart:
                r = json.loads(done.read_text(encoding="utf-8"))
                print(
                    f"\n  seed {seed}  {GREEN}already done{RESET} "
                    f"{DIM}(frame AP {r['dev']['frame_ap']:.4f}){RESET}"
                )
                continue

            print(f"\n  seed {seed}")
            seed_everything(1337 + seed)
            train_loader = P8.paired_loader(
                datasets["train"], args.batch_size, shuffle=True, seed=seed
            )
            tcfg = make_cfg(arm, seed, fpw, pos_weight)
            model = make_model(arm)
            runner = Trainer(model, tcfg, args.device, run_dir)
            resumed = run_dir / "last.pt"
            if resumed.exists() and not args.restart:
                start = runner.resume(resumed)
                print(f"    {YELLOW}resuming from epoch {start}{RESET}")

            t0 = time.time()
            runner.fit(train_loader, dev_loader)
            elapsed = time.time() - t0
            runner.resume(run_dir / "best.pt")
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

            dev_metrics = P8.diagnostics(runner, dev_loader)
            no_visual = P8.diagnostics(runner, dev_loader, drop_visual=True)
            no_audio = P8.diagnostics(runner, dev_loader, drop_audio=True)
            preds = collect_predictions(runner, dev_loader)
            preds_nv = collect_predictions(runner, dev_loader, drop_visual=True)
            np.savez_compressed(run_dir / "dev_predictions.npz", **preds)
            np.savez_compressed(
                run_dir / "dev_predictions_drop_visual.npz",
                **{k: v for k, v in preds_nv.items() if k != "boundary_scores"},
            )

            row = {
                "seed": 1337 + seed,
                "arm": arm,
                "frame_loss": loss,
                "lambda_cls": args.cls_weight,
                "lambda_loc": lam_loc,
                "lambda_sync": args.sync_weight,
                "lambda_bnd": lam_bnd,
                "boundary_head": lam_bnd > 0,
                "select_metric": "frame_ap",
                "backbone": args.backbone,
                "feature": args.feature,
                "augmentation": False,
                "git_sha": sha,
                "n_params": model.n_parameters,
                "epochs_run": len(runner.history),
                "best_epoch": runner.best_epoch,
                "wall_clock_s": round(elapsed, 2),
                "n_clips_train": len(datasets["train"]),
                "n_clips_dev": len(datasets["dev"]),
                "dev": dev_metrics,
                "dev_drop_visual": no_visual,
                "dev_drop_audio": no_audio,
                "train_config": asdict(tcfg),
            }
            done.write_text(json.dumps(row, indent=2, default=float) + "\n", encoding="utf-8")
            runner.write_history(run_dir / "history.json")
            print(
                f"    best epoch {runner.best_epoch}  frame AP {dev_metrics['frame_ap']:.4f}  "
                f"AUC {dev_metrics['auc']:.4f}  {DIM}(-visual frame AP "
                f"{no_visual['frame_ap']:.4f}){RESET}  {elapsed:.0f}s"
            )
            del model, runner
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n{GREEN}TRAINING DONE{RESET}  {DIM}run scripts/35_experiment_g.py{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
