"""Phase 9: train the three augmentation arms (P9-1 .. P9-5; plan section I).

    python scripts/30_train_augmented.py --arms F0            # the control, first
    python scripts/30_train_augmented.py --seeds 3            # all three arms

⛔ **Three arms, because two would be uninterpretable.** Section I's design warning is
explicit: comparing "GAN augmentation" against "no augmentation" cannot separate the GAN
from augmentation in general, and most published ablations get this wrong.

| arm | contents | answers |
|---|---|---|
| `F0` | no augmentation | the floor -- and the section 5.1 model unchanged |
| `F1` | splicing + classical (P9-1, P9-2) | does augmentation help at all? |
| `F2` | F1 + GAN samples (P9-3) | **F2 vs F1 -- the only valid GAN claim** |

**F2 depends on F1's checkpoint, by design.** The generator is trained on `fused` embeddings
collected from that seed's F1 model, so the GAN learns the distribution the classifier
actually produces rather than an arbitrary one. That ordering is enforced below: F2 refuses
to run for a seed whose F1 checkpoint is missing.

⚠️ **The distribution the GAN learns is a snapshot.** F2 then trains from scratch, so its
fusion layer moves away from the one the samples describe. This is inherent to augmenting in
a learned space, it is stated in `SequenceGenerator`'s docstring, and it is a concrete reason
F2 may not beat F1 -- the outcome P9-6 pre-commits to reporting either way.

⛔ **P9-4 runs before any sample is used.** `generate_samples` raises `ModeCollapseError` rather
than warning, because a collapsed generator injects a perfectly learnable constant marker and
would make F2 *beat* F1 for a reason that has nothing to do with augmentation quality.

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
from src.data.manifest import read_manifest  # noqa: E402
from src.models.attention_model import AttentionFusionModel  # noqa: E402
from src.models.backbones.audio import NATIVE_DIM  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.training.augment import AugmentedPairs, ClassicalConfig, SpliceConfig  # noqa: E402
from src.training.gan import (  # noqa: E402
    ModeCollapseError,
    collect_fused,
    generate_samples,
    train_gan,
)
from src.training.reporting import baseline_zero, git_sha  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402
from src.utils.console import init_console  # noqa: E402


def _load_phase8():
    """Import Phase 8's sweep script for the helpers Experiment F must share with E.

    `diagnostics` *defines* what "frame AP", "attention lift" and the modality ablation mean
    in this project. Re-implementing them here would let Phase 9's numbers drift from Phase
    8's while still being printed in the same column, which is the one thing an experiment
    ladder cannot survive. The module name starts with a digit, so it is loaded by path.
    """
    path = REPO_ROOT / "scripts" / "27_train_attention.py"
    spec = importlib.util.spec_from_file_location("phase8_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase8_sweep"] = module
    spec.loader.exec_module(module)
    return module


P8 = _load_phase8()
GREEN, RED, YELLOW, DIM, RESET = P8.GREEN, P8.RED, P8.YELLOW, P8.DIM, P8.RESET
VISUAL_DIM = P8.VISUAL_DIM

# (splice, classical, gan). F0 first so a killed sweep still leaves the control.
ARMS = {
    "F0": (False, False, False),
    "F1": (True, True, False),
    "F2": (True, True, True),
}


class GanAugmentedTrainer(Trainer):
    """`Trainer` plus a loss term on GAN-generated fused sequences (arm F2 only).

    Overrides `_forward` rather than the training loop, so every other term -- classification,
    auxiliary heads, frame head, sync -- is computed by exactly the code F0 and F1 use. The
    generated term is strictly additive, which is what makes `F2 - F1` attributable to the
    samples and nothing else.
    """

    def __init__(self, *args, generator=None, gan_weight: float = 0.3, gan_batch: int = 8, **kw):
        super().__init__(*args, **kw)
        self.generator = generator
        self.gan_weight = gan_weight
        self.gan_batch = gan_batch

    def _forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        loss, out = super()._forward(batch)
        if self.generator is None or not self.gan_weight or not self.model.training:
            return loss, out

        n_frames = int(batch["mask"].sum(dim=1).max())
        with torch.no_grad():
            fake, labels = self.generator.sample(self.gan_batch, n_frames, device=self.device)
        gen_out = self.model.forward_fused(fake, None)
        gan_loss = self.criterion(gen_out["logit"], labels.float())
        return loss + self.gan_weight * gan_loss, out


def build_train_dataset(base, arm: str, args, seed: int):
    """The one object that differs between arms."""
    splice, classical, _ = ARMS[arm]
    if not (splice or classical):
        return base
    n_spliced = int(round(len(base) * args.splice_ratio)) if splice else 0
    return AugmentedPairs(
        base,
        n_spliced=n_spliced,
        splice_cfg=SpliceConfig(
            min_frames=args.splice_min_frames,
            max_frames=args.splice_max_frames,
        ),
        classical_cfg=ClassicalConfig(prob=args.classical_prob) if classical else None,
        seed=seed,
    )


def fit_gan_for(arm: str, seed: int, args, model_factory, train_loader, out_dir: Path, device: str):
    """Train the P9-3 generator on F1's fused embeddings and clear the P9-4 gate."""
    f1_ckpt = Path(args.out_root) / "phase9_F1" / f"seed{seed}" / "best.pt"
    if not f1_ckpt.exists():
        raise FileNotFoundError(
            f"F2 needs F1's checkpoint for seed {seed} ({f1_ckpt}). The generator is trained "
            "on the embeddings F1 produces -- run --arms F1 first."
        )
    frozen = model_factory().to(device)
    state = torch.load(f1_ckpt, map_location=device)
    frozen.load_state_dict(state.get("model", state))

    print(f"    collecting fused embeddings from {DIM}{f1_ckpt}{RESET}")
    seqs, labels = collect_fused(frozen, train_loader, device=device)
    print(f"    {len(seqs)} sequences, {int(labels.sum())} fake")

    gen, _, history = train_gan(
        seqs,
        labels,
        steps=args.gan_steps,
        batch_size=args.gan_batch_size,
        n_frames=args.gan_frames,
        device=device,
        seed=seed,
    )

    # ⛔ P9-4. Refuses rather than warns -- see the module docstring.
    reference = torch.stack([s[: args.gan_frames] for s in seqs if len(s) >= args.gan_frames][:64])
    _, _, div = generate_samples(gen, reference, 64, args.gan_frames, device=device)
    print(f"    {GREEN}P9-4 passed{RESET} {DIM}{div.as_dict()}{RESET}")
    (out_dir / "gan_history.json").write_text(
        json.dumps({"history": history, "diversity": div.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    del frozen, seqs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return gen, div


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Train the Phase 9 augmentation arms")
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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--attn-dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--frame-weight", type=float, default=1.0)
    ap.add_argument("--sync-weight", type=float, default=0.3)
    # P9-1
    ap.add_argument("--splice-ratio", type=float, default=0.5, help="spliced examples per real one")
    ap.add_argument("--splice-min-frames", type=int, default=5)
    ap.add_argument("--splice-max-frames", type=int, default=50)
    # P9-2
    ap.add_argument("--classical-prob", type=float, default=0.5)
    # P9-3
    ap.add_argument("--gan-steps", type=int, default=1500)
    ap.add_argument("--gan-batch-size", type=int, default=32)
    ap.add_argument("--gan-frames", type=int, default=64)
    ap.add_argument("--gan-weight", type=float, default=0.3)
    ap.add_argument("--gan-samples", type=int, default=8, help="generated clips per real batch")
    ap.add_argument("--out-root", default="experiments")
    ap.add_argument("--report-dir", default="reports")
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

    def make_model():
        return AttentionFusionModel(
            visual_dim=VISUAL_DIM[args.backbone],
            feature=args.feature,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            modality_dropout=args.modality_dropout,
            cross_attention=True,
            temporal="transformer",
            positional=True,
        )

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 9  Augmentation (Experiment F) -- {args.backbone} + "
        f"{args.feature}\n{DIM}{'=' * 72}{RESET}"
    )
    print(f"  visual   : {config_hash(vcfg)}  audio: {config_hash(acfg)}")
    print("  base     : the section 5.1 model (E full), unchanged")
    print(
        f"  splice   : x{args.splice_ratio} of train, {args.splice_min_frames}-"
        f"{args.splice_max_frames} frames"
    )
    print(f"  arms     : {', '.join(args.arms)}")
    for split, ds in datasets.items():
        b = ds.label_balance
        print(
            f"    {split:<6} {b['n']:>5} clips  {b['positive']:>4} fake / {b['negative']:>4} real"
        )

    dev_loader = P8.paired_loader(datasets["dev"], args.batch_size, shuffle=False, seed=0)
    sha = git_sha()

    for arm in args.arms:
        splice, classical, use_gan = ARMS[arm]
        print(
            f"\n{DIM}{'=' * 72}{RESET}\n  arm '{arm}'  splice={splice} classical={classical} "
            f"gan={use_gan}"
        )
        runs: list[dict] = []

        for seed in range(args.seeds):
            run_dir = Path(args.out_root) / f"phase9_{arm}" / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            done = run_dir / "result.json"
            if done.exists() and not args.restart:
                runs.append(json.loads(done.read_text(encoding="utf-8")))
                print(
                    f"\n  seed {seed}  {GREEN}already done{RESET} "
                    f"{DIM}(AUC {runs[-1]['dev']['auc']:.4f}){RESET}"
                )
                continue

            print(f"\n  seed {seed}")
            seed_everything(1337 + seed)
            train_ds = build_train_dataset(datasets["train"], arm, args, seed)
            balance = train_ds.label_balance
            if splice:
                print(
                    f"    train {balance['n']} clips after splicing "
                    f"({balance['positive']} fake / {balance['negative']} real)"
                )
            train_loader = P8.paired_loader(train_ds, args.batch_size, shuffle=True, seed=seed)

            generator, diversity_row = None, None
            if use_gan:
                plain_loader = P8.paired_loader(
                    datasets["train"], args.batch_size, shuffle=False, seed=seed
                )
                try:
                    generator, div = fit_gan_for(
                        arm, seed, args, make_model, plain_loader, run_dir, args.device
                    )
                    diversity_row = div.as_dict()
                except (FileNotFoundError, ModeCollapseError) as exc:
                    print(f"    {RED}{exc}{RESET}")
                    return 1

            pos = balance["negative"] / max(balance["positive"], 1)
            # Splicing changes the forged-frame rate, so the frame head's pos_weight has to
            # be recomputed per arm. F0 reuses Phase 8's helper so its number is identical to
            # the one the E full arm trained with.
            if isinstance(train_ds, AugmentedPairs):
                rate = train_ds.frame_positive_rate()
                frame_pw = (1 - rate) / rate if 0 < rate < 1 else None
                print(
                    f"    frame pos_weight: {frame_pw:.2f} "
                    f"{DIM}(forged frame rate {rate:.4f}){RESET}"
                )
            else:
                frame_pw = P8.frame_pos_weight_for(train_ds)

            tcfg = TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                pos_weight=pos,
                patience=args.patience,
                seed=1337 + seed,
                aux_weight=args.aux_weight,
                frame_weight=args.frame_weight,
                frame_pos_weight=frame_pw,
                sync_weight=args.sync_weight,
            )
            model = make_model()
            trainer_cls = GanAugmentedTrainer if use_gan else Trainer
            extra = (
                {
                    "generator": generator,
                    "gan_weight": args.gan_weight,
                    "gan_batch": args.gan_samples,
                }
                if use_gan
                else {}
            )
            runner = trainer_cls(model, tcfg, args.device, run_dir, **extra)

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
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

            dev_metrics = P8.diagnostics(runner, dev_loader)
            no_visual = P8.diagnostics(runner, dev_loader, drop_visual=True)
            no_audio = P8.diagnostics(runner, dev_loader, drop_audio=True)
            best_epoch = runner.best_epoch

            row = {
                "seed": 1337 + seed,
                "arm": arm,
                "splice": splice,
                "classical": classical,
                "gan": use_gan,
                "splice_ratio": args.splice_ratio if splice else 0.0,
                "classical_prob": args.classical_prob if classical else 0.0,
                "gan_weight": args.gan_weight if use_gan else 0.0,
                "gan_diversity": diversity_row,
                "backbone": args.backbone,
                "feature": args.feature,
                "defences": True,
                "sync_weight": args.sync_weight,
                "git_sha": sha,
                "n_params": model.n_parameters,
                "epochs_run": len(runner.history),
                "best_epoch": best_epoch,
                "wall_clock_s": round(elapsed, 2),
                "n_clips_train": balance["n"],
                "dev": dev_metrics,
                "dev_drop_visual": no_visual,
                "dev_drop_audio": no_audio,
                "train_config": asdict(tcfg),
            }
            done.write_text(json.dumps(row, indent=2, default=float) + "\n", encoding="utf-8")
            runner.write_history(run_dir / "history.json")
            runs.append(row)
            print(
                f"    best epoch {best_epoch}  AUC {dev_metrics['auc']:.4f}  "
                f"frame AP {dev_metrics.get('frame_ap', float('nan')):.4f}  "
                f"{DIM}(-visual {no_visual['auc']:.4f}){RESET}  {elapsed:.0f}s"
            )

        aucs = np.array([r["dev"]["auc"] for r in runs])
        faps = np.array([r["dev"].get("frame_ap", np.nan) for r in runs], dtype=float)
        dv = np.array([r["dev_drop_visual"]["auc"] for r in runs])
        da = np.array([r["dev_drop_audio"]["auc"] for r in runs])
        floors = baseline_zero(np.array([r for r in datasets["dev"].labels], dtype=float), seed=0)
        out = Path(args.report_dir) / f"phase9_{arm}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "arm": arm,
                    "splice": splice,
                    "classical": classical,
                    "gan": use_gan,
                    "backbone": args.backbone,
                    "feature": args.feature,
                    "defences": True,
                    "git_sha": sha,
                    "n_clips": len(full),
                    "n_params": runs[0]["n_params"],
                    "splits": {s: ds.label_balance for s, ds in datasets.items()},
                    "baseline_zero": floors,
                    "dev_auc_mean": float(aucs.mean()),
                    "dev_auc_std": float(aucs.std()),
                    "dev_frame_ap_mean": float(np.nanmean(faps)),
                    "dev_frame_ap_std": float(np.nanstd(faps)),
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
            f"frame AP {np.nanmean(faps):.4f}  drop-visual {aucs.mean() - dv.mean():+.4f}"
        )
        print(f"  wrote {out}")

    print(f"\n{GREEN}TRAINING DONE{RESET}  {DIM}run scripts/31_experiment_f.py for the gate{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
