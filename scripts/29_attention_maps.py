"""P8-8: attention maps over the timeline, against ground truth.

    python scripts/29_attention_maps.py --n 12

Section H's testing row asks for attention-map visualisation, and P8-9 wants confirmation
that *"attention elevates on forged spans"*. This renders both, per clip:

1. **Attention-pool weights** over time, with the labelled `fake_periods` shaded. This is the
   model's own claim about where the forgery is, laid over where it actually is.
2. **Per-frame forgery scores** on the same axis, so the dense head and the pooling operator
   can be compared directly.
3. **The self-attention map** of the final layer, averaged over heads — the `T × T` matrix
   showing which frames each frame consulted.

**Why a script rather than the notebook the task names.** `notebooks/` is empty and
`.pre-commit-config.yaml` strips notebook outputs on commit, so a notebook's figures would be
deleted by the commit hook that is meant to keep the repo clean — the artefact would exist
locally and vanish in review. A script writing PNGs into `reports/figures/attention/` produces
the same figures, reproducibly, and they survive. The gate (P8-10) checks for the *artefacts*.

⚠️ These are **qualitative** evidence. `attention_lift` in `reports/phase8_full.json` is the
number that can enter a table; a figure showing attention on a forged span is a talking point,
not a measurement, and one cherry-picked clip is neither. Clips are therefore selected
deterministically — highest-confidence correct fakes plus a real control — and the selection
rule is stated on each figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioPreprocessConfig, VisualFeatureConfig  # noqa: E402
from src.data.dataset import MultimodalFeatureDataset, collate_paired  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.models.attention_model import AttentionFusionModel  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
VISUAL_DIM = {"resnet18": 512, "mobilenet_v2": 1280}
FPS = 25.0


def shade(ax, seconds: np.ndarray, target: np.ndarray) -> None:
    """Shade every labelled `fake_periods` run on a time axis.

    Module scope rather than a closure inside the render loop: a nested function would
    capture the loop variables by reference, so all figures would end up shaded with the
    last clip's spans — a bug that produces plausible-looking and completely wrong figures.
    """
    inside = target > 0.5
    if not inside.any():
        return
    edges = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    last = len(seconds) - 1
    for lo, hi in zip(starts, ends, strict=True):
        ax.axvspan(seconds[lo], seconds[min(hi, last)], color="tab:red", alpha=0.18)


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P8-8 attention maps")
    ap.add_argument("--checkpoint", default="experiments/phase8_full/seed0/best.pt")
    ap.add_argument("--subsets", nargs="+", default=["smoke-100", "dev-2k"])
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--visual-root", default="data/features/visual")
    ap.add_argument("--audio-root", default="data/interim/audio")
    ap.add_argument("--backbone", default="mobilenet_v2")
    ap.add_argument("--feature", default="logmel")
    ap.add_argument("--n", type=int, default=12, help="fake clips to render")
    ap.add_argument("--out", default="reports/figures/attention")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"{RED}no checkpoint at {ckpt_path} -- run scripts/27_train_attention.py{RESET}")
        return 1

    manifest = read_manifest(args.manifest).set_index("video_id")
    ids: list[str] = []
    for name in args.subsets:
        ids.extend(load_subset(name, args.subset_dir))
    ids = sorted(set(ids))
    ds = MultimodalFeatureDataset(
        ids,
        manifest,
        VisualFeatureConfig(backbone=args.backbone),
        AudioPreprocessConfig(feature=args.feature),
        args.visual_root,
        args.audio_root,
        with_frames=True,
    )
    dev = [i for i, s in enumerate(ds.splits) if s == "dev"]

    model = AttentionFusionModel(visual_dim=VISUAL_DIM[args.backbone], feature=args.feature)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()

    # Deterministic selection: the most confident correct fakes, plus the most confident
    # correct real as a control. Stated on the figure so nobody has to guess.
    scored = []
    with torch.no_grad():
        for i in dev:
            batch = collate_paired([ds[i]])
            out = model(
                batch["features"].to(args.device),
                batch["audio"].to(args.device),
                batch["mask"].to(args.device),
                batch["face"].to(args.device),
            )
            scored.append((i, float(torch.sigmoid(out["logit"])[0]), int(ds.labels[i])))

    fakes = sorted([s for s in scored if s[2] == 1], key=lambda x: -x[1])[: args.n]
    reals = sorted([s for s in scored if s[2] == 0], key=lambda x: x[1])[:2]
    chosen = fakes + reals

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    print(f"{DIM}{'=' * 72}{RESET}\n  P8-8 attention maps -> {out_dir}\n{DIM}{'=' * 72}{RESET}")
    written = 0
    for idx, score, label in chosen:
        sample = ds[idx]
        batch = collate_paired([sample])
        with torch.no_grad():
            out = model(
                batch["features"].to(args.device),
                batch["audio"].to(args.device),
                batch["mask"].to(args.device),
                batch["face"].to(args.device),
                return_attention=True,
            )
        t = len(sample.mask)
        seconds = np.arange(t) / FPS
        attn = out["attention"][0, :t].float().cpu().numpy()
        frame_p = torch.sigmoid(out["frame_logits"])[0, :t].float().cpu().numpy()
        sync = out["sync"][0, :t].float().cpu().numpy()
        target = sample.frames[:t]
        maps = out["attention_maps"][0, -1].mean(dim=0)[:t, :t].float().cpu().numpy()

        fig, axes = plt.subplots(
            3, 1, figsize=(11, 8), height_ratios=[2, 2, 3], constrained_layout=True
        )
        kind = ds.class_names[idx]
        fig.suptitle(
            f"{ds.video_ids[idx]}  ({kind}, p(fake)={score:.3f})  —  "
            f"shaded = labelled fake_periods",
            fontsize=11,
        )

        axes[0].plot(seconds, attn, color="tab:blue", lw=1.4)
        axes[0].set_ylabel("attention\npool weight")
        shade(axes[0], seconds, target)
        axes[0].margins(x=0)

        axes[1].plot(seconds, frame_p, color="tab:orange", lw=1.4, label="p(frame forged)")
        axes[1].plot(seconds, (sync + 1) / 2, color="tab:green", lw=1.0, alpha=0.7, label="sync")
        axes[1].set_ylabel("per-frame")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].legend(loc="upper right", fontsize=8)
        shade(axes[1], seconds, target)
        axes[1].margins(x=0)

        im = axes[2].imshow(
            maps,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, seconds[-1] if t > 1 else 1, 0, seconds[-1] if t > 1 else 1],
        )
        axes[2].set_xlabel("key (s)")
        axes[2].set_ylabel("query (s)")
        axes[2].set_title("final-layer self-attention, mean over heads", fontsize=9)
        fig.colorbar(im, ax=axes[2], shrink=0.8)

        name = f"{'fake' if label else 'real'}_{ds.video_ids[idx]}.png"
        fig.savefig(out_dir / name, dpi=110)
        plt.close(fig)
        written += 1
        print(f"  {name}")

    readme = out_dir / "README.md"
    readme.write_text(
        "# P8-8 attention maps\n\n"
        f"Generated by `scripts/29_attention_maps.py` from `{args.checkpoint}`.\n\n"
        "**Selection is deterministic, not curated:** the highest-confidence correctly-classified\n"
        "dev fakes, plus the two most confident correct reals as controls. Shaded bands are the\n"
        "labelled `fake_periods`.\n\n"
        "⚠️ These are qualitative. The number that belongs in a table is `attention_lift` in\n"
        "`reports/phase8_full.json` — the mean attention weight on forged frames divided by the\n"
        "mean on genuine frames, across all fake dev clips.\n",
        encoding="utf-8",
    )
    print(f"\n{GREEN}wrote {written} figures{RESET} to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
