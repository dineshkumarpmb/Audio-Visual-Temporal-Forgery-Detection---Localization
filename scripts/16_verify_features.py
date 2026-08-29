"""P4-4: sanity-check the cached visual features before trusting anything trained on them.

    python scripts/16_verify_features.py --backbone resnet18

Three checks, each catching a different silent failure:

**Determinism.** Re-extract a handful of clips and compare the raw bytes. The backbone is
frozen and pinned to `eval()`, so this must be bit-exact. If it is not, the most likely
cause is BatchNorm having slipped into batch-statistics mode — which would make cached
features disagree with live ones and quietly poison every downstream number.

**Embedding norms.** A frozen ImageNet trunk should produce features in a stable, sane
range. All-zero features mean a dead ReLU stack or a normalisation bug; exploding norms
mean the ImageNet normalisation was skipped. Either shows up here in seconds and nowhere
else until AUC comes back at 0.5.

**Identity separation (t-SNE).** Clips sharing a `source_id` are the same underlying
recording, so their mean embeddings should sit together. If they do not, the features
carry no usable identity signal and something upstream — alignment, crops, the backbone
itself — is broken. This is a *positive control*: it does not test forgery detection, it
tests that the features encode anything at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import VideoPreprocessConfig, VisualFeatureConfig, config_hash  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.models.backbones.visual import build_backbone  # noqa: E402
from src.preprocessing.cache import cache_root  # noqa: E402
from src.preprocessing.face import FaceLandmarkerPool  # noqa: E402
from src.preprocessing.pipeline import align_all, detect_track  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, ok, detail))
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
        return ok

    @property
    def failed(self) -> list[str]:
        return [n for n, ok, _ in self.rows if not ok]


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="P4-4 visual feature sanity checks")
    ap.add_argument("--backbone", choices=("resnet18", "mobilenet_v2"), default="resnet18")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--feature-root", default="data/features/visual")
    ap.add_argument("--video-root", default="data/raw/LAV-DF")
    ap.add_argument("--n-determinism", type=int, default=4)
    ap.add_argument("--out", default=None, help="default: reports/feature_checks_<backbone>.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(1337)
    cfg = VisualFeatureConfig(backbone=args.backbone)
    root = cache_root(cfg, args.feature_root)
    out_path = Path(args.out or f"reports/feature_checks_{args.backbone}.md")

    files = sorted(root.glob("*.npz"))
    print(
        f"{DIM}{'=' * 72}{RESET}\n  P4-4  Feature sanity -- {args.backbone}\n{DIM}{'=' * 72}{RESET}"
    )
    if not files:
        print(f"{RED}no features under {root}{RESET} -- run scripts/13_extract_visual.py")
        return 1
    print(f"  {len(files)} cached clips  {DIM}{root}{RESET}\n")

    c = Checks()
    facts: dict[str, object] = {
        "backbone": args.backbone,
        "config_hash": config_hash(cfg),
        "n_clips": len(files),
    }
    manifest = read_manifest(args.manifest).set_index("video_id")

    # ---- embedding norms --------------------------------------------------------
    norms, zero_clips, all_feats, ids = [], 0, [], []
    for f in files:
        with np.load(f) as d:
            feats = np.asarray(d["features"], dtype=np.float32)
        n = np.linalg.norm(feats, axis=1)
        norms.append(n)
        if not np.any(n > 1e-6):
            zero_clips += 1
        all_feats.append(feats.mean(axis=0))
        ids.append(f.stem)
    flat = np.concatenate(norms)
    facts["norm"] = {
        k: float(v)
        for k, v in zip(
            ("min", "p05", "median", "p95", "max", "mean"),
            (
                flat.min(),
                np.percentile(flat, 5),
                np.median(flat),
                np.percentile(flat, 95),
                flat.max(),
                flat.mean(),
            ),
            strict=True,
        )
    }

    c.add("no all-zero clips", zero_clips == 0, f"{zero_clips} dead of {len(files)}")
    c.add(
        "all embeddings finite",
        bool(np.isfinite(flat).all()),
        f"{int((~np.isfinite(flat)).sum())} non-finite values",
    )
    c.add(
        "norms in a sane range",
        0.1 < np.median(flat) < 1e4,
        f"median {np.median(flat):.2f}, range [{flat.min():.2f}, {flat.max():.2f}]",
    )

    # ---- determinism ------------------------------------------------------------
    video_cfg = VideoPreprocessConfig()
    probe_ids = [i for i in ids if _find_video(Path(args.video_root), i)][: args.n_determinism]
    if probe_ids:
        backbone = build_backbone(cfg.backbone).to(args.device)
        landmarker = FaceLandmarkerPool()
        identical = 0
        for vid in probe_ids:
            path = _find_video(Path(args.video_root), vid)
            n_video = int(manifest.loc[vid, "n_frames"])
            track = detect_track(path, n_video, video_cfg, landmarker)
            crops = align_all(path, track, video_cfg)
            with torch.no_grad():
                chunks = [
                    backbone(
                        torch.from_numpy(np.ascontiguousarray(crops[s : s + 64]))
                        .to(args.device)
                        .permute(0, 3, 1, 2)
                    )
                    .float()
                    .cpu()
                    .numpy()
                    for s in range(0, len(crops), 64)
                ]
            fresh = np.concatenate(chunks).astype(np.float16)
            with np.load(root / f"{vid}.npz") as d:
                cached = np.asarray(d["features"])
            identical += int(np.array_equal(fresh, cached))
        landmarker.close()
        c.add(
            "re-extraction is byte-identical",
            identical == len(probe_ids),
            f"{identical}/{len(probe_ids)} clips",
        )
        facts["determinism_identical"] = identical
        facts["determinism_n"] = len(probe_ids)
    else:
        c.add("re-extraction is byte-identical", False, "no source video available locally")

    # ---- identity separation ----------------------------------------------------
    embeddings = np.stack(all_feats)
    source = manifest.loc[ids, "source_id"].to_numpy()
    sep = _identity_separation(embeddings, source)
    facts["identity"] = sep
    c.add(
        "same-identity clips are closer than different-identity",
        sep["ratio"] < 1.0,
        f"within {sep['within']:.3f} vs between {sep['between']:.3f} "
        f"(ratio {sep['ratio']:.3f}, {sep['n_groups']} groups)",
    )

    _tsne(embeddings, source, ids, args.backbone, facts)

    _write(out_path, c, facts)
    Path(f"reports/feature_checks_{args.backbone}.json").write_text(
        json.dumps(facts, indent=2, default=float) + "\n", encoding="utf-8"
    )

    print(f"\n{DIM}{'-' * 72}{RESET}")
    passed = len(c.rows) - len(c.failed)
    print(f"  {passed}/{len(c.rows)} checks passed")
    if c.failed:
        print(f"\n{RED}FEATURE CHECKS FAILED{RESET} -- {c.failed}")
        return 1
    print(f"\n{GREEN}FEATURES OK{RESET}  wrote {out_path}")
    return 0


def _find_video(root: Path, vid: str) -> Path | None:
    for sub in root.iterdir() if root.exists() else []:
        p = sub / f"{vid}.mp4"
        if p.exists():
            return p
    return None


def _identity_separation(embeddings: np.ndarray, source: np.ndarray) -> dict:
    """Mean cosine distance within a `source_id` group vs between groups.

    A ratio below 1 means clips from the same recording really are closer together, i.e.
    the embeddings carry identity. It is a positive control on the features, not a
    forgery metric.
    """
    x = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    dist = 1.0 - x @ x.T
    same = source[:, None] == source[None, :]
    off = ~np.eye(len(x), dtype=bool)

    within = dist[same & off]
    between = dist[~same & off]
    groups = int(
        sum(1 for _, n in zip(*np.unique(source, return_counts=True), strict=True) if n > 1)
    )
    return {
        "within": float(within.mean()) if within.size else float("nan"),
        "between": float(between.mean()) if between.size else float("nan"),
        "ratio": float(within.mean() / between.mean())
        if within.size and between.size
        else float("nan"),
        "n_groups": groups,
        "n_within_pairs": int(within.size),
    }


def _tsne(
    embeddings: np.ndarray, source: np.ndarray, ids: list[str], backbone: str, facts: dict
) -> None:
    """Identity-separation scatter. Never fatal — a missing plot must not fail the gate."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        n = len(embeddings)
        if n < 10:
            return
        coords = TSNE(
            n_components=2, random_state=1337, perplexity=min(30, max(5, n // 4)), init="pca"
        ).fit_transform(embeddings)

        # Colour only the identities with more than one clip; the rest are grey context.
        uniq, counts = np.unique(source, return_counts=True)
        multi = set(uniq[counts > 1])
        fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
        solo = np.array([s not in multi for s in source])
        ax.scatter(
            coords[solo, 0],
            coords[solo, 1],
            c="#cccccc",
            s=14,
            label="single-clip identity",
            linewidths=0,
        )
        for i, sid in enumerate(sorted(multi)):
            sel = source == sid
            ax.scatter(
                coords[sel, 0], coords[sel, 1], s=30, color=plt.cm.tab20(i % 20), linewidths=0
            )
        ax.set_title(
            f"{backbone}: identity separation (t-SNE of mean embeddings)\n"
            f"{len(multi)} identities with >1 clip coloured, {n} clips"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        out = Path(f"reports/figures/tsne_identity_{backbone}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        facts["tsne_path"] = str(out)
        print(f"  [{DIM}plot{RESET}] wrote {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{YELLOW}WARN{RESET}] t-SNE skipped: {type(exc).__name__}: {exc}")


def _write(path: Path, c: Checks, facts: dict) -> None:
    n = facts["norm"]
    idn = facts["identity"]
    lines = [
        f"# P4-4 — visual feature sanity checks ({facts['backbone']})",
        "",
        f"Generated by `scripts/16_verify_features.py`. Config `{facts['config_hash']}`, "
        f"{facts['n_clips']} cached clips.",
        "",
        f"**Result: {len(c.rows) - len(c.failed)}/{len(c.rows)} checks passed.**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in c.rows:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## Embedding norms",
        "",
        "| min | p05 | median | p95 | max |",
        "|---|---|---|---|---|",
        f"| {n['min']:.2f} | {n['p05']:.2f} | {n['median']:.2f} | {n['p95']:.2f} | "
        f"{n['max']:.2f} |",
        "",
        "All-zero features would mean a dead trunk or a normalisation bug; exploding norms",
        "would mean the ImageNet normalisation was skipped. Both surface here in seconds,",
        "and nowhere else until AUC comes back at 0.5 and the cause is unclear.",
        "",
        "## Identity separation (positive control)",
        "",
        f"Mean cosine distance **within** a `source_id`: **{idn['within']:.4f}**; "
        f"**between** identities: **{idn['between']:.4f}**; ratio **{idn['ratio']:.4f}** over "
        f"{idn['n_groups']} multi-clip identities ({idn['n_within_pairs']} within-pairs).",
        "",
        "A ratio below 1 means clips of the same underlying recording really do sit together,",
        "so the embeddings encode *something* real. This is not a forgery metric — it is the",
        'check that distinguishes "the model is bad at this task" from "the features are',
        'broken", which are otherwise indistinguishable from a low AUC alone.',
        "",
    ]
    if facts.get("tsne_path"):
        lines += [f"Plot: `{facts['tsne_path']}` (gitignored; regenerate with this script).", ""]
    if "determinism_identical" in facts:
        lines += [
            "## Determinism",
            "",
            f"{facts['determinism_identical']}/{facts['determinism_n']} clips re-extracted",
            "**byte-identically**. The backbone is frozen and `train()` is overridden to keep it",
            "in eval, so BatchNorm always uses running statistics. Without that override a",
            "`Trainer.train()` call would flip it into batch-statistics mode and cached features",
            "would silently stop matching live ones.",
        ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
