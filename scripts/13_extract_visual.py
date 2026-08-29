"""Phase 4 Stage-A: frozen-backbone visual features (P4-2, P4-3).

    python scripts/13_extract_visual.py --subset dev-2k --backbone resnet18 --from-video
    python scripts/13_extract_visual.py --subset dev-2k --backbone mobilenet_v2 --from-video

The plan names this `scripts/04_extract_visual.py`; that number was already taken by the
Phase 1 statistics script, so it lands here. Everything else follows section 5.1.

⛔ **`--from-video` is the PF-10 answer and the default for anything past smoke-100.**
Face crops are 7.52 MB/video, so dev-2k would be 14.7 GB on disk and dev-10k 73.5 GB —
more than Kaggle's entire 20 GB `/kaggle/working`. In this mode a video is decoded,
aligned, featurised and discarded in one pass, so peak disk is one clip. The features
themselves are ~200 KB/video, a 37x reduction.

Reading from a warm crop cache stays the default when one exists, because re-running the
second backbone over cached crops is far faster than re-decoding.

**Precision (PF-4).** Section 5.1 asks for autocast fp16. Phase 0 measured fp16 arithmetic
at **0.13x** fp32 on this GTX 1650 — TU117 is the one Turing die without tensor cores — so
compute is fp32 and only the stored output is cast to fp16. That is storage, not
arithmetic: it halves the cache and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import VideoPreprocessConfig, VisualFeatureConfig, config_hash  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.models.backbones.visual import build_backbone  # noqa: E402
from src.preprocessing.cache import cache_root, is_cached, load_faces  # noqa: E402
from src.preprocessing.face import FaceLandmarkerPool  # noqa: E402
from src.preprocessing.pipeline import align_all, detect_track  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
FEATURE_ROOT = Path("data/features/visual")


def feature_path(video_id: str, cfg: VisualFeatureConfig, root: Path) -> Path:
    return cache_root(cfg, root) / f"{video_id}.npz"


@torch.no_grad()
def featurise(crops: np.ndarray, backbone, cfg: VisualFeatureConfig, device: str) -> np.ndarray:
    """`(T, S, S, 3)` uint8 -> `(T, D)` float16.

    Batched so a 500-frame clip never puts 500 images on a 4 GB card at once.
    """
    out = np.empty(
        (len(crops), cfg.feature_dim), dtype=np.float16 if cfg.store_fp16 else np.float32
    )
    for start in range(0, len(crops), cfg.batch_size):
        chunk = np.ascontiguousarray(crops[start : start + cfg.batch_size])
        batch = torch.from_numpy(chunk).to(device).permute(0, 3, 1, 2)
        feats = backbone(batch).float().cpu().numpy()
        out[start : start + len(chunk)] = feats.astype(out.dtype)
    return out


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Extract frozen visual features (Phase 4 Stage A)")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--backbone", choices=("resnet18", "mobilenet_v2"), default="resnet18")
    ap.add_argument("--video-root", default=None, help="default: data/raw/LAV-DF/<subset>")
    ap.add_argument("--faces-root", default="data/interim/faces")
    ap.add_argument("--out-root", default=str(FEATURE_ROOT))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--from-video",
        action="store_true",
        help="decode+align+featurise in one pass, never writing crops (PF-10)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = VisualFeatureConfig(backbone=args.backbone, batch_size=args.batch_size)
    video_cfg = VideoPreprocessConfig()
    video_root = Path(args.video_root or f"data/raw/LAV-DF/{args.subset}")
    out_root = Path(args.out_root)

    ids = load_subset(args.subset, args.subset_dir)
    if args.limit:
        ids = ids[: args.limit]
    manifest = read_manifest(args.manifest).set_index("video_id")

    backbone = build_backbone(cfg.backbone).to(args.device)
    info = backbone.info()

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 4  Visual features -- {cfg.backbone}\n{DIM}{'=' * 72}{RESET}"
    )
    print(
        f"  config  : {config_hash(cfg)}  {DIM}dim={info.feature_dim} "
        f"params={info.n_params / 1e6:.2f}M trainable={info.n_trainable} "
        f"batch={cfg.batch_size} store={'fp16' if cfg.store_fp16 else 'fp32'}{RESET}"
    )
    print(
        f"  device  : {args.device}"
        + (f"  {DIM}{torch.cuda.get_device_name(0)}{RESET}" if args.device == "cuda" else "")
    )
    print(f"  source  : {'video (streaming, PF-10)' if args.from_video else 'cached crops'}")

    todo = [v for v in ids if args.force or not feature_path(v, cfg, out_root).exists()]
    skipped = len(ids) - len(todo)
    print(f"  videos  : {len(ids)} in subset, {skipped} cached, {len(todo)} to do")
    if not todo:
        print(f"\n{GREEN}NOTHING TO DO{RESET} -- all features present")
        return 0

    landmarker = FaceLandmarkerPool() if args.from_video else None
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    done, failed, n_frames_total = 0, [], 0
    # A video that has not been downloaded yet is not a pipeline failure -- it is simply
    # absent. Counting it as one would make the exit code meaningless while a subset is
    # still being fetched.
    missing: list[str] = []
    found_fractions: list[float] = []

    for i, vid in enumerate(todo, 1):
        try:
            n_video = int(manifest.loc[vid, "n_frames"])
            if args.from_video:
                path = video_root / f"{vid}.mp4"
                if not path.exists():
                    missing.append(vid)
                    continue
                track = detect_track(path, n_video, video_cfg, landmarker)
                crops = align_all(path, track, video_cfg)
                found = track.found
            else:
                if not is_cached(vid, video_cfg, args.faces_root):
                    failed.append((vid, "crops not cached -- use --from-video"))
                    continue
                crops, found = load_faces(vid, video_cfg, args.faces_root)
                crops, found = np.asarray(crops), np.asarray(found)

            feats = featurise(crops, backbone, cfg, args.device)
            dest = feature_path(vid, cfg, out_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".npz.tmp")
            with tmp.open("wb") as fh:
                np.savez(fh, features=feats, found=found)
            tmp.replace(dest)

            done += 1
            n_frames_total += len(feats)
            found_fractions.append(float(found.mean()))
        except Exception as exc:  # noqa: BLE001 - one bad video must not kill a long run
            failed.append((vid, f"{type(exc).__name__}: {exc}"))

        if i % 100 == 0 or i == len(todo):
            rate = i / max(time.time() - start, 1e-9)
            print(f"    {i}/{len(todo)}  {DIM}{rate:.2f} videos/s{RESET}", flush=True)

    if landmarker is not None:
        landmarker.close()

    elapsed = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else 0.0
    root = cache_root(cfg, out_root)
    size = sum(f.stat().st_size for f in root.glob("*.npz")) if root.exists() else 0

    print(f"\n{DIM}{'-' * 72}{RESET}")
    print(
        f"  extracted {done}, failed {len(failed)}"
        + (f", {YELLOW}{len(missing)} video(s) not downloaded{RESET}" if missing else "")
    )
    print(
        f"  wall-clock {elapsed:.1f}s"
        + (
            f"  ({elapsed / done:.2f}s/video, {n_frames_total / elapsed:.0f} frames/s)"
            if done
            else ""
        )
    )
    print(
        f"  peak VRAM {peak_mb:.0f} MB   cache {root}  {size / 1024**2:.1f} MB"
        + (f" ({size / done / 1024:.0f} KB/video)" if done else "")
    )
    for vid, err in failed[:8]:
        print(f"  [{RED}FAIL{RESET}] {vid}: {err[:100]}")

    if done:
        facts = {
            "subset": args.subset,
            "backbone": cfg.backbone,
            "config_hash": config_hash(cfg),
            "feature_dim": cfg.feature_dim,
            "n_params": info.n_params,
            "n_extracted": done,
            "n_failed": len(failed),
            "n_frames": n_frames_total,
            "wall_clock_s": round(elapsed, 2),
            "seconds_per_video": round(elapsed / done, 4),
            "frames_per_second": round(n_frames_total / elapsed, 1),
            "peak_vram_mb": round(peak_mb, 1),
            "cache_bytes": size,
            "bytes_per_video": round(size / done),
            "face_found_mean": round(float(np.mean(found_fractions)), 4),
            "n_missing": len(missing),
            "source": "video" if args.from_video else "crops",
            "device": args.device,
        }
        out = Path(f"reports/visual_features_{cfg.backbone}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out}")

    ok = not failed
    print(f"\n{GREEN if ok else RED}{'FEATURES OK' if ok else 'FEATURES HAD FAILURES'}{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
