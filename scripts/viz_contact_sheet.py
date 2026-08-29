"""P2-7: contact sheets of extracted crops, for the ✋ P2-8 eyeball check.

    python scripts/viz_contact_sheet.py                    # 20 random videos
    python scripts/viz_contact_sheet.py --video-id dev_101081

Writes one PNG per video to `reports/figures/contact_sheets/`. Each sheet is a grid of
frames sampled evenly across the clip, so misalignment, rotation, identity flips and
detection dropouts are all visible at a glance.

What to look for when checking these by hand (P2-8):

  * faces upright, eyes level and at the same height in every tile;
  * the same person throughout — an identity flip means the P2-4 tracking policy lost
    the subject;
  * red-tinted tiles: frames where `face_found` is False, i.e. the geometry is held
    over from a neighbour and the pixels are not to be trusted;
  * heavy smearing at an edge: the crop is running off the source frame. Some is
    expected on LAV-DF (224x224 pre-cropped source, ~11% of crop area — see PF-9),
    a lot means the margin is set too wide.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import VideoPreprocessConfig, config_hash  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.preprocessing.cache import cache_root, is_cached, load_faces  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def contact_sheet(crops: np.ndarray, found: np.ndarray, cols: int = 8, rows: int = 4) -> np.ndarray:
    """Grid of `cols * rows` evenly-spaced frames; unfound frames tinted red."""
    n = min(cols * rows, len(crops))
    idx = np.linspace(0, len(crops) - 1, n).astype(int)
    size = crops.shape[1]

    tiles = []
    for i in idx:
        tile = crops[i].astype(np.uint8).copy()
        if not found[i]:
            # Unmistakable but still legible: push red, damp green and blue.
            tile[..., 0] = np.minimum(255, tile[..., 0].astype(np.int16) + 60)
            tile[..., 1] = (tile[..., 1] * 0.6).astype(np.uint8)
            tile[..., 2] = (tile[..., 2] * 0.6).astype(np.uint8)
        tiles.append(tile)

    while len(tiles) < cols * rows:
        tiles.append(np.zeros((size, size, 3), dtype=np.uint8))

    return np.concatenate(
        [np.concatenate(tiles[r * cols : (r + 1) * cols], axis=1) for r in range(rows)], axis=0
    )


def main() -> int:
    init_console()
    import cv2

    ap = argparse.ArgumentParser(description="Contact sheets for extracted face crops")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--faces-root", default="data/interim/faces")
    ap.add_argument("--out-dir", default="reports/figures/contact_sheets")
    ap.add_argument("--video-id", default=None, help="just this one")
    ap.add_argument("--n", type=int, default=20, help="how many videos to sheet")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = VideoPreprocessConfig()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.video_id:
        ids = [args.video_id]
    else:
        ids = [
            v
            for v in load_subset(args.subset, args.subset_dir)
            if is_cached(v, cfg, args.faces_root)
        ]
        random.Random(args.seed).shuffle(ids)
        ids = sorted(ids[: args.n])

    if not ids:
        print(f"{RED}no cached crops under {cache_root(cfg, args.faces_root)}{RESET}")
        print("  run: python scripts/09_extract_faces.py --subset " + args.subset)
        return 1

    print(f"  config {config_hash(cfg)}  ->  {out}")
    summary = []
    for vid in ids:
        crops, found = load_faces(vid, cfg, args.faces_root)
        sheet = contact_sheet(np.asarray(crops), np.asarray(found))
        path = out / f"{vid}.png"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        summary.append((vid, len(crops), float(np.mean(found))))
        print(
            f"  [{GREEN}OK{RESET}] {vid:<18} {len(crops):>4} frames  "
            f"face_found {np.mean(found):.3f}  {DIM}{path}{RESET}"
        )

    worst = sorted(summary, key=lambda r: r[2])[:3]
    print(f"\n  {DIM}lowest face_found: " + ", ".join(f"{v}={f:.2f}" for v, _, f in worst) + RESET)
    print(
        f"\n{GREEN}{len(ids)} contact sheet(s) written{RESET} -- now do P2-8: open them and look."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
