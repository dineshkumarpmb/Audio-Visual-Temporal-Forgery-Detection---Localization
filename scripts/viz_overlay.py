"""P2-9: burn `fake_periods` onto video as a red overlay, for the ⛔✋ P2-10 check.

    python scripts/viz_overlay.py --subset smoke-100 --n 20
    python scripts/viz_overlay.py --video-id train_092507

Writes annotated MP4s to `reports/figures/overlays/`.

**Why this exists.** PROJECT_PLAN section 6.3 calls ground-truth alignment the
highest-risk step in the project, and this is the only check that catches a label
bug nothing else will. If `fake_periods` were misread — wrong units, wrong timeline,
off-by-one-frame rasterisation — every metric downstream would still look healthy
while the model learned from targets pointing at the wrong frames. Watching the red
band land on a visibly manipulated mouth is the evidence that the labels are real.

Each frame is annotated with:
  * a red border and "FORGED" banner while the frame is inside a `fake_periods` span;
  * a timeline strip along the bottom showing all spans and a playhead;
  * the frame index, timestamp, and the video's class (real / visual_only / ...).

Spans are rasterised the same way the training targets are — `round(t * 25)`, clamped
to `n_frames` — so what you watch is literally what the model will be taught.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import LAVDF_FPS  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def frame_mask(fake_periods, n_frames: int, fps: float = LAVDF_FPS) -> np.ndarray:
    """Rasterise spans to a per-frame boolean — the training target, exactly.

    Kept identical to the localization target builder on purpose: if this drifts from
    that, P2-10 stops proving anything about the labels the model actually sees.
    """
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in fake_periods:
        a = max(0, int(round(float(start) * fps)))
        b = min(n_frames, int(round(float(end) * fps)))
        if b > a:
            mask[a:b] = True
    return mask


def annotate(frame: np.ndarray, i: int, mask: np.ndarray, label: str) -> np.ndarray:
    """Draw the border, banner, timeline and captions onto one BGR frame."""
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    n = len(mask)
    forged = bool(mask[i])

    if forged:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.rectangle(out, (0, 0), (w, 22), (0, 0, 200), -1)
        cv2.putText(
            out, "FORGED", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Timeline strip: grey track, red spans, white playhead.
    bar_h, pad = 10, 4
    y0 = h - bar_h - pad
    cv2.rectangle(out, (pad, y0), (w - pad, y0 + bar_h), (60, 60, 60), -1)
    track = w - 2 * pad
    for j in range(n):
        if mask[j]:
            x = pad + int(track * j / n)
            cv2.line(out, (x, y0), (x, y0 + bar_h), (0, 0, 255), 1)
    px = pad + int(track * i / max(n - 1, 1))
    cv2.line(out, (px, y0 - 2), (px, y0 + bar_h + 2), (255, 255, 255), 1)

    caption = f"{i:>4}/{n}  t={i / LAVDF_FPS:6.2f}s  {label}"
    cv2.putText(
        out, caption, (pad, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, caption, (pad, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out


def render(video_path: Path, row, out_path: Path, scale: int = 2) -> dict:
    """Write one annotated MP4. Returns what was drawn, for the report."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    n_frames = int(row.n_frames)
    mask = frame_mask(row.fake_periods, n_frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), LAVDF_FPS, (w, h))
    i = 0
    try:
        while i < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            big = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
            writer.write(annotate(big, i, mask, row.class_name))
            i += 1
    finally:
        cap.release()
        writer.release()

    return {
        "video_id": row.Index if hasattr(row, "Index") else "",
        "frames": i,
        "forged_frames": int(mask.sum()),
        "spans": [[float(a), float(b)] for a, b in row.fake_periods],
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Burn fake_periods onto video (P2-9)")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--video-root", default=None, help="default: data/raw/LAV-DF/<subset>")
    ap.add_argument("--out-dir", default="reports/figures/overlays")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument(
        "--fakes-only",
        action="store_true",
        default=True,
        help="only videos that have spans (a real video shows nothing)",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    video_root = Path(args.video_root or f"data/raw/LAV-DF/{args.subset}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest).set_index("video_id")

    if args.video_id:
        ids = [args.video_id]
    else:
        ids = [
            v
            for v in load_subset(args.subset, args.subset_dir)
            if (video_root / f"{v}.mp4").exists()
        ]
        if args.fakes_only:
            ids = [v for v in ids if len(manifest.loc[v, "fake_periods"]) > 0]
        random.Random(args.seed).shuffle(ids)
        ids = sorted(ids[: args.n])

    if not ids:
        print(f"{RED}no videos found under {video_root}{RESET}")
        print(f"  fetch them: python scripts/05_fetch_metadata.py --subset {args.subset}")
        return 1

    print(f"  rendering {len(ids)} overlay(s) -> {out}")
    for vid in ids:
        row = manifest.loc[vid]
        info = render(video_root / f"{vid}.mp4", row, out / f"{vid}.mp4")
        spans = ", ".join(f"[{a:.2f},{b:.2f}]" for a, b in info["spans"])
        print(
            f"  [{GREEN}OK{RESET}] {vid:<18} {info['frames']:>4}f  "
            f"forged {info['forged_frames']:>3}f  {DIM}{spans}{RESET}"
        )

    print(
        f"\n{YELLOW}⛔✋ P2-10 is now yours:{RESET} watch these and confirm the red band lands on"
    )
    print("  visibly manipulated speech. This is the only check that catches a label bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
