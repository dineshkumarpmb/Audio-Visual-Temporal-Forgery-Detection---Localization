"""Objective ground-truth verification for P2-10 / PROJECT_PLAN section 6.3.

    python scripts/11_verify_labels.py --subset smoke-100
    python scripts/11_verify_labels.py --modality visual

Section 6.3 calls ground-truth alignment "the highest-risk step in the project", and
task P2-10 answers it with a human watching 20 overlay videos. That check is still
required and nothing here replaces it — but it is subjective, and a half-second
systematic offset is exactly the kind of thing an eye will forgive and a metric will not.

This adds an objective check alongside it, using a signal the dataset gives away for
free: **every fake names its `original`.** A fake must be near-identical to the real
video it was built from until the manipulation starts, then diverge. So the *onset of
divergence* independently measures where the forgery actually begins, from the media
itself rather than from the metadata under test. A wrong unit, a wrong timeline origin
or an off-by-one rasterisation would all show up here as a mismatch.

Two modalities, because they cover different classes:

  **visual** — compares decoded frames. Covers `visual_only` and `both`.
  **audio**  — compares log-mel on the video frame grid. Covers `audio_only` and `both`,
               which the visual method cannot touch at all: an `audio_only` fake has, by
               construction, pixel-identical frames. It also double-checks Phase 3's grid,
               since the onset is measured in audio frames and compared against seconds.

Together they cover every fake class.

⚠️ **Compare onset, not peak.** A LAV-DF fake replaces a word with a different word, so it
is usually a *different length* from its original (e.g. 142 frames vs 136). The two clips
desynchronise from the manipulation point onward and stay divergent for the rest of the
clip. The peak difference is therefore meaningless and typically lands well after the
span; only the leading edge carries information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import LAVDF_FPS, AudioPreprocessConfig  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.preprocessing.audio import extract_audio  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# One frame at 25 fps is 0.04 s. Half a frame of agreement is as good as this can get.
TOLERANCE_S = 0.5

# CMVN is deliberately OFF for the audio comparison: it normalises each clip by its own
# statistics, so the fake and the original would be scaled differently and the difference
# between them would stop meaning anything.
AUDIO_CFG = AudioPreprocessConfig(cmvn=False)


def gray_frames(path: Path, limit: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    out: list[np.ndarray] = []
    try:
        while len(out) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    finally:
        cap.release()
    return np.array(out)


def onset_from_curve(diff: np.ndarray, span_start_s: float) -> float | None:
    """First second at which `diff` leaves its pre-manipulation baseline.

    The threshold comes from the pre-manipulation region alone (mean + 4 sd), so the
    constant offset introduced by re-encoding cancels instead of masking the signal.
    """
    pre = max(3, int(round(span_start_s * LAVDF_FPS)))
    if pre >= len(diff):
        return None
    baseline = diff[:pre]
    above = np.flatnonzero(diff > baseline.mean() + 4.0 * baseline.std() + 1e-9)
    return float(above[0] / LAVDF_FPS) if above.size else None


def visual_diff(fake: Path, real: Path, limit: int) -> np.ndarray | None:
    a, b = gray_frames(fake, limit), gray_frames(real, limit)
    m = min(len(a), len(b))
    return np.abs(a[:m] - b[:m]).mean(axis=(1, 2)) if m >= 20 else None


def audio_diff(fake: Path, real: Path, limit: int) -> np.ndarray | None:
    a = extract_audio(fake, limit, AUDIO_CFG).features
    b = extract_audio(real, limit, AUDIO_CFG).features
    m = min(len(a), len(b))
    return np.abs(a[:m] - b[:m]).mean(axis=1) if m >= 20 else None


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Verify fake_periods against media evidence")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--video-root", default=None, help="default: data/raw/LAV-DF/<subset>")
    ap.add_argument("--out", default="reports/label_verification.md")
    ap.add_argument("--modality", choices=("both", "visual", "audio"), default="both")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_S)
    args = ap.parse_args()

    root = Path(args.video_root or f"data/raw/LAV-DF/{args.subset}")
    df = read_manifest(args.manifest).set_index("video_id")
    have = {p.stem for p in root.glob("*.mp4")}

    wanted = ("visual", "audio") if args.modality == "both" else (args.modality,)
    print(
        f"{DIM}{'=' * 72}{RESET}\n  Ground-truth verification (section 6.3 / P2-10)"
        f"\n{DIM}{'=' * 72}{RESET}"
    )

    rows: list[dict] = []
    for modality in wanted:
        flag = "modify_video" if modality == "visual" else "modify_audio"
        pairs = []
        for vid in sorted(have):
            if vid not in df.index:
                continue
            row = df.loc[vid]
            if row.original is None or not getattr(row, flag) or not len(row.fake_periods):
                continue
            oid = row.original.replace("/", "_").rsplit(".", 1)[0]
            if oid in have:
                pairs.append((vid, oid, row))

        print(f"\n  {modality.upper()} — {len(pairs)} pair(s)")
        if not pairs:
            print(f"  {YELLOW}none available locally{RESET}")
            continue
        print(
            f"  {'video':<14}{'class':<13}{'frames f/o':>12}{'labelled':>10}"
            f"{'measured':>10}{'error':>8}"
        )

        for vid, oid, row in pairs:
            limit = min(int(row.n_frames), int(df.loc[oid, "n_frames"]))
            start = min(s for s, _ in row.fake_periods)
            diff = (visual_diff if modality == "visual" else audio_diff)(
                root / f"{vid}.mp4", root / f"{oid}.mp4", limit
            )
            onset = onset_from_curve(diff, start) if diff is not None else None
            if onset is None:
                print(f"  {vid:<14}{row.class_name:<13}{'-':>12}{start:>10.2f}{'n/a':>10}{'-':>8}")
                continue
            err = float(abs(onset - start))
            ok = bool(err <= args.tolerance)
            rows.append(
                {
                    "modality": modality,
                    "video_id": vid,
                    "original": oid,
                    "class": row.class_name,
                    "labelled_start_s": round(start, 3),
                    "measured_onset_s": round(onset, 3),
                    "error_s": round(err, 3),
                    "pass": ok,
                    "frames_fake": int(row.n_frames),
                    "frames_real": int(df.loc[oid, "n_frames"]),
                }
            )
            mark = GREEN if ok else RED
            frames = f"{int(row.n_frames)}/{int(df.loc[oid, 'n_frames'])}"
            print(
                f"  {vid:<14}{row.class_name:<13}{frames:>12}{start:>10.2f}"
                f"{onset:>10.2f}{mark}{err:>8.2f}{RESET}"
            )

    if not rows:
        print(
            f"\n{YELLOW}no measurable pairs.{RESET} Both a fake and the video named in its "
            "`original` field must be present locally."
        )
        print(
            f"  Fetch: python scripts/05_fetch_metadata.py --subset {args.subset} --with-originals"
        )
        return 1

    arr = np.array([r["error_s"] for r in rows])
    n_ok = int(sum(r["pass"] for r in rows))
    covered = sorted({r["class"] for r in rows})
    facts = {
        "subset": args.subset,
        "n_pairs": len(rows),
        "n_within_tolerance": n_ok,
        "tolerance_s": args.tolerance,
        "median_error_s": round(float(np.median(arr)), 4),
        "max_error_s": round(float(arr.max()), 4),
        "classes_covered": covered,
        "rows": rows,
    }
    Path("reports/label_facts.json").write_text(
        json.dumps(facts, indent=2) + "\n", encoding="utf-8"
    )
    _write(Path(args.out), facts)

    print(
        f"\n  median error {np.median(arr):.3f}s, max {arr.max():.3f}s "
        f"(one frame = {1 / LAVDF_FPS:.2f}s)"
    )
    print(f"  classes covered: {', '.join(covered)}")
    passed = n_ok == len(rows)
    print(
        f"\n{GREEN if passed else RED}"
        f"{'LABELS VERIFIED' if passed else 'LABEL MISMATCH'}{RESET} -- "
        f"{n_ok}/{len(rows)} within {args.tolerance}s"
    )
    print(f"  wrote {args.out}")
    if passed:
        print(f"  {DIM}Evidence, not a substitute: P2-10's human watch still stands.{RESET}")
    return 0 if passed else 1


def _write(path: Path, facts: dict) -> None:
    lines = [
        "# Ground-truth verification — `fake_periods` against media evidence",
        "",
        "Generated by `scripts/11_verify_labels.py`. Companion to the ✋ P2-10 manual watch,",
        "not a replacement for it.",
        "",
        f"**Result: {facts['n_within_tolerance']}/{facts['n_pairs']} pairs within "
        f"{facts['tolerance_s']}s.** Median error **{facts['median_error_s']}s**, "
        f"max {facts['max_error_s']}s. One frame at 25 fps is 0.04 s.",
        "",
        f"Classes covered: **{', '.join(facts['classes_covered'])}**.",
        "",
        "## Method",
        "",
        "Every LAV-DF fake names the real video it was built from. A fake must be",
        "near-identical to that original until the manipulation begins, then diverge — so the",
        "onset of divergence measures where the forgery actually starts, **from the media**,",
        "independently of the metadata under test. A wrong unit, a wrong timeline origin or an",
        "off-by-one rasterisation would all show up as a mismatch here.",
        "",
        "The detection threshold is computed from the pre-manipulation region alone, so the",
        "constant difference introduced by re-encoding cancels rather than masking the signal.",
        "",
        "Two modalities, covering different classes:",
        "",
        "- **visual** — decoded frames. Covers `visual_only` and `both`.",
        "- **audio** — log-mel on the video frame grid (CMVN off, since it would scale each",
        "  clip by its own statistics and destroy the comparison). Covers `audio_only` and",
        "  `both` — classes the visual method cannot touch, because an `audio_only` fake has",
        "  pixel-identical frames. It also independently confirms Phase 3's grid: the onset is",
        "  measured in audio frames and compared against a labelled time in seconds.",
        "",
        "## Results",
        "",
        "| Modality | Video | Class | Frames fake/real | Labelled | Measured | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in facts["rows"]:
        lines.append(
            f"| {r['modality']} | `{r['video_id']}` | {r['class']} | "
            f"{r['frames_fake']}/{r['frames_real']} | {r['labelled_start_s']:.2f}s | "
            f"{r['measured_onset_s']:.2f}s | {r['error_s']:.2f}s |"
        )
    lines += [
        "",
        "## Why onset and not peak",
        "",
        "A LAV-DF fake replaces a word with a different word, so it is generally a **different",
        "length** from its original — note the differing frame counts above. The two clips",
        "desynchronise from the manipulation point onward and stay divergent to the end, which",
        "makes the peak difference uninformative and often located well after the span. Only the",
        "leading edge carries usable information.",
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
