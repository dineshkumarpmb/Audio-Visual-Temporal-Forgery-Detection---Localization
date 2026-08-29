"""Phase 3: extract log-mel (or MFCC) on the video frame grid (P3-1 .. P3-7, P3-9).

    python scripts/12_extract_audio.py --subset smoke-100
    python scripts/12_extract_audio.py --subset smoke-100 --feature mfcc
    python scripts/12_extract_audio.py --subset smoke-100 --force

⛔ **This is the P3-9 gate**, and it is strict: `audio_frames == video_frames ± 1` must
hold on **100%** of files, with no rounding fudge. The script exits non-zero on a single
violation. Section 3's Phase 3 note is the reason: "every alignment bug you let through
here becomes an unexplainable localization failure in Phase 10."

Resumable and content-hashed on the same terms as Phase 2, but keyed on
`AudioPreprocessConfig` alone, so switching `--feature mfcc` for Experiment K builds a
separate cache instead of invalidating the face crops.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioPreprocessConfig, config_hash  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.preprocessing.audio import AudioError, assert_aligned, extract_audio  # noqa: E402
from src.preprocessing.cache import cache_root, is_audio_cached, save_audio  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Extract audio features on the video grid")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--video-root", default=None, help="default: data/raw/LAV-DF/<subset>")
    ap.add_argument("--out-root", default="data/interim/audio")
    ap.add_argument("--feature", choices=("logmel", "mfcc"), default="logmel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = AudioPreprocessConfig(feature=args.feature)
    video_root = Path(args.video_root or f"data/raw/LAV-DF/{args.subset}")

    ids = load_subset(args.subset, args.subset_dir)
    if args.limit:
        ids = ids[: args.limit]
    manifest = read_manifest(args.manifest).set_index("video_id")
    present = [v for v in ids if (video_root / f"{v}.mp4").exists()]

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 3  Audio features -- {args.subset}\n{DIM}{'=' * 72}{RESET}"
    )
    print(
        f"  config  : {config_hash(cfg)}  {DIM}{cfg.feature} sr={cfg.sample_rate} "
        f"n_fft={cfg.n_fft} hop={cfg.hop_length} ({cfg.hop_ms:.0f}ms = 1 video frame) "
        f"n_mels={cfg.n_mels}{RESET}"
    )
    print(f"  videos  : {len(present)} of {len(ids)} present")
    if not present:
        print(
            f"\n{RED}nothing to do{RESET} -- fetch with scripts/05_fetch_metadata.py "
            f"--subset {args.subset}"
        )
        return 1

    start = time.time()
    violations: list[str] = []
    failures: list[tuple[str, str]] = []
    silent: list[str] = []
    pads: list[float] = []
    fresh = skipped = 0

    for i, vid in enumerate(present, 1):
        if not args.force and is_audio_cached(vid, cfg, args.out_root):
            skipped += 1
            continue
        n_video = int(manifest.loc[vid, "n_frames"])
        try:
            res = extract_audio(video_root / f"{vid}.mp4", n_video, cfg)
            assert_aligned(res.n_frames, n_video, vid)  # the P3-9 gate, per file
        except AudioError as exc:
            failures.append((vid, str(exc)[:120]))
            if "audio frames vs" in str(exc):
                violations.append(vid)
            continue
        if res.is_silent:
            silent.append(vid)  # P3-7
        pads.append(res.pad_frames)
        save_audio(vid, res.features, cfg, args.out_root)
        fresh += 1
        if i % 25 == 0 or i == len(present):
            print(
                f"    {i}/{len(present)}  {DIM}{i / (time.time() - start):.1f} videos/s{RESET}",
                flush=True,
            )

    elapsed = time.time() - start
    root = cache_root(cfg, args.out_root)
    size = sum(f.stat().st_size for f in root.glob("*.npy")) if root.exists() else 0

    print(f"\n{DIM}{'-' * 72}{RESET}")
    print(f"  extracted {fresh}, skipped {skipped} cached, failed {len(failures)}")
    print(f"  wall-clock {elapsed:.1f}s" + (f"  ({elapsed / fresh:.3f}s/video)" if fresh else ""))
    print(f"  cache {root}  {len(list(root.glob('*.npy')))} files, {size / 1024**2:.1f} MB")

    for vid, err in failures[:8]:
        print(f"  [{RED}FAIL{RESET}] {vid}: {err}")
    if silent:
        print(f"  [{YELLOW}no_audio{RESET}] {len(silent)} silent waveform(s): {silent[:5]}")

    ok = True
    print("\n  ⛔ P3-9 alignment gate (audio_frames == video_frames +/-1, 100% required)")
    checked = fresh + skipped
    if violations:
        print(f"  [{RED}FAIL{RESET}] {len(violations)} alignment violation(s): {violations[:8]}")
        ok = False
    elif fresh == 0 and skipped:
        print(f"  [{DIM}skip{RESET}] all {skipped} cached; re-run with --force to re-verify")
    else:
        print(f"  [{GREEN}PASS{RESET}] {fresh}/{fresh} extracted files aligned exactly")
    if failures and not violations:
        ok = False

    if pads:
        arr = np.array(pads)
        print(
            f"\n  waveform fit: mean {arr.mean():+.2f} frames, range "
            f"[{arr.min():+.2f}, {arr.max():+.2f}]  {DIM}(negative = trimmed){RESET}"
        )
        facts = {
            "subset": args.subset,
            "config_hash": config_hash(cfg),
            "config": cfg.model_dump(mode="json"),
            "n_extracted": fresh,
            "n_failed": len(failures),
            "n_silent": len(silent),
            "n_alignment_violations": len(violations),
            "n_checked": checked,
            "wall_clock_s": round(elapsed, 2),
            "seconds_per_video": round(elapsed / fresh, 4) if fresh else None,
            "pad_frames_mean": round(float(arr.mean()), 3),
            "pad_frames_min": round(float(arr.min()), 3),
            "pad_frames_max": round(float(arr.max()), 3),
            "cache_bytes": size,
        }
        out = Path(f"reports/audio_facts_{cfg.feature}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out}")

    print(f"\n{GREEN if ok else RED}{'AUDIO OK' if ok else 'AUDIO GATE FAILED'}{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
