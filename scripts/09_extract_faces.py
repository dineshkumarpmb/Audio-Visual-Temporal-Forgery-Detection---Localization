"""Phase 2: extract aligned face crops for a subset (P2-1 .. P2-6, P2-13).

    python scripts/09_extract_faces.py --subset smoke-100
    python scripts/09_extract_faces.py --subset smoke-100 --workers 2
    python scripts/09_extract_faces.py --subset smoke-100 --limit 5 --force

Resumable by construction (P2-5): a video whose `.npz` already exists is skipped, so
an interrupted run resumes by simply being re-run. R13/X-4 assume interruption is the
normal case on this machine, and an 8-hour extraction here will be interrupted.

`--workers` defaults to 2, the cap PROJECT_PLAN section 12.2 sets for a 6 GB box with
~0.3 GB free at rest. Each worker owns its own MediaPipe landmarker; the model is not
thread-safe and is cheap enough to instantiate per process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import VideoPreprocessConfig, config_hash  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.preprocessing.cache import cache_stats  # noqa: E402
from src.preprocessing.face import FaceLandmarkerPool  # noqa: E402
from src.preprocessing.pipeline import extract_video  # noqa: E402
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# P2-13's acceptance threshold: a clean sample should find a face on >90% of frames.
FOUND_THRESHOLD = 0.9


def _job(args: tuple) -> dict:
    """One video, in a worker process. Returns a plain dict so it pickles cheaply."""
    video_id, path, n_frames, cfg_json, root, force = args
    cfg = VideoPreprocessConfig.model_validate_json(cfg_json)
    global _LANDMARKER  # noqa: PLW0603 - one model per process, reused across videos
    try:
        landmarker = _LANDMARKER
    except NameError:
        landmarker = _LANDMARKER = FaceLandmarkerPool()
    res = extract_video(path, video_id, n_frames, cfg, landmarker, root=root, force=force)
    return {
        "video_id": res.video_id,
        "n_frames": res.n_frames,
        "found_fraction": res.found_fraction,
        "cached": res.cached,
        "error": res.error,
    }


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Extract aligned face crops (Phase 2)")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--video-root", default=None, help="default: data/raw/LAV-DF/<subset>")
    ap.add_argument("--out-root", default="data/interim/faces")
    ap.add_argument("--workers", type=int, default=2, help="section 12.2 caps this at 2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = VideoPreprocessConfig()
    video_root = Path(args.video_root or f"data/raw/LAV-DF/{args.subset}")

    ids = load_subset(args.subset, args.subset_dir)
    if args.limit:
        ids = ids[: args.limit]
    manifest = read_manifest(args.manifest).set_index("video_id")

    present = [v for v in ids if (video_root / f"{v}.mp4").exists()]
    missing = len(ids) - len(present)

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 2  Face extraction -- {args.subset}\n{DIM}{'=' * 72}{RESET}"
    )
    print(
        f"  config    : {config_hash(cfg)}  {DIM}crop={cfg.crop_size} margin={cfg.face_margin} "
        f"every={cfg.detect_every_n} align={cfg.align}{RESET}"
    )
    print(
        f"  videos    : {len(present)} present"
        + (f", {YELLOW}{missing} missing{RESET}" if missing else "")
    )
    print(f"  workers   : {args.workers}")
    if missing:
        print(
            f"  {DIM}fetch them with: python scripts/05_fetch_metadata.py --subset {args.subset}{RESET}"
        )
    if not present:
        print(f"\n{RED}nothing to do{RESET}")
        return 1

    jobs = [
        (
            v,
            str(video_root / f"{v}.mp4"),
            int(manifest.loc[v, "n_frames"]),
            cfg.model_dump_json(),
            args.out_root,
            args.force,
        )
        for v in present
    ]

    start = time.time()
    results: list[dict] = []
    if args.workers <= 1:
        for j in jobs:
            results.append(_job(j))
            _tick(len(results), len(jobs), start)
    else:
        import multiprocessing as mp

        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_job, jobs, chunksize=1):
                results.append(r)
                _tick(len(results), len(jobs), start)

    elapsed = time.time() - start
    return _report(results, cfg, args, elapsed)


def _tick(done: int, total: int, start: float) -> None:
    if done % 10 == 0 or done == total:
        rate = done / max(time.time() - start, 1e-9)
        print(f"    {done}/{total}  {DIM}{rate:.2f} videos/s{RESET}", flush=True)


def _report(results: list[dict], cfg: VideoPreprocessConfig, args, elapsed: float) -> int:
    failed = [r for r in results if r["error"]]
    fresh = [r for r in results if not r["cached"] and not r["error"]]
    skipped = [r for r in results if r["cached"]]
    stats = cache_stats(cfg, args.out_root)

    print(f"\n{DIM}{'-' * 72}{RESET}")
    print(f"  extracted {len(fresh)}, skipped {len(skipped)} cached, failed {len(failed)}")
    print(
        f"  wall-clock {elapsed:.1f}s" + (f"  ({elapsed / len(fresh):.2f}s/video)" if fresh else "")
    )
    print(f"  cache {stats['dir']}  {stats['n']} files, {stats['bytes'] / 1024**2:.1f} MB")

    for r in failed[:10]:
        print(f"  [{RED}FAIL{RESET}] {r['video_id']}: {r['error'][:110]}")

    ok = True
    if fresh:
        fracs = sorted(r["found_fraction"] for r in fresh)
        mean = sum(fracs) / len(fracs)
        below = [r for r in fresh if r["found_fraction"] < FOUND_THRESHOLD]
        print(
            f"\n  P2-13 face_found: mean {mean:.3f}, median {fracs[len(fracs) // 2]:.3f}, "
            f"min {fracs[0]:.3f}"
        )
        if mean >= FOUND_THRESHOLD:
            print(f"  [{GREEN}PASS{RESET}] mean face_found {mean:.3f} >= {FOUND_THRESHOLD}")
        else:
            print(f"  [{RED}FAIL{RESET}] mean face_found {mean:.3f} < {FOUND_THRESHOLD}")
            ok = False
        if below:
            print(
                f"  {YELLOW}{len(below)} video(s) below threshold{RESET}: "
                + ", ".join(f"{r['video_id']}={r['found_fraction']:.2f}" for r in below[:8])
            )

        facts = {
            "subset": args.subset,
            "config_hash": config_hash(cfg),
            "config": cfg.model_dump(mode="json"),
            "n_extracted": len(fresh),
            "n_failed": len(failed),
            "wall_clock_s": round(elapsed, 2),
            "seconds_per_video": round(elapsed / len(fresh), 3),
            "face_found_mean": round(mean, 4),
            "face_found_min": round(fracs[0], 4),
            "cache_bytes": stats["bytes"],
        }
        out = Path("reports/extraction_facts.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {out}")

    if failed:
        ok = False
    print(f"\n{GREEN if ok else RED}{'EXTRACTION OK' if ok else 'EXTRACTION HAD FAILURES'}{RESET}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
