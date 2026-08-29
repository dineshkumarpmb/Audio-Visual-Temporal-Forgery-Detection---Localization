"""P2-11 / P2-12 / P2-13: the Phase 2 gate checks, on real video.

    python scripts/10_verify_preprocessing.py
    python scripts/10_verify_preprocessing.py --n 5

Three things the unit tests cannot prove, because they need actual decoding and actual
MediaPipe inference:

  **P2-11 determinism.** Extract the same videos twice into separate cache roots and
  compare the arrays byte for byte. Anything non-deterministic on this path — a RANSAC
  solver, an unseeded shuffle, a thread-order-dependent reduction — shows up here and
  nowhere else, and it would quietly invalidate every Part 7 comparison.

  **P2-12 resumability.** Delete part of a cache, re-run, and confirm the survivors are
  skipped and untouched while the missing ones are rebuilt identically. R13/X-4 assume
  interruption is routine on this machine; "it should resume" is not evidence.

  **P2-13 face_found.** Mean detection rate over the sample, against the 0.9 floor.

Writes reports/preprocessing_verification.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import PreprocessConfig, config_hash  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.preprocessing.cache import cache_path, is_cached, load_faces  # noqa: E402
from src.preprocessing.face import FaceLandmarkerPool  # noqa: E402
from src.preprocessing.pipeline import extract_video  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
FOUND_THRESHOLD = 0.9


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


def extract_all(ids, manifest, video_root: Path, cfg, root: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with FaceLandmarkerPool() as lm:
        for vid in ids:
            res = extract_video(
                video_root / f"{vid}.mp4",
                vid,
                int(manifest.loc[vid, "n_frames"]),
                cfg,
                lm,
                root=root,
            )
            if res.error:
                raise RuntimeError(f"{vid}: {res.error}")
            out[vid] = res.found_fraction
    return out


def payload(vid: str, cfg, root: Path) -> bytes:
    """Raw bytes of the cached arrays — what byte-identical actually means."""
    crops, found = load_faces(vid, cfg, root)
    return np.asarray(crops).tobytes() + np.asarray(found).tobytes()


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Phase 2 gate checks on real video")
    ap.add_argument("--subset", default="smoke-100")
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--work-root", default="data/interim/_verify")
    ap.add_argument("--out", default="reports/preprocessing_verification.md")
    ap.add_argument("--n", type=int, default=6, help="videos to use (each is extracted twice)")
    args = ap.parse_args()

    cfg = PreprocessConfig()
    video_root = Path(args.video_root or f"data/raw/LAV-DF/{args.subset}")
    manifest = read_manifest(args.manifest).set_index("video_id")
    ids = [
        v for v in load_subset(args.subset, args.subset_dir) if (video_root / f"{v}.mp4").exists()
    ][: args.n]

    if not ids:
        print(f"{RED}no videos under {video_root}{RESET}")
        return 1

    work = Path(args.work_root)
    root_a, root_b = work / "a", work / "b"
    for r in (root_a, root_b):
        if r.exists():
            shutil.rmtree(r)

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Phase 2 gate -- {len(ids)} videos, config {config_hash(cfg)}"
        f"\n{DIM}{'=' * 72}{RESET}"
    )
    c = Checks()
    facts: dict[str, object] = {
        "subset": args.subset,
        "n_videos": len(ids),
        "config_hash": config_hash(cfg),
    }

    # ---- P2-11 determinism ------------------------------------------------------
    print(f"\n{DIM}  P2-11 determinism: extracting twice{RESET}")
    t0 = time.time()
    found_a = extract_all(ids, manifest, video_root, cfg, root_a)
    t_first = time.time() - t0
    found_b = extract_all(ids, manifest, video_root, cfg, root_b)

    mismatched = [v for v in ids if payload(v, cfg, root_a) != payload(v, cfg, root_b)]
    c.add(
        "byte-identical across two runs",
        not mismatched,
        f"{len(ids) - len(mismatched)}/{len(ids)} identical"
        + (f", differing: {mismatched[:3]}" if mismatched else ""),
    )
    c.add(
        "face_found identical across two runs",
        all(found_a[v] == found_b[v] for v in ids),
        f"{len(ids)} videos",
    )
    facts["determinism_identical"] = len(ids) - len(mismatched)

    # ---- P2-12 resumability -----------------------------------------------------
    print(f"\n{DIM}  P2-12 resumability: deleting half a cache and re-running{RESET}")
    victims = ids[: max(1, len(ids) // 2)]
    survivors = [v for v in ids if v not in victims]
    before = {v: cache_path(v, cfg, root_a).stat().st_mtime_ns for v in survivors}
    checksums = {v: payload(v, cfg, root_a) for v in victims}
    for v in victims:
        cache_path(v, cfg, root_a).unlink()

    c.add(
        "deleted entries report as not cached",
        all(not is_cached(v, cfg, root_a) for v in victims),
        f"{len(victims)} deleted",
    )

    t0 = time.time()
    extract_all(ids, manifest, video_root, cfg, root_a)
    t_resume = time.time() - t0

    c.add(
        "deleted entries are rebuilt",
        all(is_cached(v, cfg, root_a) for v in victims),
        f"{len(victims)} rebuilt",
    )
    c.add(
        "rebuilt entries are byte-identical to the originals",
        all(payload(v, cfg, root_a) == checksums[v] for v in victims),
        f"{len(victims)} verified",
    )
    c.add(
        "surviving entries were not rewritten",
        all(cache_path(v, cfg, root_a).stat().st_mtime_ns == before[v] for v in survivors),
        f"{len(survivors)} untouched (mtime unchanged)",
    )
    c.add(
        "resume is faster than a full run",
        t_resume < t_first,
        f"{t_resume:.1f}s resume vs {t_first:.1f}s full",
    )
    facts["full_run_s"] = round(t_first, 2)
    facts["resume_run_s"] = round(t_resume, 2)

    # ---- P2-13 face_found -------------------------------------------------------
    print(f"\n{DIM}  P2-13 detection rate{RESET}")
    mean = float(np.mean(list(found_a.values())))
    facts["face_found_mean"] = round(mean, 4)
    facts["face_found_min"] = round(min(found_a.values()), 4)
    c.add(
        f"mean face_found >= {FOUND_THRESHOLD}",
        mean >= FOUND_THRESHOLD,
        f"{mean:.3f} (min {min(found_a.values()):.3f})",
    )

    shutil.rmtree(work, ignore_errors=True)

    _write(Path(args.out), c, facts, ids)
    Path("reports/preprocessing_facts.json").write_text(
        json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(f"\n{DIM}{'-' * 72}{RESET}")
    passed = len(c.rows) - len(c.failed)
    print(f"  {passed}/{len(c.rows)} checks passed")
    if c.failed:
        print(f"\n{RED}PHASE 2 GATE FAILED{RESET} -- {c.failed}")
        return 1
    print(f"\n{GREEN}PHASE 2 GATE PASSED{RESET} -- determinism, resumability and detection rate")
    print(f"  wrote {args.out}")
    print(f"  {YELLOW}Still manual: P2-8 (contact sheets) and ⛔ P2-10 (watch 20 overlays).{RESET}")
    return 0


def _write(path: Path, c: Checks, facts: dict, ids: list[str]) -> None:
    lines = [
        "# Phase 2 — preprocessing verification (P2-11, P2-12, P2-13)",
        "",
        "Generated by `scripts/10_verify_preprocessing.py` on real video.",
        "",
        f"**Result: {len(c.rows) - len(c.failed)}/{len(c.rows)} checks passed.** "
        f"Sample: {facts['n_videos']} videos from `{facts['subset']}`, "
        f"config `{facts['config_hash']}`.",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in c.rows:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += [
        "",
        "## What determinism means here",
        "",
        "The two runs write to separate cache roots and the **raw array bytes** are compared,",
        "not a summary statistic. Every step on this path is analytic: MediaPipe CPU inference,",
        "the closed-form Umeyama similarity solve, and `cv2.warpAffine`. `estimateAffinePartial2D`",
        "was deliberately not used — its RANSAC default samples randomly and would break this.",
        "",
        "## What resumability means here",
        "",
        f"Half the cache ({', '.join(ids[: max(1, len(ids) // 2)][:3])}...) is deleted and the run",
        "repeated. The check is not merely that the run completes: rebuilt entries must be",
        "byte-identical to what was deleted, and survivors must keep their original mtime,",
        "proving they were skipped rather than silently rewritten.",
        "",
        f"Full run {facts['full_run_s']}s vs resume {facts['resume_run_s']}s.",
        "",
        "## Still manual",
        "",
        "- **P2-8** — open `reports/figures/contact_sheets/` and confirm faces are upright,",
        "  eyes level, one identity per clip.",
        "- **⛔ P2-10** — watch `reports/figures/overlays/` and confirm the red band lands on",
        "  visibly manipulated speech. Nothing automated can substitute for this.",
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
