"""Is the audio of a `visual_only` fake identical to its original's audio? (PF-19)

    python scripts/21_check_audio_provenance.py --n 8

Phase 5's log-Mel arm scores **0.9728** on `visual_only` forgeries — clips whose audio the
dataset labels as *unmodified*. An audio-only model should be at chance there, and MFCC is
(0.5365) while seeing identical clip lengths, so the clip-length shortcut cannot explain
it. That leaves one hypothesis worth testing rather than asserting: the audio of a
`visual_only` fake is not bit-for-bit the original's audio, because the clip was rebuilt,
and log-Mel is reading a **processing fingerprint** rather than a manipulation.

This settles it by comparing each fake against the real video its metadata names in
`original`, on the waveform the rest of the pipeline actually consumes.

⚠️ **Divergence *onset* is the informative quantity, not overall difference.** A fake is
generally a different length from its original (Phase 2's note), so an edit re-times
everything after it and a whole-prefix comparison is guaranteed to look different whatever
the answer is. This mirrors Phase 2's method exactly: find the first sample where the two
waveforms part company, and compare that instant against the labelled span.

  * onset at **t = 0** -> the stream is not a faithful copy anywhere; a global,
    label-correlated difference is available, and an audio model can score on `visual_only`
    without reading any manipulation. This test does not distinguish *why* (re-encode,
    resample, global offset) -- only that the difference is global rather than local.
  * onset at **the labelled edit** -> the audio is a faithful copy up to the edit, so only
    a boundary artefact exists and the high score needs another explanation.

Downloads the handful of `original` videos it needs (a few MB, PF-16's paced fetcher).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import AudioPreprocessConfig  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import load_subset  # noqa: E402
from src.preprocessing.audio import decode_audio  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
ORIGINALS = Path("data/raw/LAV-DF/_originals")


def _fetcher():
    """Reuse `scripts/05_fetch_metadata.py`; a digit-leading module needs a manual import."""
    spec = importlib.util.spec_from_file_location(
        "f05", REPO_ROOT / "scripts" / "05_fetch_metadata.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Compare visual_only audio against its original")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="reports/audio_provenance.json")
    args = ap.parse_args()

    print(
        f"{DIM}{'=' * 72}{RESET}\n  PF-19  audio provenance of `visual_only` fakes"
        f"\n{DIM}{'=' * 72}{RESET}"
    )

    manifest = read_manifest("data/manifests/manifest_v1.parquet").set_index("video_id")
    ids = set(load_subset("dev-2k", "data/manifests/subsets"))
    ids |= set(load_subset("smoke-100", "data/manifests/subsets"))
    local = {
        p.stem: p
        for d in ("dev-2k", "smoke-100")
        for p in Path(f"data/raw/LAV-DF/{d}").glob("*.mp4")
    }

    rows = manifest.loc[sorted(ids & set(manifest.index))]
    vo = rows[(rows.modify_video) & (~rows.modify_audio)]
    # `original` is a dataset-relative member path like "test/000001.mp4"; the id form
    # replaces the separator. Taking only the stem would drop the split and 404.
    pairs = [
        (v, r.original)
        for v, r in vo.iterrows()
        if isinstance(r.original, str) and r.original and v in local
    ][: args.n]
    if not pairs:
        print(f"{RED}no local visual_only clips with a named original{RESET}")
        return 1

    fetch = _fetcher()
    auth = fetch.credentials()
    fetch.THROTTLE = fetch._Throttle(25.0)
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {len(pairs)} originals -> {ORIGINALS}")
    for _, member in pairs:
        dest = ORIGINALS / member.replace("/", "_")
        if dest.exists():
            continue
        try:
            dest.write_bytes(fetch.fetch(f"LAV-DF/{member}", auth))
        except Exception as exc:  # noqa: BLE001
            print(f"    {YELLOW}skip {member}: {type(exc).__name__}{RESET}")

    cfg = AudioPreprocessConfig()
    # Loud enough to be real divergence rather than codec dither on a [-1, 1] waveform.
    tol = 1e-4
    results = []
    print(
        f"\n  {'fake':<14}{'original':<14}{'onset s':>9}{'label s':>9}"
        f"{'identical pfx':>14}{'verdict':>12}"
    )
    for fake, member in pairs:
        orig = member.replace("/", "_").removesuffix(".mp4")
        op = ORIGINALS / member.replace("/", "_")
        if not op.exists():
            continue
        wf = decode_audio(local[fake], cfg)
        wo = decode_audio(op, cfg)
        n = min(len(wf), len(wo))
        if n == 0:
            continue
        d = np.abs(wf[:n] - wo[:n])
        above = np.flatnonzero(d > tol)
        onset = float(above[0]) / cfg.sample_rate if above.size else float(n) / cfg.sample_rate
        periods = manifest.loc[fake, "fake_periods"]
        label = float(min(p[0] for p in periods)) if len(periods) else float("nan")
        # "Re-encoded from the start" is the finding; a handful of samples is codec warm-up.
        from_start = onset < 0.05
        results.append(
            {
                "fake": fake,
                "original": orig,
                "n_fake": int(len(wf)),
                "n_original": int(len(wo)),
                "divergence_onset_s": onset,
                "labelled_span_start_s": label,
                "identical_prefix_samples": int(above[0]) if above.size else int(n),
                "diverges_from_start": bool(from_start),
            }
        )
        mark = RED if from_start else GREEN
        verdict = "rebuilt" if from_start else "copy->edit"
        print(
            f"  {fake:<14}{orig:<14}{onset:>9.3f}{label:>9.3f}"
            f"{(int(above[0]) if above.size else n):>14}{mark}{verdict:>12}{RESET}"
        )

    if not results:
        print(f"{RED}no comparable pairs -- originals could not be fetched{RESET}")
        return 1

    n_from_start = sum(r["diverges_from_start"] for r in results)
    onsets = np.array([r["divergence_onset_s"] for r in results])
    labels = np.array([r["labelled_span_start_s"] for r in results])
    verdict = (
        f"{n_from_start}/{len(results)} diverge from t=0: the whole audio stream is rebuilt, so a "
        "global processing fingerprint is available to any model reading fine spectral structure"
        if n_from_start > len(results) / 2
        else f"{len(results) - n_from_start}/{len(results)} are a faithful copy up to the labelled "
        "edit, so only a boundary artefact exists -- the visual_only score needs another explanation"
    )
    print(f"\n  diverging from t=0 : {n_from_start}/{len(results)}")
    print(f"  median onset       : {np.median(onsets):.3f} s")
    print(f"  median label start : {np.nanmedian(labels):.3f} s")
    print(f"  {(YELLOW if n_from_start else GREEN)}{verdict}{RESET}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "n_pairs": len(results),
                "n_diverging_from_start": n_from_start,
                "median_onset_s": float(np.median(onsets)),
                "median_labelled_start_s": float(np.nanmedian(labels)),
                "tolerance": tol,
                "verdict": verdict,
                "pairs": results,
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
