"""P1-6/P1-8/P1-9: build and validate `manifest_v1.parquet`.

    python scripts/02_build_manifest.py
    python scripts/02_build_manifest.py --probe          # on Kaggle, where video exists
    python scripts/02_build_manifest.py --version v2     # never overwrite v1

Exits non-zero on any leakage or schema violation, so this is usable as a CI gate
in the same way `make check` is.

`--probe` runs ffprobe on every file. Off by default: decision PF-6 keeps the 25 GB
of video on Kaggle, so on this machine there is nothing to probe and the metadata-only
manifest is the correct artefact. The Kaggle notebook (CL-3) runs it with --probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.leakage import LeakageError  # noqa: E402
from src.data.manifest import build_manifest, validate_manifest, write_manifest  # noqa: E402
from src.data.metadata import (  # noqa: E402
    assert_fake_periods_are_seconds,
    assert_frame_timeline,
    assert_n_fakes_matches,
    load_metadata,
)
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Build manifest_v1.parquet from LAV-DF metadata")
    ap.add_argument("--data-root", default="data/raw/LAV-DF", help="dir holding train/ dev/ test/")
    ap.add_argument("--metadata", default=None, help="default: <data-root>/metadata.min.json")
    ap.add_argument("--out-dir", default="data/manifests")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--probe", action="store_true", help="run ffprobe on every file (slow)")
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing manifest")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    metadata_path = Path(args.metadata) if args.metadata else data_root / "metadata.min.json"
    if not metadata_path.exists():
        print(f"{RED}metadata not found: {metadata_path}{RESET}")
        print("  Per PF-6 the raw dataset lives on Kaggle. Fetch just the metadata with:")
        print("    python scripts/05_fetch_metadata.py")
        return 1

    print(f"{DIM}{'=' * 72}{RESET}\n  P1-6  Build manifest_{args.version}\n{DIM}{'=' * 72}{RESET}")
    print(f"  metadata : {metadata_path}")
    print(f"  probe    : {'yes (ffprobe every file)' if args.probe else 'no (metadata only)'}")

    entries = load_metadata(metadata_path)
    print(f"\n  [{GREEN}OK{RESET}] parsed {len(entries):,} entries")

    # ---- P1-5: the assertions that must never become warnings -------------------
    evidence = assert_fake_periods_are_seconds(entries)
    print(
        f"  [{GREEN}OK{RESET}] fake_periods are SECONDS  {DIM}max_end={evidence['max_end']} "
        f"max_duration={evidence['max_duration']} integral_ends={evidence['all_ends_integral']}{RESET}"
    )
    assert_n_fakes_matches(entries)
    print(f"  [{GREEN}OK{RESET}] n_fakes == len(fake_periods)")

    tl = assert_frame_timeline(entries)
    print(
        f"  [{GREEN}OK{RESET}] timeline  {DIM}duration==audio/16k+0.128 on "
        f"{tl['duration_equals_padded_audio']:,}/{tl['n']:,}; video_frames within +/-1 of "
        f"duration*25 on only {tl['video_frames_within_1_of_duration_x_fps']:,}{RESET}"
    )
    if evidence["n_out_of_bounds"]:
        print(
            f"  [{YELLOW}WARN{RESET}] {evidence['n_out_of_bounds']} entries with spans past "
            f"`duration` -> quarantined bad_label: {evidence['out_of_bounds_files']}"
        )

    # ---- build + validate --------------------------------------------------------
    print(f"\n{DIM}  building...{RESET}")
    df = build_manifest(entries, data_root, probe=args.probe)

    try:
        report = validate_manifest(df)
    except LeakageError as exc:
        print(f"\n  [{RED}FAIL{RESET}] {exc}")
        return 1

    print(
        f"  [{GREEN}OK{RESET}] no split leakage  {DIM}train^dev={report['leakage']['train^dev']} "
        f"train^test={report['leakage']['train^test']} dev^test={report['leakage']['dev^test']}{RESET}"
    )
    print(
        f"  [{GREEN}OK{RESET}] {report['n_ok']:,}/{report['n_total']:,} usable "
        f"({report['exclusion_rate']:.4%} excluded)"
    )
    for status, n in report["status_counts"].items():
        if n and status != "ok":
            print(f"       {status:<12} {n:>6,}")

    out = Path(args.out_dir) / f"manifest_{args.version}.parquet"
    if out.exists() and args.force:
        out.unlink()
    try:
        write_manifest(df, out)
    except FileExistsError as exc:
        print(f"\n  [{RED}FAIL{RESET}] {exc}")
        return 1

    report_path = Path(args.out_dir) / f"manifest_{args.version}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\n  wrote {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
    print(f"  wrote {report_path}")
    print(f"\n{GREEN}MANIFEST OK{RESET} -- next: scripts/03_make_subset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
