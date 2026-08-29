"""P1-15/P1-16: build the stratified development subsets and commit their id lists.

    python scripts/03_make_subset.py                 # smoke-100, dev-2k, dev-10k
    python scripts/03_make_subset.py --name dev-2k
    python scripts/03_make_subset.py --verify        # re-derive and compare fingerprints

The id lists in data/manifests/subsets/ are committed to git on purpose (P1-16): every
experiment must run on literally the same videos, and a fingerprint mismatch is how you
find out that it did not. `--verify` is the CI form of that check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.data.subset import (  # noqa: E402
    SUBSET_SIZES,
    load_subset,
    make_subset,
    subset_fingerprint,
    write_subset,
)
from src.seed import seed_everything  # noqa: E402
from src.utils.console import init_console  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Build reproducible LAV-DF development subsets")
    ap.add_argument("--manifest", default="data/manifests/manifest_v1.parquet")
    ap.add_argument("--out-dir", default="data/manifests/subsets")
    ap.add_argument("--name", default=None, help=f"one of {sorted(SUBSET_SIZES)}; default: all")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--verify", action="store_true", help="compare against committed id lists")
    args = ap.parse_args()

    seed_everything(args.seed)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"{RED}manifest not found: {manifest}{RESET}  -- run scripts/02_build_manifest.py")
        return 1

    df = read_manifest(manifest)
    names = [args.name] if args.name else list(SUBSET_SIZES)

    print(
        f"{DIM}{'=' * 72}{RESET}\n  P1-15  Development subsets (seed={args.seed})\n{DIM}{'=' * 72}{RESET}"
    )
    failures = 0

    for name in names:
        sub = make_subset(df, name, seed=args.seed)
        ids = sorted(sub["video_id"].tolist())
        fp = subset_fingerprint(ids)

        if args.verify:
            try:
                committed = sorted(load_subset(name, args.out_dir))
            except FileNotFoundError:
                print(f"  [{RED}FAIL{RESET}] {name}: no committed id list to verify against")
                failures += 1
                continue
            if committed != ids:
                print(
                    f"  [{RED}FAIL{RESET}] {name}: differs from the committed list "
                    f"({subset_fingerprint(committed)} != {fp})"
                )
                failures += 1
                continue
            print(f"  [{GREEN}OK{RESET}] {name:<10} matches committed list  {DIM}fp={fp}{RESET}")
            continue

        write_subset(sub, name, args.out_dir, args.seed)
        splits = sub["split"].value_counts().sort_index().to_dict()
        classes = sub["class_name"].value_counts().sort_index().to_dict()
        print(f"  [{GREEN}OK{RESET}] {name:<10} n={len(sub):>6,}  {DIM}fp={fp}{RESET}")
        print(f"       split   {json.dumps({k: int(v) for k, v in splits.items()})}")
        print(f"       class   {json.dumps({k: int(v) for k, v in classes.items()})}")

    if failures:
        print(f"\n{RED}SUBSET VERIFY FAILED{RESET} -- {failures} subset(s) drifted from git")
        return 1
    print(
        f"\n{GREEN}{'SUBSETS VERIFIED' if args.verify else 'SUBSETS WRITTEN'}{RESET}"
        f" -- commit data/manifests/subsets/ (P1-16)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
