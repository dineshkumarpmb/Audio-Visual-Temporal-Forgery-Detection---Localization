"""Fetch LAV-DF metadata (and an optional video sample) from the Kaggle mirror.

    python scripts/05_fetch_metadata.py
    python scripts/05_fetch_metadata.py --spotcheck 28

Decision PF-6 keeps the 25.6 GB of video on Kaggle. This pulls only what genuinely
has to be local:

  * `metadata.min.json` (2.6 MB compressed / 33.8 MB raw) -- the manifest is built
    from it, and it is the artefact P1-5's assertions run against.
  * `README.md` (2.3 KB) -- the authors' own field documentation.
  * with `--spotcheck N`, a stratified handful of videos for the ffprobe checks that
    metadata alone cannot answer (P1-13, and CL-7's re-encode test).

Why raw HTTP rather than the kaggle client: `KaggleApi.dataset_download_file()`
404s on this dataset for every path form tried. The documented `?file_name=` query
parameter works, and Kaggle returns each single-file download wrapped in a zip, which
is unwrapped here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.console import init_console  # noqa: E402

MIRROR = "elin75/localized-audio-visual-deepfake-dataset-lav-df"
BASE = "https://www.kaggle.com/api/v1/datasets/download"

# Measured from the mirror's own file listing on 2026-08-29. A mismatch means the
# mirror changed under us and CL-7 must be re-run before anything downstream is trusted.
EXPECTED = {
    "LAV-DF/metadata.min.json": 33_837_990,
    "LAV-DF/README.md": 2_324,
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def credentials() -> tuple[str, str]:
    import os

    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not token.exists():
        raise SystemExit(f"no Kaggle credentials: set KAGGLE_USERNAME/KAGGLE_KEY or {token}")
    c = json.loads(token.read_text(encoding="utf-8"))
    return c["username"], c["key"]


def fetch(member: str, auth: tuple[str, str], timeout: int = 900) -> bytes:
    """Download one dataset member, unwrapping Kaggle's zip envelope."""
    import requests

    r = requests.get(f"{BASE}/{MIRROR}?file_name={quote(member)}", auth=auth, timeout=timeout)
    r.raise_for_status()
    if r.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            return z.read(z.namelist()[0])
    return r.content


def main() -> int:
    init_console()
    ap = argparse.ArgumentParser(description="Fetch LAV-DF metadata from the Kaggle mirror")
    ap.add_argument("--out", default="data/raw/LAV-DF")
    ap.add_argument("--spotcheck", type=int, default=0, help="also fetch N videos for ffprobe")
    ap.add_argument("--spotcheck-dir", default="data/raw/_spotcheck")
    ap.add_argument(
        "--subset",
        default=None,
        help="also fetch every video in a committed subset list, e.g. smoke-100. "
        "Phase 2's manual checks (P2-8, P2-10) need real video on this machine; "
        "PF-6 permits exactly this and nothing larger.",
    )
    ap.add_argument("--subset-dir", default="data/manifests/subsets")
    ap.add_argument(
        "--with-originals",
        action="store_true",
        help="also fetch the real video each fake was built from -- needed by "
        "scripts/11_verify_labels.py to check fake_periods against pixel evidence",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    auth = credentials()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"{DIM}{'=' * 72}{RESET}\n  Fetch LAV-DF metadata (PF-6: video stays on Kaggle)"
        f"\n{DIM}{'=' * 72}{RESET}"
    )

    for member, expected_size in EXPECTED.items():
        dest = out / Path(member).name
        if dest.exists() and dest.stat().st_size == expected_size:
            print(f"  [{DIM}skip{RESET}] {dest.name} already present ({expected_size:,} bytes)")
            continue
        data = fetch(member, auth)
        dest.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()[:16]
        mark = GREEN if len(data) == expected_size else YELLOW
        print(f"  [{mark}OK{RESET}] {dest.name:<22} {len(data):>12,} bytes  sha256:{digest}")
        if len(data) != expected_size:
            print(
                f"       {YELLOW}expected {expected_size:,} -- the mirror changed; re-run CL-7{RESET}"
            )

    if args.spotcheck:
        from src.data.metadata import CLASS_NAMES, load_metadata

        sdir = Path(args.spotcheck_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        entries = load_metadata(out / "metadata.min.json")

        buckets: dict[tuple, list] = {}
        for m in entries:
            buckets.setdefault((m.split, CLASS_NAMES[(m.modify_video, m.modify_audio)]), []).append(
                m
            )
        per = max(1, args.spotcheck // len(buckets))
        rng = random.Random(args.seed)
        sample = [m for k in sorted(buckets, key=str) for m in rng.sample(buckets[k], per)]

        # Always include the bad_label entries -- they are the ones worth eyeballing.
        sample += [
            m
            for m in entries
            if m.fake_periods and max(e for _, e in m.fake_periods) > m.duration + 1e-6
        ]

        print(f"\n  fetching {len(sample)} videos for ffprobe checks -> {sdir}")
        for m in sample:
            dest = sdir / f"{m.video_id}.mp4"
            if dest.exists():
                continue
            dest.write_bytes(fetch(f"LAV-DF/{m.file}", auth))
        total = sum(f.stat().st_size for f in sdir.glob("*.mp4"))
        print(
            f"  [{GREEN}OK{RESET}] {len(list(sdir.glob('*.mp4')))} videos, {total / 1024**2:.1f} MB"
        )

    if args.subset:
        from src.data.subset import load_subset

        ids = load_subset(args.subset, args.subset_dir)
        sdir = out / args.subset
        sdir.mkdir(parents=True, exist_ok=True)
        todo = [v for v in ids if not (sdir / f"{v}.mp4").exists()]
        print(f"\n  subset {args.subset}: {len(ids)} videos -> {sdir}")
        print(f"  {len(ids) - len(todo)} already present, {len(todo)} to fetch")
        for i, vid in enumerate(todo, 1):
            split, num = vid.split("_", 1)
            (sdir / f"{vid}.mp4").write_bytes(fetch(f"LAV-DF/{split}/{num}.mp4", auth))
            if i % 20 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}", flush=True)
        total = sum(f.stat().st_size for f in sdir.glob("*.mp4"))
        print(
            f"  [{GREEN}OK{RESET}] {len(list(sdir.glob('*.mp4')))} videos, {total / 1024**2:.1f} MB"
        )

    print(f"\n{GREEN}DONE{RESET} -- next: python scripts/02_build_manifest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
