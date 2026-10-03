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

⚠️ **This endpoint is rate-limited two separate ways -- see PF-16, which supersedes
PF-13.** PF-13 saw only the first and concluded dev-2k was unobtainable locally. It is
not. Measured 2026-08-30:

  1. **Too much concurrency -> HTTP 404** on files that plainly exist, which is what
     made PF-13 read this as a per-file or per-session *quota*:

         workers=1   60/60 ok    31 files/min
         workers=3   45/45 ok    67 files/min
         workers=5   45/45 ok   108 files/min
         workers=8   64 ok, then a run of 404s -- while a serial probe kept getting 200

  2. **Sustained volume -> HTTP 429 with `Retry-After: 180`.** After ~500 files in a
     burst every request 429s for three minutes, then serves normally again. This is a
     scheduling instruction, not an error, and `fetch()` sleeps on it via a shared
     `THROTTLE` so all workers back off together.

The practical lesson is that **pacing beats bursting**: an unpaced 5-worker pool spends
most of its time inside `Retry-After` and settles at ~6 files/min, while `--rate 25` never
trips the limiter and holds ~25. Defaults are `--workers 5 --rate 25`; `--max-hours`
bounds the whole thing.

PF-6 still keeps the *full* 25.5 GB corpus on Kaggle; this path is for subset-scale work.

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


class _Throttle:
    """Process-wide pacing + cooldown shared by every download thread.

    Two jobs, and the first is the one that actually matters:

    * **Pace under the limit.** Bursting and then absorbing `Retry-After` is far slower
      than never tripping the limiter. Measured: 60 serial files at 31/min drew *zero*
      429s, while 5 workers at full tilt settled into a burst-then-wait cycle worth about
      6 files/min. `rate` spaces request starts globally so the whole pool stays under
      the line.
    * **Back off together when it still trips.** When Kaggle answers 429 it sets
      `Retry-After` (measured 180 s, decaying to 5 s as the window refills). One worker
      seeing that is enough to know *all* of them must stop, so the deadline is shared
      and threads arriving during a cooldown wait rather than each earning their own 429.

    Jitter on wake keeps the pool from resuming in lockstep.
    """

    def __init__(self, rate_per_min: float = 0.0) -> None:
        import threading

        self._lock = threading.Lock()
        self._resume_at = 0.0
        self._next_slot = 0.0
        self._interval = 60.0 / rate_per_min if rate_per_min > 0 else 0.0
        self.pauses = 0

    def wait(self) -> None:
        import random
        import time

        while True:
            with self._lock:
                remaining = self._resume_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 5.0) + random.uniform(0, 0.4))
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def pause(self, seconds: float) -> None:
        import time

        with self._lock:
            deadline = time.monotonic() + seconds
            if deadline > self._resume_at:
                self._resume_at = deadline
                self.pauses += 1
                print(
                    f"    {DIM}429 -- backing off {seconds:.0f}s (Retry-After){RESET}",
                    flush=True,
                )


THROTTLE = _Throttle()  # replaced in main() once --rate is known


def fetch(member: str, auth: tuple[str, str], timeout: int = 300, *, max_wait: int = 1800) -> bytes:
    """Download one dataset member, unwrapping Kaggle's zip envelope.

    Honours **HTTP 429 + `Retry-After`** (PF-16). This endpoint rate-limits by request
    volume and says exactly how long to wait; sleeping on that header is the whole
    difference between a subset that downloads in minutes and one that reads as
    "unobtainable". A 429 is *not* a failure -- it is a scheduling instruction.
    """
    import requests

    waited = 0
    while True:
        THROTTLE.wait()
        r = requests.get(f"{BASE}/{MIRROR}?file_name={quote(member)}", auth=auth, timeout=timeout)
        if r.status_code == 429 and waited < max_wait:
            delay = int(r.headers.get("Retry-After", 60))
            THROTTLE.pause(delay)
            waited += delay
            continue
        r.raise_for_status()
        if r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                return z.read(z.namelist()[0])
        return r.content


# PF-13 note 4: subset ids sort `dev_* < test_* < train_*`, so the first quota-limited
# run spent itself almost entirely on dev and left the train split at 58 clips -- the
# one split that actually limits learning. Fetch train first. This reorders *arrival*
# only; split membership still comes from the committed subset, so section 3.5 RULE 1
# is untouched.
SPLIT_PRIORITY = {"train": 0, "dev": 1, "test": 2}


def _prioritised(video_ids: list[str]) -> list[str]:
    return sorted(video_ids, key=lambda v: (SPLIT_PRIORITY.get(v.split("_", 1)[0], 3), v))


def _fetch_window(
    video_ids: list[str],
    dest: Path,
    auth: tuple[str, str],
    workers: int,
    *,
    trip: int = 25,
    attempts: int = 3,
) -> tuple[int, list[str]]:
    """Download as much of `video_ids` as the current quota window allows.

    Returns `(n_fetched, still_missing)`.

    Kaggle serves one small file per request, so the transfer is latency-bound rather
    than bandwidth-bound and threads help a lot -- measured ~9 videos/min serially.

    Each file lands via a temp name + rename. A partial file left by an interrupt would
    otherwise be skipped as "already present" on the next run and then fail to decode
    much later, which is a miserable thing to debug.

    PF-13: the volume quota is signalled as a **404 on every file**, so a *run* of
    consecutive failures -- never a single one -- is the signal to stop and wait.

    Two refinements, both learned the hard way on 2026-08-30:

    * **Retry before counting a failure.** Eight-way concurrency produces occasional
      bursts of transient connection errors. A first cut counted those toward the trip
      and stopped a *healthy* window after 66 files -- a serial probe seconds later got
      HTTP 200 on the very next ids. Only a file that fails every attempt counts.
    * **Only 404 counts toward the trip.** That is the specific signal PF-13 identified.
      Any other error resets the run, so a network blip cannot masquerade as the quota.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    stop = threading.Event()
    lock = threading.Lock()
    state = {"consecutive": 0, "ok": 0, "failed": 0}

    def one(vid: str) -> None:
        if stop.is_set():
            return
        split, num = vid.split("_", 1)
        member = f"LAV-DF/{split}/{num}.mp4"
        data, saw_404 = None, False
        for attempt in range(attempts):
            if stop.is_set():
                return
            try:
                data = fetch(member, auth)
                break
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                saw_404 = "404" in str(exc)
                if attempt + 1 < attempts:
                    time.sleep(1.5 * (attempt + 1))
        if data is None:
            with lock:
                state["failed"] += 1
                state["consecutive"] = state["consecutive"] + 1 if saw_404 else 0
                if state["consecutive"] >= trip:
                    stop.set()
            return
        tmp = dest / f".{vid}.part"
        tmp.write_bytes(data)
        tmp.replace(dest / f"{vid}.mp4")
        with lock:
            state["consecutive"] = 0
            state["ok"] += 1
            if state["ok"] % 100 == 0:
                print(
                    f"    {state['ok']} fetched this window ({state['failed']} failed)",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, video_ids))

    missing = [v for v in video_ids if not (dest / f"{v}.mp4").exists()]
    return state["ok"], missing


def _wait_for_quota(
    probe_id: str, dest: Path, auth: tuple[str, str], *, step: int = 300, cap: int = 5400
) -> bool:
    """Sleep until the download endpoint answers again, probing every `step` seconds.

    PF-13 measured the window as clearing "after about an hour", but that was one
    observation; probing costs one file and finds the real edge instead of assuming it.
    The probe *keeps* what it downloads, so a successful probe is not wasted quota.
    """
    import time

    waited = 0
    split, num = probe_id.split("_", 1)
    while waited < cap:
        time.sleep(step)
        waited += step
        try:
            data = fetch(f"LAV-DF/{split}/{num}.mp4", auth, timeout=120)
        except Exception:  # noqa: BLE001 - still throttled
            print(f"    {DIM}still throttled after {waited // 60} min{RESET}", flush=True)
            continue
        tmp = dest / f".{probe_id}.part"
        tmp.write_bytes(data)
        tmp.replace(dest / f"{probe_id}.mp4")
        print(f"  {GREEN}quota cleared{RESET} after ~{waited // 60} min", flush=True)
        return True
    return False


def _fetch_many(
    video_ids: list[str],
    dest: Path,
    auth: tuple[str, str],
    workers: int,
    *,
    max_hours: float = 0.0,
) -> list[str]:
    """Fetch every id, riding out PF-13's quota windows. Returns what is still missing.

    With `max_hours == 0` this is a single window and behaves exactly as before -- the
    smoke-scale path Phases 1-3 used. A positive `max_hours` opts into the wait-and-resume
    loop needed to pull a full subset locally.
    """
    import time

    todo = _prioritised(video_ids)
    deadline = time.time() + max_hours * 3600 if max_hours else 0.0
    window = 0
    while todo:
        window += 1
        got, todo = _fetch_window(todo, dest, auth, workers)
        print(f"  window {window}: +{got} fetched, {len(todo)} remaining", flush=True)
        if not todo or not deadline:
            break
        if time.time() >= deadline:
            print(f"  {YELLOW}--max-hours reached with {len(todo)} remaining{RESET}")
            break
        if got == 0:
            print(f"  {YELLOW}window {window} fetched nothing -- giving up{RESET}")
            break
        if not _wait_for_quota(todo[0], dest, auth):
            print(f"  {YELLOW}quota did not clear within 90 min -- stopping{RESET}")
            break
        todo = [v for v in todo if not (dest / f"{v}.mp4").exists()]
    return todo


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
    ap.add_argument(
        "--workers",
        type=int,
        default=5,
        help="parallel downloads. PF-16: >5 makes Kaggle 404 files that exist; 5 is the "
        "measured ceiling and gives ~108 files/min.",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=25.0,
        help="target files/min across all workers. PF-16: staying just under the limit "
        "beats bursting into Retry-After penalties -- 31/min drew zero 429s, while an "
        "unpaced pool settled at ~6/min. 0 disables pacing.",
    )
    ap.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="ride out PF-13's quota windows for up to this many hours, waiting and "
        "resuming instead of stopping at the first run of 404s. 0 (default) = a single "
        "window, the smoke-scale behaviour Phases 1-3 used.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_PRIORITY),
        default=None,
        help="with --subset, fetch only these splits' videos. Phase 11 fetches the test "
        "split alone: every downstream loader trains on whatever features exist, so "
        "fetching more train/dev clips would silently change the ablations' training set.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    global THROTTLE
    THROTTLE = _Throttle(args.rate)

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
        if args.splits:
            ids = [v for v in ids if v.split("_", 1)[0] in args.splits]
        sdir = out / args.subset
        sdir.mkdir(parents=True, exist_ok=True)
        todo = [v for v in ids if not (sdir / f"{v}.mp4").exists()]
        print(f"\n  subset {args.subset}: {len(ids)} videos -> {sdir}")
        print(
            f"  {len(ids) - len(todo)} already present, {len(todo)} to fetch "
            f"({args.workers} workers, "
            f"{'single window' if not args.max_hours else str(args.max_hours) + 'h budget'})"
        )
        missing = _fetch_many(todo, sdir, auth, args.workers, max_hours=args.max_hours)
        have = sorted(f.stem for f in sdir.glob("*.mp4"))
        total = sum(f.stat().st_size for f in sdir.glob("*.mp4"))
        by_split: dict[str, int] = {}
        for v in have:
            by_split[v.split("_", 1)[0]] = by_split.get(v.split("_", 1)[0], 0) + 1
        mark = GREEN if not missing else YELLOW
        print(f"  [{mark}OK{RESET}] {len(have)}/{len(ids)} videos, {total / 1024**2:.1f} MB")
        print("       splits: " + "  ".join(f"{k} {n}" for k, n in sorted(by_split.items())))
        if missing:
            # Anything downstream has to know it is running on a partial subset. Carrying
            # this by hand through every report is exactly what made Phase 4's first pass
            # so easy to misread.
            print(f"       {YELLOW}{len(missing)} still missing (PF-13 quota){RESET}")

    print(f"\n{GREEN}DONE{RESET} -- next: python scripts/02_build_manifest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
