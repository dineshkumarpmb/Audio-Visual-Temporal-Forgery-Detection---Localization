"""Fetch the MediaPipe model bundles Phase 2 needs (decision PF-8).

    python scripts/08_fetch_models.py

mediapipe 1.0.0 removed `mp.solutions` and ships no weights in the wheel, so the
face-landmark bundle must be fetched explicitly. Sizes and hashes are pinned: a
changed bundle changes every crop, and that must be a visible event rather than a
silent drift in the feature cache.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.console import init_console  # noqa: E402

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        3_758_596,
        "64184e229b263107",
    ),
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        229_746,
        "b4578f35940bf5a1",
    ),
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def main() -> int:
    init_console()
    import requests

    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    failures = 0

    for name, (url, size, digest_prefix) in MODELS.items():
        dest = out / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"  [{DIM}skip{RESET}] {name} already present ({size:,} B)")
            continue
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"  [{RED}FAIL{RESET}] {name}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        got = hashlib.sha256(r.content).hexdigest()[:16]
        dest.write_bytes(r.content)
        if len(r.content) == size and got == digest_prefix:
            print(f"  [{GREEN}OK{RESET}] {name:<32} {len(r.content):>10,} B  sha256:{got}")
        else:
            print(
                f"  [{YELLOW}WARN{RESET}] {name}: got {len(r.content):,} B sha256:{got}, "
                f"expected {size:,} B sha256:{digest_prefix} -- the upstream bundle changed, "
                "so every crop will change. Re-run Phase 2 extraction from scratch."
            )

    if failures:
        print(f"\n{RED}{failures} model(s) failed{RESET}")
        return 1
    print(f"\n{GREEN}MODELS READY{RESET} -- next: python scripts/09_extract_faces.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
