"""Content-hashed, resumable, crash-safe feature cache (P2-5, P2-12; section 3.8).

Three properties, each earned rather than assumed:

**Content-hashed.** The directory is `data/interim/faces/{sha256(preprocess_config)[:8]}/`.
Change any preprocessing parameter and the cache path changes with it, so two feature
versions can never mix. This is section 3.8 rule 1 and it removes an entire class of
"why did my numbers change?" debugging.

**Resumable.** `is_cached()` is the skip check. An 8-hour extraction on this machine
*will* be interrupted (R13, X-4), so resuming is the normal path, not the exceptional one.

**Crash-safe.** Writes go to a temporary file in the same directory and are then
atomically renamed. Without this, an interrupt during `np.save` leaves a truncated
`.npy` that `is_cached()` happily reports as done and that fails, much later, as a
confusing load error. Same-directory rename is atomic on NTFS and POSIX alike.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.config import PreprocessConfig, config_hash

FACES_ROOT = Path("data/interim/faces")


def cache_root(cfg: PreprocessConfig, root: Path | str = FACES_ROOT) -> Path:
    """The directory this config's crops live in."""
    return Path(root) / config_hash(cfg)


def cache_path(video_id: str, cfg: PreprocessConfig, root: Path | str = FACES_ROOT) -> Path:
    return cache_root(cfg, root) / f"{video_id}.npz"


def is_cached(video_id: str, cfg: PreprocessConfig, root: Path | str = FACES_ROOT) -> bool:
    """True if a complete artefact exists. A zero-byte file is not complete."""
    path = cache_path(video_id, cfg, root)
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def save_faces(
    video_id: str,
    crops: np.ndarray,
    found: np.ndarray,
    cfg: PreprocessConfig,
    root: Path | str = FACES_ROOT,
) -> Path:
    """Write crops + mask atomically.

    `crops` is `(T, S, S, 3)` uint8 and `found` is `(T,)` bool. Stored uncompressed
    (`np.savez`, not `savez_compressed`): these are read every epoch, and zlib on
    uint8 image data buys little while costing decompression time on every access.
    """
    if crops.dtype != np.uint8:
        raise TypeError(f"crops must be uint8, got {crops.dtype}")
    if found.dtype != np.bool_:
        raise TypeError(f"found must be bool, got {found.dtype}")
    if len(crops) != len(found):
        raise ValueError(f"length mismatch: {len(crops)} crops vs {len(found)} mask entries")

    path = cache_path(video_id, cfg, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez(fh, crops=crops, found=found)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)  # atomic within a directory
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def load_faces(
    video_id: str, cfg: PreprocessConfig, root: Path | str = FACES_ROOT
) -> tuple[np.ndarray, np.ndarray]:
    """Read back crops and mask. Uses mmap so a batch does not pull whole clips into RAM."""
    path = cache_path(video_id, cfg, root)
    with np.load(path, mmap_mode="r") as data:
        return data["crops"], data["found"]


def cache_stats(cfg: PreprocessConfig, root: Path | str = FACES_ROOT) -> dict:
    """Size and count of a cache directory, for progress reporting."""
    directory = cache_root(cfg, root)
    if not directory.exists():
        return {"dir": str(directory), "n": 0, "bytes": 0, "partial": 0}
    files = list(directory.glob("*.npz"))
    return {
        "dir": str(directory),
        "n": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "partial": len(list(directory.glob("*.npz.tmp"))),
    }
