"""Video -> aligned face crops, end to end (P2-1 .. P2-6).

Ties the four modules together into the one operation the rest of the project needs:
`extract_video()` turns a path into `(T, S, S, 3)` uint8 crops plus a `(T,)` bool
`face_found` mask, cached and resumable.

Memory discipline (P2-6). A 20 s clip is 500 frames; at 112x112x3 the *output* is
18 MB, which is fine, but the decoded 640x480 source would be 440 MB if held. So
frames are consumed one at a time and only the small crops accumulate. The two passes
this requires — detect on a subsample, then align — are done by decoding twice rather
than by buffering frames. Decode is 36-60x realtime on this machine (Phase 0) and RAM
is the binding constraint, so paying decode twice to halve peak memory is the right
trade here and would not be on a larger box.

Determinism (P2-11): no RNG anywhere on this path. MediaPipe CPU inference, the
analytic Umeyama solve and `cv2.warpAffine` are all deterministic, so two runs over
the same input produce byte-identical `.npz` payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import PreprocessConfig
from src.preprocessing.align import align_face
from src.preprocessing.cache import is_cached, save_faces
from src.preprocessing.face import FaceLandmarkerPool, FaceTrack, choose_face, interpolate_track
from src.preprocessing.video import DecodeError, assert_frame_count, iter_frames

# A detection gap wider than this is treated as "the face left", not "interpolate".
# 15 frames = 0.6 s at 25 fps, comparable to the mean forged span (0.65 s), so a gap
# this wide could hide an entire manipulated segment behind invented landmarks.
DEFAULT_MAX_GAP = 15


@dataclass
class ExtractionResult:
    video_id: str
    n_frames: int
    found_fraction: float
    cached: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def detect_track(
    path: str | Path,
    n_frames: int,
    cfg: PreprocessConfig,
    landmarker: FaceLandmarkerPool,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
) -> FaceTrack:
    """Pass 1: landmarks every `detect_every_n` frames, then interpolate (P2-3)."""
    keyframes: dict[int, np.ndarray] = {}
    previous: np.ndarray | None = None

    for i, frame in enumerate(iter_frames(path, target_fps=cfg.target_fps)):
        if i % cfg.detect_every_n:
            continue
        candidates = landmarker.detect(frame)
        if not candidates:
            continue
        chosen = choose_face(candidates, previous)  # P2-4
        keyframes[i] = chosen
        previous = chosen

    return interpolate_track(keyframes, n_frames, max_gap=max_gap)


def align_all(path: str | Path, track: FaceTrack, cfg: PreprocessConfig) -> np.ndarray:
    """Pass 2: warp each frame onto the canonical template (P2-2).

    Frames with no face still produce a crop — the geometry is held from the nearest
    detection — so the array stays rectangular and indexable by frame number. The
    `found` mask is what tells downstream code to ignore them.
    """
    size = cfg.crop_size
    crops = np.zeros((len(track.found), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(iter_frames(path, target_fps=cfg.target_fps)):
        if i >= len(crops):
            break
        crops[i] = align_face(frame, track.points[i], crop_size=size, margin=cfg.face_margin)
    return crops


def extract_video(
    path: str | Path,
    video_id: str,
    n_frames: int,
    cfg: PreprocessConfig,
    landmarker: FaceLandmarkerPool,
    *,
    root: Path | str = "data/interim/faces",
    max_gap: int = DEFAULT_MAX_GAP,
    force: bool = False,
) -> ExtractionResult:
    """Extract one video, skipping if already cached (P2-5).

    `n_frames` must be the manifest's `n_frames` (metadata `video_frames`), not
    `round(duration * 25)` — see decision PF-7.
    """
    if not force and is_cached(video_id, cfg, root):
        return ExtractionResult(video_id, n_frames, float("nan"), cached=True)

    try:
        track = detect_track(path, n_frames, cfg, landmarker, max_gap=max_gap)
        crops = align_all(path, track, cfg)
        assert_frame_count(len(crops), n_frames, path)
        save_faces(video_id, crops, track.found, cfg, root)
    except DecodeError as exc:
        return ExtractionResult(video_id, 0, 0.0, cached=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad video must not kill a long run
        return ExtractionResult(
            video_id, 0, 0.0, cached=False, error=f"{type(exc).__name__}: {exc}"
        )

    return ExtractionResult(video_id, len(crops), track.found_fraction, cached=False)
