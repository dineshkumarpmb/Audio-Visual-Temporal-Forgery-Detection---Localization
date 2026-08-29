"""MediaPipe face detection, tracking and the 5-point extraction (P2-2, P2-3, P2-4).

⚠️ **Decision B-1 needed amending.** The plan specifies "MediaPipe face detect + 468
landmarks" via `mp.solutions.face_mesh`. That API no longer exists: mediapipe 1.0.0
removed `mp.solutions` entirely in favour of the Tasks API, and ships no model
weights in the wheel. So this module uses `mediapipe.tasks.python.vision.FaceLandmarker`
with a `face_landmarker.task` bundle fetched to `models/` by
`scripts/08_fetch_models.py`. The detector is the same BlazeFace + FaceMesh pipeline
underneath; only the Python surface and the weight distribution changed. Recorded as
**decision PF-8**.

Three behaviours worth stating plainly, because each is a judgement call:

**Detection cadence (P2-3).** Landmarks are computed every `detect_every_n` frames
and linearly interpolated between. Faces in these clips move slowly relative to 25 fps,
and detection is the dominant cost, so this is a ~5x saving for sub-pixel error. Gaps
longer than `max_gap` are *not* interpolated across — a long gap means the face really
left, and inventing landmarks there would silently fabricate training data.

**Multiple faces (P2-4).** Policy is **most temporally consistent track, tie-broken by
area**. Picking the largest face per frame independently is the obvious approach and is
wrong: it flips between speakers mid-clip whenever a background face is briefly closer
to camera, producing a crop sequence that jumps between identities. Tracking by nearest
centre to the previous accepted face keeps one identity for the whole clip.

**`face_found` (P2-3).** Every frame carries a boolean. Interpolated frames count as
found; frames inside a too-long gap do not. Downstream masks loss on these — a crop
with no face behind it must never contribute a gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path("models/face_landmarker.task")

# Face-mesh indices for the 5 canonical points, in image-left-to-right order matching
# ARCFACE_112. Eye centres are the mean of the inner and outer corner rather than a
# single landmark: corners are far more stable than lid points under blinking.
EYE_IMAGE_LEFT = (33, 133)  # subject's right eye
EYE_IMAGE_RIGHT = (362, 263)  # subject's left eye
NOSE_TIP = 1
MOUTH_IMAGE_LEFT = 61
MOUTH_IMAGE_RIGHT = 291


@dataclass
class FaceTrack:
    """Per-frame landmark points and validity for one video.

    `points` is `(T, 5, 2)` float32 in pixel coordinates; rows where `found` is False
    are filled with the last known value and must not be trusted.
    """

    points: np.ndarray
    found: np.ndarray

    @property
    def found_fraction(self) -> float:
        return float(self.found.mean()) if self.found.size else 0.0


def five_points(landmarks, width: int, height: int) -> np.ndarray:
    """Reduce a face-mesh landmark list to the 5 alignment points, in pixels."""

    def xy(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float64)

    return np.array(
        [
            (xy(EYE_IMAGE_LEFT[0]) + xy(EYE_IMAGE_LEFT[1])) / 2.0,
            (xy(EYE_IMAGE_RIGHT[0]) + xy(EYE_IMAGE_RIGHT[1])) / 2.0,
            xy(NOSE_TIP),
            xy(MOUTH_IMAGE_LEFT),
            xy(MOUTH_IMAGE_RIGHT),
        ],
        dtype=np.float64,
    )


def _bbox_area(points: np.ndarray) -> float:
    span = points.max(axis=0) - points.min(axis=0)
    return float(span[0] * span[1])


def choose_face(candidates: list[np.ndarray], previous: np.ndarray | None) -> np.ndarray:
    """P2-4's multiple-face policy: continue the existing track, else take the largest.

    With no previous face (first detection of the clip) the largest wins, on the
    assumption that the subject of a talking-head clip is the nearest face. Afterwards
    the nearest centre to the previous accepted face wins, which keeps identity stable
    even when a background face is momentarily larger.
    """
    if not candidates:
        raise ValueError("no candidates")
    if len(candidates) == 1:
        return candidates[0]
    if previous is None:
        return max(candidates, key=_bbox_area)
    prev_centre = previous.mean(axis=0)
    return min(candidates, key=lambda p: float(np.linalg.norm(p.mean(axis=0) - prev_centre)))


class FaceLandmarkerPool:
    """Owns one MediaPipe landmarker. Not thread-safe; one per worker process."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, *, max_faces: int = 5) -> None:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"face landmark model missing at {path} -- run scripts/08_fetch_models.py"
            )
        self._vision = vision
        options = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return the 5-point set for every face found in an RGB frame."""
        import mediapipe as mp

        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = self._landmarker.detect(image)
        return [five_points(lms, w, h) for lms in result.face_landmarks]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkerPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def interpolate_track(
    keyframes: dict[int, np.ndarray], n_frames: int, *, max_gap: int
) -> FaceTrack:
    """Fill every frame from sparse detections (P2-3).

    Between two detections closer than `max_gap` apart, points are linearly
    interpolated and marked found. Across a wider gap, and before the first or after
    the last detection, points are held constant and marked **not** found — a held
    value is a placeholder to keep the array shape regular, never a claim about pixels.
    """
    points = np.zeros((n_frames, 5, 2), dtype=np.float32)
    found = np.zeros(n_frames, dtype=bool)
    if not keyframes:
        return FaceTrack(points, found)

    keys = sorted(keyframes)

    # Before the first and after the last detection: hold, do not claim.
    points[: keys[0]] = keyframes[keys[0]]
    points[keys[-1] :] = keyframes[keys[-1]]

    for a, b in zip(keys, keys[1:], strict=False):
        points[a] = keyframes[a]
        found[a] = True
        if b - a > max_gap:
            points[a + 1 : b] = keyframes[a]  # hold, stay unfound
            continue
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            points[i] = (1.0 - t) * keyframes[a] + t * keyframes[b]
            found[i] = True

    points[keys[-1]] = keyframes[keys[-1]]
    found[keys[-1]] = True
    return FaceTrack(points, found)
