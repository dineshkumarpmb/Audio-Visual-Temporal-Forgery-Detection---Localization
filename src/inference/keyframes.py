"""Keyframe thumbnails for the result page and the PDF report.

A timeline says *where*; a thumbnail lets a reader see *what* -- the face at the moment a span
starts, its middle, its end, and the frames either side for comparison. They are chosen after
the model has run (the spans are not known before) and rendered from the analyzer's own
decoded frames, so the picture is exactly what the model saw, not a fresh decode that could
differ by a frame.

Thumbnails are small JPEGs (320 px wide) with the tracked face box drawn on frames where a
face was found. They never touch disk: the API keeps them in memory beside the job and drops
them with it (`job_ttl_s`), the same lifetime as the result itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THUMB_WIDTH = 320
JPEG_QUALITY = 82
CONTEXT_S = 0.5  # how far before / after a span the comparison frames sit
MAX_SEGMENTS = 4  # thumbnails for the 4 most confident spans; the table still lists all
SAMPLES_FOR_REAL = 4


@dataclass
class Keyframe:
    frame: int
    time_s: float
    role: str  # before | start | middle | end | after | sample
    segment: int | None  # index into the result's segments, None for samples
    face_found: bool
    jpeg: bytes = b""

    def meta(self) -> dict:
        return {
            "frame": self.frame,
            "time_s": round(self.time_s, 3),
            "role": self.role,
            "segment": self.segment,
            "face_found": self.face_found,
        }


def select_keyframes(
    n_frames: int, segments: list[tuple[float, float, float]], fps: float
) -> list[tuple[int, str, int | None]]:
    """`(frame, role, segment_index)` to render, in time order, without duplicates.

    Per span (the `MAX_SEGMENTS` most confident): half a second before, first frame, middle,
    last frame, half a second after. With no spans, `SAMPLES_FOR_REAL` evenly spaced frames,
    so a real video's report still shows the face that was checked.
    """
    if n_frames <= 0:
        return []
    last = n_frames - 1
    picks: dict[int, tuple[str, int | None]] = {}
    if not segments:
        for k in range(SAMPLES_FOR_REAL):
            f = min(int((k + 0.5) / SAMPLES_FOR_REAL * n_frames), last)
            picks.setdefault(f, ("sample", None))
    else:
        ranked = sorted(range(len(segments)), key=lambda i: -segments[i][2])[:MAX_SEGMENTS]
        for i in sorted(ranked):
            s, e, _ = segments[i]
            first = min(int(round(s * fps)), last)
            final = min(max(int(round(e * fps)) - 1, first), last)
            ctx = int(round(CONTEXT_S * fps))
            wanted = [
                (first - ctx, "before"),
                (first, "start"),
                ((first + final) // 2, "middle"),
                (final, "end"),
                (final + ctx, "after"),
            ]
            for f, role in wanted:
                # A span's own frames win over another span's context frames.
                if 0 <= f <= last and (f not in picks or role in ("start", "middle", "end")):
                    picks[f] = (role, i)
    return [(f, role, seg) for f, (role, seg) in sorted(picks.items())]


def face_box(
    points: np.ndarray, width: int, height: int, margin: float = 0.55
) -> tuple[int, int, int, int]:
    """A square box around the 5 alignment points (eyes, nose, mouth corners), clipped."""
    cx, cy = points.mean(axis=0)
    span = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0))
    half = span * (1 + margin)
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half * 0.9, 0))
    x1, y1 = int(min(cx + half, width - 1)), int(min(cy + half * 1.1, height - 1))
    return x0, y0, x1, y1


def render_thumbnail(frame: np.ndarray, points: np.ndarray | None, *, highlight: bool) -> bytes:
    """RGB frame -> JPEG bytes, `THUMB_WIDTH` wide, face box drawn if `points` is given.

    The box is red inside a flagged span (`highlight`) and white elsewhere -- the same
    meaning the timeline gives red.
    """
    import cv2

    h, w = frame.shape[:2]
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if points is not None:
        x0, y0, x1, y1 = face_box(points, w, h)
        colour = (59, 59, 208) if highlight else (240, 240, 240)  # BGR of #d03b3b / near-white
        cv2.rectangle(bgr, (x0, y0), (x1, y1), colour, max(2, w // 160))
    scale = THUMB_WIDTH / w
    if scale < 1:
        bgr = cv2.resize(
            bgr, (THUMB_WIDTH, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:  # pragma: no cover - cv2 only fails on an empty image
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
