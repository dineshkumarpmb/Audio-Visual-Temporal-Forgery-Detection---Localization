"""Derived evidence for the result page and the PDF: key figures, per-span statistics, and
which modality drove each flagged span.

Pure functions of an analysis result's arrays, so the page and the report can never disagree.

**Modality attribution, and its limits.** The section 5.1 model can be run with either stream
removed (`drop_visual` / `drop_audio`, the same switches modality dropout trained it with), so
for every frame there are three scores: both streams, audio only, video only. A span is
attributed to the voice when the audio-only run still flags it and the video-only run does
not, and vice versa. This describes *what the model relies on*, not ground truth about how
the video was edited. On LAV-DF the model leans on audio for almost everything (PF-21:
zeroing video costs < 0.01 AUC), so "voice" will be the common answer even for face-only
edits -- the report says so rather than letting the attribution read as forensic fact.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MARGIN = 0.15  # one stream's span mean must lead the other's by this to be named the driver


def _span(arr: np.ndarray | None, a: int, b: int) -> np.ndarray | None:
    if arr is None:
        return None
    part = np.asarray(arr, dtype=np.float64)[a:b]
    return part if part.size else None


def attribute(audio_mean: float | None, visual_mean: float | None, threshold: float) -> str:
    """`voice` | `face` | `both` | `neither` -- which single-stream run still flags the span."""
    if audio_mean is None or visual_mean is None:
        return "unknown"
    a_hit, v_hit = audio_mean > threshold, visual_mean > threshold
    if a_hit and v_hit:
        return "both"
    if a_hit or audio_mean - visual_mean > MARGIN:
        return "voice" if audio_mean >= visual_mean else "face"
    if v_hit or visual_mean - audio_mean > MARGIN:
        return "face"
    return "neither"


def segment_evidence(
    segments: list[tuple[float, float, float]],
    frame_scores: np.ndarray,
    fps: float,
    threshold: float,
    *,
    audio_only: np.ndarray | None = None,
    visual_only: np.ndarray | None = None,
    boundary: np.ndarray | None = None,
    face_found: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    out = []
    n = len(frame_scores)
    for i, (s, e, conf) in enumerate(segments):
        a, b = max(int(round(s * fps)), 0), min(int(round(e * fps)), n)
        full = _span(frame_scores, a, b)
        au, vi = _span(audio_only, a, b), _span(visual_only, a, b)
        ff = _span(face_found, a, b)
        row: dict[str, Any] = {
            "index": i,
            "start_s": round(s, 3),
            "end_s": round(e, 3),
            "duration_s": round(e - s, 3),
            "frames": [a, b],
            "mean_score": round(float(conf), 4),
            "peak_score": round(float(full.max()), 4) if full is not None else None,
            "audio_only_mean": round(float(au.mean()), 4) if au is not None else None,
            "visual_only_mean": round(float(vi.mean()), 4) if vi is not None else None,
            "face_coverage": round(float(ff.mean()), 4) if ff is not None else None,
        }
        row["driver"] = attribute(row["audio_only_mean"], row["visual_only_mean"], threshold)
        if boundary is not None and b > a:
            bnd = np.asarray(boundary, dtype=np.float64)
            # How sure the boundary head is of each edge: its peak within +-2 frames.
            row["start_confidence"] = round(float(bnd[max(a - 2, 0) : a + 3, 0].max()), 4)
            row["end_confidence"] = round(float(bnd[max(b - 3, 0) : b + 2, 1].max()), 4)
        out.append(row)
    return out


def key_figures(
    segments: list[tuple[float, float, float]],
    frame_scores: np.ndarray,
    duration_s: float,
    clip_score: float,
    face_found_fraction: float,
) -> dict[str, Any]:
    forged = float(sum(e - s for s, e, _ in segments))
    scores = np.asarray(frame_scores, dtype=np.float64)
    return {
        "clip_score": round(float(clip_score), 6),
        "n_segments": len(segments),
        "forged_s": round(forged, 3),
        "forged_fraction": round(forged / duration_s, 4) if duration_s > 0 else 0.0,
        "duration_s": round(float(duration_s), 3),
        "peak_frame_score": round(float(scores.max()), 4) if scores.size else 0.0,
        "mean_frame_score": round(float(scores.mean()), 4) if scores.size else 0.0,
        "face_coverage": round(float(face_found_fraction), 4),
    }


def _duration(s: float) -> str:
    return f"{s:.2f} s" if s < 10 else f"{s:.1f} s"


def summary_sentence(is_fake: bool, segments: list, duration_s: float) -> str:
    """The headline sentence -- worded exactly as the frontend's `summarise` (Verdict.tsx)."""
    n = len(segments)
    forged = float(sum(seg[1] - seg[0] for seg in segments))
    if not is_fake:
        if n == 0:
            return "No part of this video was flagged as manipulated."
        moments = "moments were" if n > 1 else "moment was"
        return f"Overall the video looks genuine, though {n} short {moments} flagged — see the timeline."
    if n == 0:
        return "The video as a whole looks manipulated, but no specific moment stood out."
    plural = "s" if n > 1 else ""
    return (
        f"{n} manipulated segment{plural} found, "
        f"{_duration(forged)} of the {_duration(duration_s)} video."
    )


def modality_summary(
    clip_score: float, audio_only_clip: float | None, visual_only_clip: float | None
) -> dict[str, Any]:
    """Whole-clip scores with each stream removed, and a plain-words reading of them."""
    if audio_only_clip is None or visual_only_clip is None:
        return {"available": False}
    drop_video = clip_score - audio_only_clip  # what the face added
    drop_audio = clip_score - visual_only_clip  # what the voice added
    if abs(clip_score - audio_only_clip) < 0.05 and abs(clip_score - visual_only_clip) >= 0.05:
        reading = "The verdict rests on the voice: removing the video barely changes it."
    elif abs(clip_score - visual_only_clip) < 0.05 and abs(clip_score - audio_only_clip) >= 0.05:
        reading = "The verdict rests on the face: removing the audio barely changes it."
    elif abs(drop_video) < 0.05 and abs(drop_audio) < 0.05:
        reading = "Either stream alone gives the same verdict."
    else:
        reading = "Both the voice and the face move the verdict."
    return {
        "available": True,
        "both": round(float(clip_score), 6),
        "audio_only": round(float(audio_only_clip), 6),
        "visual_only": round(float(visual_only_clip), 6),
        "reading": reading,
        "caveat": (
            "This shows what the model relies on, not how the video was edited. Trained on "
            "LAV-DF, it leans on the soundtrack for almost every decision."
        ),
    }
