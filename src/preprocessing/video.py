"""PyAV decoding with exact fps resampling (P2-1, P2-6).

Two constraints shape this module:

**RAM (P2-6).** The machine has ~6 GB and often <0.3 GB free at rest. A 20 s clip at
25 fps decoded to 640x480 RGB is ~440 MB held at once. So decoding is a *generator*
throughout — frames are yielded, consumed and dropped. Nothing here ever materialises
a full video, and nothing should be changed to do so.

**Frame count (P2-1, corrected by decision PF-7).** The plan's task text says
`assert T == round(duration * 25) +/- 1`. That is wrong for LAV-DF: metadata's
`duration` is `audio_frames/16000 + 0.128`, an audio-derived padded figure, so
`round(duration * 25)` overshoots the true frame count by 2-5 on 99.8% of the
dataset. The invariant this module asserts is `T == video_frames`, which matched
ffprobe on 28/28 spot-checked files.

Resampling to 25 fps is a genuine no-op for LAV-DF -- every file is exactly 25.00 --
but it is implemented properly anyway, by nearest presentation timestamp, because
"some videos are not 25 fps" is listed as an expected problem and silently ingesting
one at 30 fps would stretch every localization target by 20%.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_FPS = 25.0


class DecodeError(RuntimeError):
    """Raised when a video cannot be decoded. Callers quarantine, never skip silently."""


def probe_video(path: str | Path) -> dict:
    """Container-level facts, without decoding pixels."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise DecodeError(f"no video stream in {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            return {
                "fps": fps,
                "n_frames": int(stream.frames) if stream.frames else None,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "duration_s": (
                    float(stream.duration * stream.time_base) if stream.duration else None
                ),
            }
    except DecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - any av failure is a decode failure
        raise DecodeError(f"cannot open {path}: {type(exc).__name__}: {exc}") from exc


def iter_frames(path: str | Path, *, target_fps: float = DEFAULT_FPS) -> Iterator[np.ndarray]:
    """Yield RGB frames as `(H, W, 3)` uint8, resampled to `target_fps`.

    Resampling picks, for each output index `i`, the decoded frame whose presentation
    time is nearest `i / target_fps`. When the source is already at `target_fps` this
    is the identity and every decoded frame passes through exactly once.

    Streaming: one frame is alive at a time. Do not wrap this in `list()`.
    """
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise DecodeError(f"no video stream in {path}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            src_fps = float(stream.average_rate) if stream.average_rate else target_fps
            same_rate = abs(src_fps - target_fps) < 1e-6

            if same_rate:
                for frame in container.decode(stream):
                    yield frame.to_ndarray(format="rgb24")
                return

            # Rate conversion by nearest source index. Held as a single pending frame
            # so memory stays O(1) regardless of the ratio.
            ratio = src_fps / target_fps
            out_i = 0
            for src_i, frame in enumerate(container.decode(stream)):
                arr = None
                # Emit every output index whose nearest source frame is this one.
                while int(round(out_i * ratio)) <= src_i:
                    if int(round(out_i * ratio)) == src_i:
                        if arr is None:
                            arr = frame.to_ndarray(format="rgb24")
                        yield arr
                    out_i += 1
    except DecodeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DecodeError(f"decode failed for {path}: {type(exc).__name__}: {exc}") from exc


def assert_frame_count(n_decoded: int, expected: int, path: str | Path, *, tol: int = 1) -> None:
    """P2-1, as corrected by PF-7: decoded count must match `video_frames`.

    `expected` must come from the manifest's `n_frames` column (which is
    metadata `video_frames`, verified byte-exact against ffprobe), NOT from
    `round(duration * 25)`.
    """
    if abs(n_decoded - expected) > tol:
        raise DecodeError(
            f"{path}: decoded {n_decoded} frames, expected {expected} (tolerance +/-{tol}). "
            "If this fires across the board, check that `expected` is video_frames and not "
            "round(duration * 25) -- see decision PF-7."
        )
