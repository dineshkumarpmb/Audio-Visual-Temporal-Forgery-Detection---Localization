"""Receiving and vetting an upload (P13-6, P13-7).

**Streamed to disk (P13-7).** The multipart body is parsed chunk by chunk straight into a
file under `upload_dir`; at no point is the upload held in memory, and the size cap is
enforced *while* reading, so an oversized body is cut off at the cap rather than spooled in
full first. An honest `Content-Length` over the cap is refused before a byte is read.

**Vetted before queueing (P13-6).** The client's filename and MIME type are checked first
(cheap, and they say what the client *thinks* it sent), then the container itself is opened
with PyAV -- the real MIME check: demuxer family must match the extension, the video codec
must be on the list, an audio stream must exist (the model is audio-led, PF-21), the
duration must be in bounds, and one frame must actually decode. Only a file that passes
all of that costs a place in the queue.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from api.errors import ApiError
from api.settings import ALLOWED_EXTENSIONS, ApiSettings

FILE_FIELD = "file"
MULTIPART_SLACK = 64 * 1024  # boundary + part headers on top of the file itself

# What browsers and curl send for the four containers. `application/octet-stream` (or no
# type at all) is accepted because the container check below is authoritative.
ALLOWED_MIME = {
    "",
    "application/octet-stream",
    "video/mp4",
    "video/x-m4v",
    "video/quicktime",
    "video/avi",
    "video/msvideo",
    "video/vnd.avi",
    "video/x-msvideo",
    "video/matroska",
    "video/x-matroska",
}
# Extension -> the name ffmpeg's demuxer reports for it (`container.format.name` is a
# comma-separated alias list, e.g. "mov,mp4,m4a,3gp,3g2,mj2").
DEMUXER = {".mp4": "mp4", ".mov": "mov", ".avi": "avi", ".mkv": "matroska"}
ALLOWED_VIDEO_CODECS = {
    "h264",
    "hevc",
    "mpeg4",
    "mpeg2video",
    "mpeg1video",
    "msmpeg4v2",
    "msmpeg4v3",
    "mjpeg",
    "vp8",
    "vp9",
    "av1",
    "prores",
}


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    filename: str  # the client's name, display only -- never used as a path
    extension: str
    size_bytes: int


def _too_large(settings: ApiSettings) -> ApiError:
    return ApiError(
        413,
        "file_too_large",
        f"the upload exceeds the {settings.max_upload_mb:g} MB limit",
        {"max_bytes": settings.max_upload_bytes},
    )


class _Receiver:
    """MultipartParser callbacks that write the `file` part to disk and ignore the rest."""

    def __init__(self, dest_dir: Path, settings: ApiSettings) -> None:
        self.dest_dir = dest_dir
        self.settings = settings
        self.saved: SavedUpload | None = None
        self.path: Path | None = None
        self._fh = None
        self._size = 0
        self._headers: dict[str, str] = {}
        self._field = b""
        self._value = b""
        self._writing = False
        self._meta: tuple[str, str] | None = None

    # -- headers -----------------------------------------------------------

    def on_part_begin(self) -> None:
        self._headers, self._field, self._value = {}, b"", b""

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def on_header_end(self) -> None:
        name = self._field.decode("latin-1").strip().lower()
        self._headers[name] = self._value.decode("latin-1").strip()
        self._field, self._value = b"", b""

    def on_headers_finished(self) -> None:
        _, params = parse_options_header(self._headers.get("content-disposition", ""))
        name = params.get(b"name", b"").decode("utf-8", "replace")
        if name != FILE_FIELD or self.path is not None or b"filename" not in params:
            self._writing = False
            return
        filename = PurePath(params[b"filename"].decode("utf-8", "replace").replace("\\", "/"))
        display = filename.name[:255]
        ext = filename.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ApiError(
                415,
                "unsupported_format",
                f"'{ext or display}' is not an accepted video format",
                {"allowed": list(ALLOWED_EXTENSIONS)},
            )
        mime, _ = parse_options_header(self._headers.get("content-type", ""))
        mime = mime.decode("latin-1").lower()
        if mime not in ALLOWED_MIME:
            raise ApiError(
                415,
                "unsupported_media_type",
                f"content type '{mime}' is not a supported video type",
                {"allowed_extensions": list(ALLOWED_EXTENSIONS)},
            )
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dest_dir / f"{uuid.uuid4().hex}{ext}"
        self._fh = self.path.open("wb")  # noqa: SIM115 - closed in on_part_end / cleanup
        self._meta = (display, ext)
        self._writing = True

    # -- body --------------------------------------------------------------

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._writing:
            return
        self._size += end - start
        if self._size > self.settings.max_upload_bytes:
            raise _too_large(self.settings)
        self._fh.write(data[start:end])

    def on_part_end(self) -> None:
        if self._writing:
            self._fh.close()
            self._fh = None
            self._writing = False
            display, ext = self._meta
            self.saved = SavedUpload(self.path, display, ext, self._size)

    def cleanup(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.path is not None:
            self.path.unlink(missing_ok=True)


async def receive_upload(request: Request, settings: ApiSettings) -> SavedUpload:
    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    if ctype != b"multipart/form-data" or b"boundary" not in params:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"send the video as multipart/form-data in a '{FILE_FIELD}' field",
        )
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.max_upload_bytes + MULTIPART_SLACK:
        raise _too_large(settings)

    rx = _Receiver(settings.upload_dir, settings)
    callbacks = {
        name: getattr(rx, name)
        for name in (
            "on_part_begin",
            "on_header_field",
            "on_header_value",
            "on_header_end",
            "on_headers_finished",
            "on_part_data",
            "on_part_end",
        )
    }
    parser = MultipartParser(params[b"boundary"], callbacks)
    try:
        async for chunk in request.stream():
            parser.write(chunk)
        parser.finalize()
    except ApiError:
        rx.cleanup()
        raise
    except Exception as exc:  # noqa: BLE001 - a malformed body or a dropped connection
        rx.cleanup()
        raise ApiError(400, "malformed_upload", "the multipart body could not be parsed") from exc

    if rx.saved is None:
        rx.cleanup()
        raise ApiError(422, "missing_file", f"no '{FILE_FIELD}' part with a filename was sent")
    if rx.saved.size_bytes == 0:
        rx.cleanup()
        raise ApiError(422, "empty_file", "the uploaded file is empty")
    return rx.saved


def inspect_media(upload: SavedUpload, settings: ApiSettings) -> dict[str, Any]:
    """Container-level vetting. Raises `ApiError`; returns the facts for `MediaInfo`."""
    import av

    corrupt = ApiError(422, "corrupt_video", "the file could not be decoded as a video")
    try:
        container = av.open(str(upload.path))
    except Exception as exc:  # noqa: BLE001 - any demuxer failure means unreadable
        raise corrupt from exc

    with container:
        demuxers = container.format.name.split(",")
        if DEMUXER[upload.extension] not in demuxers:
            raise ApiError(
                415,
                "format_mismatch",
                f"the file is named {upload.extension} but its contents are '{demuxers[0]}'",
                {"extension": upload.extension, "container": container.format.name},
            )
        if not container.streams.video:
            raise ApiError(422, "no_video_stream", "the file contains no video stream")
        vs = container.streams.video[0]
        codec = vs.codec_context.name
        if codec not in ALLOWED_VIDEO_CODECS:
            raise ApiError(
                415,
                "unsupported_codec",
                f"video codec '{codec}' is not supported",
                {"allowed": sorted(ALLOWED_VIDEO_CODECS)},
            )
        if not container.streams.audio:
            raise ApiError(
                422,
                "no_audio",
                "the file has no audio track; the detector needs the soundtrack",
            )
        audio_codec = container.streams.audio[0].codec_context.name

        if container.duration is not None:
            duration = container.duration / av.time_base
        elif vs.duration is not None:
            duration = float(vs.duration * vs.time_base)
        else:
            raise corrupt
        if not settings.min_duration_s <= duration <= settings.max_duration_s:
            raise ApiError(
                422,
                "duration_out_of_range",
                f"the video is {duration:.1f} s; accepted length is "
                f"{settings.min_duration_s:g}-{settings.max_duration_s:g} s",
                {"duration_s": round(duration, 3)},
            )

        try:  # headers can be intact while the stream is garbage: decode one frame
            frame = next(container.decode(video=0), None)
        except Exception as exc:  # noqa: BLE001
            raise corrupt from exc
        if frame is None:
            raise corrupt

        return {
            "container": container.format.name,
            "video_codec": codec,
            "audio_codec": audio_codec,
            "width": vs.codec_context.width,
            "height": vs.codec_context.height,
            "fps": float(vs.average_rate) if vs.average_rate else None,
            "duration_s": round(duration, 3),
            "size_bytes": upload.size_bytes,
        }
