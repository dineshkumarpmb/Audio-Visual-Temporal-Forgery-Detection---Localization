"""P1-9: quarantine rules from PROJECT_PLAN section 3.7.

Fail loudly, quarantine, never silently skip. Every excluded video carries a reason
code, and the counts land in reports/dataset_statistics.md -- "if you exclude 30% of
the data, that is a finding, not a footnote".

One rule deserves emphasis. `bad_label` (fake_periods outside the video) is flagged as
*probably our bug, not their data*. In LAV-DF it is genuinely their data -- exactly 4
of 136,304 entries -- but that conclusion was reached by inspecting all four, not by
assuming.

`no_face` is not decidable here: it needs the face detector, so Phase 2 assigns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import LAVDF_FPS

# Reason codes. `ok` is the only status that survives into training.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_CORRUPT = "corrupt"
STATUS_NO_AUDIO = "no_audio"
STATUS_NO_VIDEO = "no_video"
STATUS_TOO_SHORT = "too_short"
STATUS_TOO_LONG = "too_long"
STATUS_BAD_LABEL = "bad_label"

ALL_STATUSES = (
    STATUS_OK,
    STATUS_MISSING,
    STATUS_CORRUPT,
    STATUS_NO_AUDIO,
    STATUS_NO_VIDEO,
    STATUS_TOO_SHORT,
    STATUS_TOO_LONG,
    STATUS_BAD_LABEL,
)


@dataclass(frozen=True)
class ProbeResult:
    """What ffprobe told us about one file. `None` fields mean 'not probed'."""

    exists: bool
    readable: bool
    has_video: bool
    has_audio: bool
    fps: float | None = None
    n_frames: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    video_duration_s: float | None = None


def classify(
    entry,  # noqa: ANN001 - MetadataEntry, kept loose to avoid a circular import
    probe: ProbeResult | None,
    *,
    min_duration_s: float = 1.0,
    max_duration_s: float = 60.0,
) -> str:
    """Return the section 3.7 status for one video.

    `probe=None` means metadata-only validation (the mode used when the 25 GB of
    video lives on Kaggle and not on this machine, per decision PF-6). Label and
    duration rules still apply; file-integrity rules are deferred rather than
    guessed at.
    """
    if probe is not None:
        if not probe.exists:
            return STATUS_MISSING
        if not probe.readable:
            return STATUS_CORRUPT
        if not probe.has_video:
            return STATUS_NO_VIDEO
        if not probe.has_audio:
            return STATUS_NO_AUDIO

    # The video timeline is video_frames / 25, never `duration` (see metadata.py).
    duration = entry.video_frames / LAVDF_FPS
    if duration < min_duration_s:
        return STATUS_TOO_SHORT
    if duration > max_duration_s:
        return STATUS_TOO_LONG

    # bad_label: a forged span that leaves the video. Checked against the *video*
    # timeline, which is the one localization targets are rasterised onto.
    for start, end in entry.fake_periods:
        if start < 0 or end <= start or end > duration + 1e-6:
            return STATUS_BAD_LABEL

    return STATUS_OK


def summarise(statuses: list[str]) -> dict[str, int]:
    """Exclusion counts by reason, in a stable order, including zeros."""
    return {s: statuses.count(s) for s in ALL_STATUSES}


def ffprobe(path: str | Path) -> ProbeResult:
    """Probe one media file. Never raises -- an unreadable file is a result, not a crash."""
    import json
    import subprocess

    p = Path(path)
    if not p.exists():
        return ProbeResult(exists=False, readable=False, has_video=False, has_audio=False)

    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return ProbeResult(exists=True, readable=False, has_video=False, has_audio=False)
        info = json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return ProbeResult(exists=True, readable=False, has_video=False, has_audio=False)

    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def _rate(value: str | None) -> float | None:
        if not value or "/" not in value:
            return None
        num, den = value.split("/")
        return float(num) / float(den) if float(den) else None

    return ProbeResult(
        exists=True,
        readable=True,
        has_video=v is not None,
        has_audio=a is not None,
        fps=_rate(v.get("r_frame_rate")) if v else None,
        n_frames=int(v["nb_frames"]) if v and v.get("nb_frames") else None,
        video_duration_s=float(v["duration"]) if v and v.get("duration") else None,
        sample_rate=int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        channels=int(a["channels"]) if a and a.get("channels") else None,
    )
