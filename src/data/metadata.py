"""LAV-DF `metadata.min.json` parsing and the P1-5 correctness assertions.

The file is `metadata.min.json`, not `metadata.json` (confirmed in PRE-4). The
minified variant drops `timestamps` and `transcript`; every field this project
uses survives, so we read the 2.6 MB minified file rather than the full one.

Two things in here are load-bearing and must never be relaxed into warnings:

1. **`fake_periods` are float SECONDS.** The authors' own loader annotates them
   `List[List[int]]`, which would make them frame indices and every localization
   target wrong by ~25x -- with a perfectly healthy-looking loss curve. P1-5
   settles it by assertion, on every load, not by reading the annotation.

2. **`duration` is not a media duration.** It equals `audio_frames/16000 + 0.128`
   for all 136,304 entries and exceeds *both* stream durations on every file
   probed. The video timeline is `video_frames / 25`. Deriving frame counts from
   `duration` yields 2-5 too many frames per video (see `assert_frame_timeline`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import LAVDF_DURATION_PAD_S, LAVDF_FPS, LAVDF_SAMPLE_RATE

# The four-class taxonomy of section 3.4, keyed by (modify_video, modify_audio).
CLASS_NAMES: dict[tuple[bool, bool], str] = {
    (False, False): "real",
    (True, False): "visual_only",
    (False, True): "audio_only",
    (True, True): "both",
}


@dataclass(frozen=True)
class MetadataEntry:
    """One row of metadata.min.json, with the derived fields Part 6 needs."""

    file: str
    split: str
    n_fakes: int
    fake_periods: list[list[float]]
    duration: float
    original: str | None
    modify_video: bool
    modify_audio: bool
    video_frames: int
    audio_frames: int
    audio_channels: int

    @property
    def video_id(self) -> str:
        """`train/000001.mp4` -> `train_000001`. Filesystem-safe and split-explicit."""
        return self.file.replace("/", "_").rsplit(".", 1)[0]

    @property
    def label(self) -> int:
        """Binary label: 1 if any modality was modified."""
        return int(self.modify_video or self.modify_audio)

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[(self.modify_video, self.modify_audio)]

    @property
    def video_duration_s(self) -> float:
        """The *video* timeline: video_frames / 25. Use this, not `duration`."""
        return self.video_frames / LAVDF_FPS


def load_metadata(path: str | Path) -> list[MetadataEntry]:
    """Parse metadata.min.json into typed entries. Raises on schema drift."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"expected a JSON list at {path}, got {type(raw).__name__}")

    entries: list[MetadataEntry] = []
    for i, r in enumerate(raw):
        try:
            entries.append(
                MetadataEntry(
                    file=r["file"],
                    split=r["split"],
                    n_fakes=r["n_fakes"],
                    fake_periods=[[float(s), float(e)] for s, e in r["fake_periods"]],
                    duration=float(r["duration"]),
                    original=r["original"],
                    modify_video=bool(r["modify_video"]),
                    modify_audio=bool(r["modify_audio"]),
                    video_frames=int(r["video_frames"]),
                    audio_frames=int(r["audio_frames"]),
                    audio_channels=int(r["audio_channels"]),
                )
            )
        except KeyError as exc:
            raise KeyError(f"metadata entry {i} ({r.get('file', '?')}) is missing {exc}") from exc
    return entries


# ---------------------------------------------------------------------------------
# P1-5 assertions. These run on every load; none may be downgraded to a warning.
# ---------------------------------------------------------------------------------


def assert_fake_periods_are_seconds(
    entries: list[MetadataEntry], *, tolerance: float = 1e-6
) -> dict:
    """The units check. Returns evidence; raises if the data looks frame-indexed.

    Two independent signals, because either alone is circumstantial:
      - Frame indices would be integers. Real span boundaries are not.
      - Frame indices for a 25 fps 20 s clip would run to ~500, far past `duration`.

    Entries whose spans exceed `duration` are *not* fatal -- section 3.7 quarantines
    them as `bad_label`. A systematic overshoot is fatal; a handful is dirty data.
    """
    spans = [(s, e) for m in entries for s, e in m.fake_periods]
    if not spans:
        raise ValueError("no fake_periods found -- cannot verify units on an all-real set")

    ends = [e for _, e in spans]
    all_integral = all(float(e).is_integer() for e in ends)

    out_of_bounds = [
        m
        for m in entries
        if m.fake_periods and max(e for _, e in m.fake_periods) > m.duration + tolerance
    ]
    frac_oob = len(out_of_bounds) / len(entries)

    evidence = {
        "n_spans": len(spans),
        "max_end": max(ends),
        "max_duration": max(m.duration for m in entries),
        "all_ends_integral": all_integral,
        "n_out_of_bounds": len(out_of_bounds),
        "out_of_bounds_files": [m.file for m in out_of_bounds],
    }

    # A frame-indexed file would show BOTH integral ends AND wholesale overshoot.
    if all_integral and frac_oob > 0.5:
        raise AssertionError(
            "fake_periods look like FRAME INDICES, not seconds: every end is integral "
            f"and {frac_oob:.1%} exceed `duration`. Every localization target would be "
            f"wrong by ~{LAVDF_FPS}x. Evidence: {evidence}"
        )
    if frac_oob > 0.01:
        raise AssertionError(
            f"{frac_oob:.2%} of entries have fake_periods beyond `duration` -- too many "
            "to be dirty data; suspect a parsing bug on our side (section 3.7 bad_label). "
            f"Evidence: {evidence}"
        )
    return evidence


def assert_n_fakes_matches(entries: list[MetadataEntry]) -> None:
    """`n_fakes` must equal `len(fake_periods)` -- a cheap parser-correctness check."""
    bad = [m.file for m in entries if m.n_fakes != len(m.fake_periods)]
    if bad:
        raise AssertionError(
            f"{len(bad)} entries where n_fakes != len(fake_periods), e.g. {bad[:5]}"
        )


def assert_frame_timeline(entries: list[MetadataEntry]) -> dict:
    """Pin down which field defines the video timeline (the P2-1 correction).

    PROJECT_PLAN Phase 2 P2-1 says `assert T == round(duration * 25) +/- 1`. That is
    wrong for this dataset and would fail on essentially every file: `duration` is
    audio-derived and padded, so `round(duration * 25)` overshoots `video_frames` by
    2-5 frames. The correct invariant is `T == video_frames`, verified against
    ffprobe in P1-13 (28/28 exact).
    """
    off = [m.video_frames - round(m.duration * LAVDF_FPS) for m in entries]
    pad_ok = sum(
        1
        for m in entries
        if abs(m.duration - (m.audio_frames / LAVDF_SAMPLE_RATE + LAVDF_DURATION_PAD_S)) < 1e-9
    )
    within_1 = sum(1 for d in off if abs(d) <= 1)
    return {
        "n": len(entries),
        "duration_equals_padded_audio": pad_ok,
        "video_frames_within_1_of_duration_x_fps": within_1,
        "offset_min": min(off),
        "offset_max": max(off),
    }


def derive_source_id(entries: list[MetadataEntry]) -> dict[str, str]:
    """P1-7: the identity key that the R3 leakage assertions depend on.

    `metadata.min.json` exposes no explicit identity field, so `source_id` is derived
    by walking the `original` pointer to its root: a fake's root is the real video it
    was built from, and a real video is its own root. Every fake derived from the same
    real source therefore shares a `source_id` with it.

    This is deliberately conservative. It cannot detect two *different* real videos of
    the same VoxCeleb2 speaker, so it prevents original-vs-derivative leakage but not
    all identity leakage -- stated plainly in reports/dataset_statistics.md rather than
    papered over. Cycles raise instead of looping forever.
    """
    by_file = {m.file: m for m in entries}
    root_of: dict[str, str] = {}

    for m in entries:
        chain: list[str] = []
        seen: set[str] = set()
        cur: str | None = m.file
        root: str | None = None
        while cur is not None:
            if cur in root_of:  # memoised: reuse the root already computed
                root = root_of[cur]
                break
            if cur in seen:
                raise ValueError(f"cycle in the `original` chain: {chain + [cur]}")
            seen.add(cur)
            chain.append(cur)
            entry = by_file.get(cur)
            nxt = entry.original if entry is not None else None
            if nxt is None:
                root = cur
                break
            cur = nxt
        if root is None:
            root = chain[-1] if chain else m.file
        for f in chain:
            root_of[f] = root

    return {f: r.replace("/", "_").rsplit(".", 1)[0] for f, r in root_of.items()}
