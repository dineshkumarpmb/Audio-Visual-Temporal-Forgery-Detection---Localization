"""P1-20 gate: the Phase 1 invariants, asserted against the real LAV-DF metadata.

Skipped when `data/raw/LAV-DF/metadata.min.json` is absent, so a fresh clone still has
a green suite; fetch it with `python scripts/05_fetch_metadata.py`.

The numbers baked in here are the ones measured in P1-10..P1-13 and published in
reports/dataset_statistics.md. They are pinned deliberately: if the mirror is
re-uploaded or silently re-encoded, this file is what notices. That is CL-7's concern
turned into a standing regression test rather than a one-off check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.leakage import check_no_leakage, source_ids_by_split
from src.data.manifest import build_manifest, validate_manifest
from src.data.metadata import (
    assert_fake_periods_are_seconds,
    assert_frame_timeline,
    assert_n_fakes_matches,
    derive_source_id,
    load_metadata,
)

METADATA = Path("data/raw/LAV-DF/metadata.min.json")

pytestmark = pytest.mark.skipif(
    not METADATA.exists(),
    reason="LAV-DF metadata absent -- run scripts/05_fetch_metadata.py",
)

# Published LAV-DF figures, independently confirmed against this mirror (CL-7).
N_TOTAL = 136_304
N_REAL = 36_431
N_FAKE = 99_873
N_SPANS = 114_253
N_BAD_LABEL = 4


@pytest.fixture(scope="module")
def entries():
    return load_metadata(METADATA)


@pytest.fixture(scope="module")
def manifest(entries):
    return build_manifest(entries, "data/raw/LAV-DF")


class TestCounts:
    def test_entry_count_matches_publication(self, entries):
        assert len(entries) == N_TOTAL

    def test_real_fake_split_matches_publication(self, entries):
        assert sum(1 for m in entries if m.label == 0) == N_REAL
        assert sum(1 for m in entries if m.label == 1) == N_FAKE

    def test_official_split_sizes(self, entries):
        from collections import Counter

        assert Counter(m.split for m in entries) == {"train": 78_703, "dev": 31_501, "test": 26_100}

    def test_four_classes_are_all_populated(self, entries):
        from collections import Counter

        counts = Counter(m.class_name for m in entries)
        assert counts == {
            "real": 36_431,
            "visual_only": 33_543,
            "audio_only": 33_170,
            "both": 33_160,
        }


class TestP15Assertions:
    def test_fake_periods_are_seconds(self, entries):
        ev = assert_fake_periods_are_seconds(entries)
        assert ev["n_spans"] == N_SPANS
        assert ev["all_ends_integral"] is False
        assert ev["max_end"] <= ev["max_duration"]

    def test_only_four_entries_are_out_of_bounds(self, entries):
        ev = assert_fake_periods_are_seconds(entries)
        assert ev["n_out_of_bounds"] == N_BAD_LABEL

    def test_n_fakes_is_consistent(self, entries):
        assert_n_fakes_matches(entries)

    def test_duration_is_padded_audio_not_media_duration(self, entries):
        """duration == audio_frames/16000 + 0.128 for every single entry."""
        tl = assert_frame_timeline(entries)
        assert tl["duration_equals_padded_audio"] == N_TOTAL

    def test_p2_1_assertion_would_fail_as_written(self, entries):
        """Guards the PF-7 correction: round(duration*25) is NOT the frame count."""
        tl = assert_frame_timeline(entries)
        assert tl["video_frames_within_1_of_duration_x_fps"] < 1_000
        assert tl["offset_min"] == -5
        assert tl["offset_max"] == -1


class TestP17SourceId:
    def test_every_fake_resolves_to_a_real_root(self, entries):
        roots = set(derive_source_id(entries).values())
        assert len(roots) == N_REAL

    def test_roots_are_exactly_the_real_videos(self, entries):
        roots = set(derive_source_id(entries).values())
        assert roots == {m.video_id for m in entries if m.label == 0}


class TestP18Leakage:
    """R3 -- the highest-impact correctness check in the project."""

    def test_official_splits_have_zero_source_overlap(self, entries):
        sid = derive_source_id(entries)
        report = check_no_leakage(source_ids_by_split((m.split, sid[m.file]) for m in entries))
        assert report["clean"] is True
        assert report["train^dev"] == 0
        assert report["train^test"] == 0
        assert report["dev^test"] == 0

    def test_source_ids_partition_cleanly_across_splits(self, entries):
        sid = derive_source_id(entries)
        by = source_ids_by_split((m.split, sid[m.file]) for m in entries)
        assert sum(len(v) for v in by.values()) == N_REAL


class TestP120Gate:
    def test_manifest_validates_end_to_end(self, manifest):
        report = validate_manifest(manifest)
        assert report["n_total"] == N_TOTAL
        assert report["n_ok"] == N_TOTAL - N_BAD_LABEL
        assert report["status_counts"]["bad_label"] == N_BAD_LABEL
        assert report["leakage"]["clean"] is True

    def test_exclusion_rate_is_negligible(self, manifest):
        assert validate_manifest(manifest)["exclusion_rate"] < 0.001

    def test_fps_and_sample_rate_are_uniform(self, manifest):
        ok = manifest[manifest["status"] == "ok"]
        assert set(ok["fps"]) == {25.0}
        assert set(ok["sample_rate"]) == {16_000}
        assert set(ok["audio_channels"]) == {1}
