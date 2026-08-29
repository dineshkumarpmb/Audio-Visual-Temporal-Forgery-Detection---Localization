"""P1-17 / P1-8: manifest construction, quarantine rules, and the leakage assertions.

The leakage tests are the point of this file. Section 3.5 RULE 3 requires them to
*fail the build*, so they live in the default pytest run, not behind a marker.
"""

from __future__ import annotations

import pytest

from src.data.leakage import LeakageError, check_no_leakage, source_ids_by_split
from src.data.manifest import MANIFEST_COLUMNS, build_manifest, validate_manifest, write_manifest
from src.data.metadata import MetadataEntry
from src.data.validate import (
    STATUS_BAD_LABEL,
    STATUS_CORRUPT,
    STATUS_MISSING,
    STATUS_NO_AUDIO,
    STATUS_NO_VIDEO,
    STATUS_OK,
    STATUS_TOO_LONG,
    STATUS_TOO_SHORT,
    ProbeResult,
    classify,
    summarise,
)


def entry(**kw) -> MetadataEntry:
    base = dict(
        file="train/000001.mp4",
        split="train",
        n_fakes=1,
        fake_periods=[[1.0, 1.5]],
        duration=4.224,
        original=None,
        modify_video=True,
        modify_audio=False,
        video_frames=103,
        audio_frames=65536,
        audio_channels=1,
    )
    base.update(kw)
    return MetadataEntry(**base)


def probe(**kw) -> ProbeResult:
    base = dict(exists=True, readable=True, has_video=True, has_audio=True)
    base.update(kw)
    return ProbeResult(**base)


class TestQuarantine:
    def test_clean_entry_is_ok(self):
        assert classify(entry(), probe()) == STATUS_OK

    def test_metadata_only_mode_skips_file_checks(self):
        """probe=None is the PF-6 mode: video lives on Kaggle, not here."""
        assert classify(entry(), None) == STATUS_OK

    @pytest.mark.parametrize(
        ("kw", "expected"),
        [
            ({"exists": False, "readable": False}, STATUS_MISSING),
            ({"readable": False}, STATUS_CORRUPT),
            ({"has_video": False}, STATUS_NO_VIDEO),
            ({"has_audio": False}, STATUS_NO_AUDIO),
        ],
    )
    def test_file_integrity_statuses(self, kw, expected):
        assert classify(entry(), probe(**kw)) == expected

    def test_too_short(self):
        # 20 frames / 25 fps = 0.8 s, under the 1 s floor
        assert classify(entry(video_frames=20, fake_periods=[]), None) == STATUS_TOO_SHORT

    def test_too_long(self):
        assert classify(entry(video_frames=100_000, fake_periods=[]), None) == STATUS_TOO_LONG

    def test_bad_label_span_past_end_of_video(self):
        """The real LAV-DF case: 4 entries with spans past the video timeline."""
        assert (
            classify(entry(video_frames=103, fake_periods=[[1.0, 99.0]]), None) == STATUS_BAD_LABEL
        )

    def test_bad_label_inverted_span(self):
        assert classify(entry(fake_periods=[[2.0, 1.0]]), None) == STATUS_BAD_LABEL

    def test_bad_label_negative_start(self):
        assert classify(entry(fake_periods=[[-1.0, 1.0]]), None) == STATUS_BAD_LABEL

    def test_bad_label_checked_against_video_not_metadata_duration(self):
        """A span inside `duration` but past video_frames/25 is still bad.

        This is exactly the gap the padded `duration` opens: 4.12 s of video but a
        `duration` of 4.224 s. A span ending at 4.20 s has no frames to land on.
        """
        assert (
            classify(entry(video_frames=103, fake_periods=[[3.0, 4.20]]), None) == STATUS_BAD_LABEL
        )

    def test_summarise_includes_zeros(self):
        counts = summarise([STATUS_OK, STATUS_OK, STATUS_BAD_LABEL])
        assert counts[STATUS_OK] == 2
        assert counts[STATUS_BAD_LABEL] == 1
        assert counts[STATUS_CORRUPT] == 0  # present, not omitted


class TestLeakage:
    """Section 3.5 RULE 3 -- these must fail the build, not warn."""

    def test_clean_splits_pass(self):
        by = source_ids_by_split([("train", "a"), ("dev", "b"), ("test", "c")])
        assert check_no_leakage(by)["clean"] is True

    @pytest.mark.parametrize(("a", "b"), [("train", "dev"), ("train", "test"), ("dev", "test")])
    def test_each_pair_is_checked(self, a, b):
        by = source_ids_by_split([(a, "shared"), (b, "shared")])
        with pytest.raises(LeakageError, match="SPLIT LEAKAGE"):
            check_no_leakage(by)

    def test_fake_in_test_with_original_in_train_is_caught(self):
        """The concrete R3 failure: a derivative separated from its source."""
        by = source_ids_by_split([("train", "train_000000"), ("test", "train_000000")])
        with pytest.raises(LeakageError):
            check_no_leakage(by)

    def test_report_returned_without_raising_when_asked(self):
        by = source_ids_by_split([("train", "x"), ("test", "x")])
        report = check_no_leakage(by, raise_on_fail=False)
        assert report["clean"] is False
        assert report["train^test"] == 1


class TestBuildManifest:
    def test_schema_and_values(self, tmp_path):
        real = entry(
            file="train/000000.mp4", original=None, modify_video=False, n_fakes=0, fake_periods=[]
        )
        fake = entry(file="train/000001.mp4", original="train/000000.mp4")
        df = build_manifest([real, fake], tmp_path)

        assert list(df.columns) == MANIFEST_COLUMNS
        assert len(df) == 2
        # source_id links the fake to its original
        assert df.loc[df.video_id == "train_000001", "source_id"].iloc[0] == "train_000000"
        # duration is the video timeline, and metadata's value is preserved separately
        assert df["duration"].iloc[0] == pytest.approx(103 / 25)
        assert df["duration_meta"].iloc[0] == pytest.approx(4.224)

    def test_validate_raises_on_leakage(self, tmp_path):
        real = entry(
            file="train/000000.mp4",
            split="train",
            original=None,
            modify_video=False,
            n_fakes=0,
            fake_periods=[],
        )
        leaked = entry(file="test/000001.mp4", split="test", original="train/000000.mp4")
        df = build_manifest([real, leaked], tmp_path)
        with pytest.raises(LeakageError):
            validate_manifest(df)

    def test_validate_reports_exclusions(self, tmp_path):
        good = entry(file="train/000000.mp4", original=None, n_fakes=0, fake_periods=[])
        bad = entry(file="train/000001.mp4", original=None, fake_periods=[[1.0, 99.0]])
        report = validate_manifest(build_manifest([good, bad], tmp_path))
        assert report["n_ok"] == 1
        assert report["status_counts"][STATUS_BAD_LABEL] == 1

    def test_duplicate_video_id_rejected(self, tmp_path):
        e = entry(file="train/000000.mp4", original=None, n_fakes=0, fake_periods=[])
        with pytest.raises(ValueError, match="duplicate"):
            validate_manifest(build_manifest([e, e], tmp_path))

    def test_write_refuses_to_overwrite(self, tmp_path):
        e = entry(file="train/000000.mp4", original=None, n_fakes=0, fake_periods=[])
        df = build_manifest([e], tmp_path)
        out = tmp_path / "manifest_v1.parquet"
        write_manifest(df, out)
        with pytest.raises(FileExistsError, match="versioned"):
            write_manifest(df, out)
