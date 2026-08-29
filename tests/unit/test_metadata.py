"""P1-5 / P1-7 unit tests: metadata parsing, unit assertions, source_id derivation.

These run on synthetic fixtures so they work with no dataset present. The same
assertions run against the real 136,304-entry file in tests/integration/.
"""

from __future__ import annotations

import json

import pytest

from src.data.metadata import (
    MetadataEntry,
    assert_fake_periods_are_seconds,
    assert_frame_timeline,
    assert_n_fakes_matches,
    derive_source_id,
    load_metadata,
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


class TestDerivedFields:
    def test_video_id_is_split_prefixed_and_path_safe(self):
        assert entry(file="train/000001.mp4").video_id == "train_000001"

    @pytest.mark.parametrize(
        ("mv", "ma", "label", "name"),
        [
            (False, False, 0, "real"),
            (True, False, 1, "visual_only"),
            (False, True, 1, "audio_only"),
            (True, True, 1, "both"),
        ],
    )
    def test_four_class_taxonomy(self, mv, ma, label, name):
        e = entry(modify_video=mv, modify_audio=ma)
        assert e.label == label
        assert e.class_name == name

    def test_video_timeline_uses_frames_not_duration(self):
        """The P2-1 correction: the video timeline is video_frames/25."""
        e = entry(video_frames=103, duration=4.224)
        assert e.video_duration_s == pytest.approx(4.12)
        assert e.video_duration_s < e.duration  # metadata duration is padded


class TestUnitAssertions:
    def test_seconds_pass(self):
        ev = assert_fake_periods_are_seconds([entry()])
        assert ev["all_ends_integral"] is False
        assert ev["n_out_of_bounds"] == 0

    def test_frame_indices_are_rejected(self):
        """The ~25x catastrophe: integral ends that overshoot duration."""
        frame_indexed = [
            entry(file=f"train/{i:06d}.mp4", fake_periods=[[25.0, 38.0]], duration=4.224)
            for i in range(10)
        ]
        with pytest.raises(AssertionError, match="FRAME INDICES"):
            assert_fake_periods_are_seconds(frame_indexed)

    def test_a_few_out_of_bounds_is_tolerated_as_bad_label(self):
        """4 of 136,304 is dirty data, not a parsing bug -- must not raise."""
        entries = [entry(file=f"train/{i:06d}.mp4") for i in range(1000)]
        entries[0] = entry(file="train/000000.mp4", fake_periods=[[1.0, 99.0]])
        ev = assert_fake_periods_are_seconds(entries)
        assert ev["n_out_of_bounds"] == 1

    def test_systematic_overshoot_raises(self):
        """>1% out of bounds means we are parsing something wrong."""
        entries = [entry(file=f"train/{i:06d}.mp4") for i in range(100)]
        for i in range(5):
            entries[i] = entry(file=f"train/{i:06d}.mp4", fake_periods=[[1.0, 99.0]])
        with pytest.raises(AssertionError, match="parsing bug"):
            assert_fake_periods_are_seconds(entries)

    def test_n_fakes_mismatch_raises(self):
        with pytest.raises(AssertionError, match="n_fakes"):
            assert_n_fakes_matches([entry(n_fakes=2, fake_periods=[[1.0, 1.5]])])

    def test_empty_periods_cannot_verify_units(self):
        with pytest.raises(ValueError, match="all-real"):
            assert_fake_periods_are_seconds([entry(n_fakes=0, fake_periods=[])])

    def test_frame_timeline_detects_the_padded_duration(self):
        report = assert_frame_timeline([entry()])
        assert report["duration_equals_padded_audio"] == 1
        assert report["offset_max"] < 0  # video_frames always short of duration*25


class TestSourceId:
    def test_real_video_is_its_own_root(self):
        e = entry(file="train/000000.mp4", original=None)
        assert derive_source_id([e])["train/000000.mp4"] == "train_000000"

    def test_fake_resolves_to_its_original(self):
        real = entry(file="train/000000.mp4", original=None, modify_video=False)
        fake = entry(file="train/000001.mp4", original="train/000000.mp4")
        got = derive_source_id([real, fake])
        assert got["train/000001.mp4"] == "train_000000"
        assert got["train/000000.mp4"] == got["train/000001.mp4"]

    def test_chain_is_followed_to_the_root(self):
        a = entry(file="train/000000.mp4", original=None)
        b = entry(file="train/000001.mp4", original="train/000000.mp4")
        c = entry(file="train/000002.mp4", original="train/000001.mp4")
        got = derive_source_id([a, b, c])
        assert got["train/000002.mp4"] == "train_000000"

    def test_cycle_raises_rather_than_hangs(self):
        a = entry(file="train/a.mp4", original="train/b.mp4")
        b = entry(file="train/b.mp4", original="train/a.mp4")
        with pytest.raises(ValueError, match="cycle"):
            derive_source_id([a, b])

    def test_dangling_original_does_not_crash(self):
        """A fake pointing at a file absent from metadata still gets a root."""
        orphan = entry(file="train/000001.mp4", original="train/missing.mp4")
        assert derive_source_id([orphan])["train/000001.mp4"] == "train_missing"


class TestLoad:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "metadata.min.json"
        p.write_text(
            json.dumps(
                [
                    {
                        "file": "train/000001.mp4",
                        "split": "train",
                        "n_fakes": 1,
                        "fake_periods": [[1.0, 1.5]],
                        "duration": 4.224,
                        "original": None,
                        "modify_video": True,
                        "modify_audio": False,
                        "video_frames": 103,
                        "audio_frames": 65536,
                        "audio_channels": 1,
                    }
                ]
            ),
            encoding="utf-8",
        )
        assert load_metadata(p)[0].video_id == "train_000001"

    def test_missing_field_names_the_entry(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps([{"file": "train/1.mp4"}]), encoding="utf-8")
        with pytest.raises(KeyError, match="train/1.mp4"):
            load_metadata(p)

    def test_non_list_rejected(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"file": "x"}), encoding="utf-8")
        with pytest.raises(TypeError):
            load_metadata(p)
