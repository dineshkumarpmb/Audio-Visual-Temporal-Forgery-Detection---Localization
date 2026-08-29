"""P2-14: unit tests for the Phase 2 preprocessing path.

These use synthetic inputs so they run with no dataset and no model weights. The parts
that genuinely need real video and real MediaPipe inference — determinism, resumability,
detection rate — are verified by `scripts/10_verify_preprocessing.py`, because faking
them here would prove nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import VideoPreprocessConfig
from src.preprocessing.align import ARCFACE_112, align_face, template, umeyama
from src.preprocessing.cache import (
    cache_path,
    cache_root,
    cache_stats,
    is_cached,
    load_faces,
    save_faces,
)
from src.preprocessing.face import choose_face, five_points, interpolate_track
from src.preprocessing.video import DecodeError, assert_frame_count


class TestTemplate:
    def test_margin_zero_is_arcface(self):
        assert np.allclose(template(112, 0.0), ARCFACE_112)

    def test_scales_with_crop_size(self):
        assert np.allclose(template(224, 0.0), ARCFACE_112 * 2)

    def test_margin_shrinks_the_face_toward_centre(self):
        """Larger margin => face occupies less of the crop => more context."""
        spans = [np.ptp(template(112, m), axis=0).mean() for m in (0.0, 0.1, 0.25)]
        assert spans[0] > spans[1] > spans[2]

    def test_margin_scales_toward_the_crop_centre(self):
        """Not toward the landmark centroid: the face is centred in the crop as it shrinks."""
        centre = np.array([56.0, 56.0])
        base = ARCFACE_112.mean(axis=0)
        for m in (0.0, 0.25, 1.0):
            expected = centre + (base - centre) / (1.0 + m)
            assert np.allclose(template(112, m).mean(axis=0), expected, atol=1e-9)

    def test_margin_moves_the_face_centroid_toward_the_crop_centre(self):
        centre = np.array([56.0, 56.0])
        d0 = np.linalg.norm(template(112, 0.0).mean(axis=0) - centre)
        d1 = np.linalg.norm(template(112, 0.25).mean(axis=0) - centre)
        assert d1 < d0

    def test_negative_margin_rejected(self):
        with pytest.raises(ValueError, match="margin"):
            template(112, -0.1)


class TestUmeyama:
    def test_identity(self):
        pts = ARCFACE_112.copy()
        m = umeyama(pts, pts)
        assert np.allclose(m[:, :2], np.eye(2), atol=1e-9)
        assert np.allclose(m[:, 2], 0, atol=1e-9)

    def test_recovers_translation(self):
        pts = ARCFACE_112.copy()
        m = umeyama(pts, pts + np.array([10.0, -5.0]))
        assert np.allclose(m[:, 2], [10.0, -5.0], atol=1e-8)

    def test_recovers_scale(self):
        pts = ARCFACE_112.copy()
        m = umeyama(pts, pts * 2.0)
        assert np.allclose(np.linalg.norm(m[:, 0]), 2.0, atol=1e-8)

    def test_recovers_rotation(self):
        theta = np.deg2rad(30.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pts = ARCFACE_112.copy()
        m = umeyama(pts, pts @ rot.T)
        assert np.allclose(m[:, :2], rot, atol=1e-8)

    def test_never_mirrors(self):
        """A similarity transform may rotate but must not reflect."""
        pts = ARCFACE_112.copy()
        flipped = pts * np.array([-1.0, 1.0])
        assert np.linalg.det(umeyama(pts, flipped)[:, :2]) > 0

    def test_deterministic(self):
        """P2-11 rests on this: no RNG anywhere in the solve."""
        a, b = ARCFACE_112.copy(), ARCFACE_112 * 1.3 + 4.0
        first = umeyama(a, b)
        for _ in range(5):
            assert np.array_equal(umeyama(a, b), first)

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="matching"):
            umeyama(np.zeros((5, 2)), np.zeros((4, 2)))

    def test_too_few_points_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            umeyama(np.zeros((1, 2)), np.zeros((1, 2)))

    def test_degenerate_points_do_not_divide_by_zero(self):
        same = np.zeros((5, 2))
        assert np.all(np.isfinite(umeyama(same, ARCFACE_112)))


class TestAlignFace:
    def test_output_shape_and_dtype(self):
        frame = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        out = align_face(frame, ARCFACE_112 * 2, crop_size=112)
        assert out.shape == (112, 112, 3)
        assert out.dtype == np.uint8

    def test_landmarks_land_on_the_template(self):
        """A synthetic frame with marks at known points must map them to the template."""
        import cv2

        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        src = ARCFACE_112 * 2.0
        for x, y in src:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)

        out = align_face(frame, src, crop_size=112, margin=0.0)
        for x, y in template(112, 0.0):
            patch = out[int(y) - 3 : int(y) + 4, int(x) - 3 : int(x) + 4]
            assert patch.max() > 128, "landmark did not land on its template position"

    def test_deterministic(self):
        frame = np.random.default_rng(1).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        first = align_face(frame, ARCFACE_112 * 2)
        assert np.array_equal(align_face(frame, ARCFACE_112 * 2), first)


class TestChooseFace:
    def test_single_candidate(self):
        pts = ARCFACE_112.copy()
        assert np.array_equal(choose_face([pts], None), pts)

    def test_no_candidates_raises(self):
        with pytest.raises(ValueError, match="no candidates"):
            choose_face([], None)

    def test_first_frame_prefers_the_largest(self):
        small, large = ARCFACE_112.copy(), ARCFACE_112 * 3
        assert np.array_equal(choose_face([small, large], None), large)

    def test_afterwards_prefers_track_continuity_over_size(self):
        """P2-4: a briefly-larger background face must not steal the track."""
        tracked = ARCFACE_112.copy()
        intruder = ARCFACE_112 * 3 + np.array([400.0, 400.0])
        chosen = choose_face([intruder, tracked + 1.0], previous=tracked)
        assert np.allclose(chosen, tracked + 1.0)


class TestInterpolateTrack:
    def test_empty_gives_nothing_found(self):
        track = interpolate_track({}, 10, max_gap=5)
        assert track.points.shape == (10, 5, 2)
        assert not track.found.any()
        assert track.found_fraction == 0.0

    def test_interpolates_within_the_gap(self):
        a, b = np.zeros((5, 2)), np.ones((5, 2)) * 10
        track = interpolate_track({0: a, 10: b}, 11, max_gap=15)
        assert track.found.all()
        assert np.allclose(track.points[5], 5.0)  # midpoint

    def test_does_not_interpolate_across_a_wide_gap(self):
        """A long gap means the face left; inventing landmarks there fabricates data."""
        a, b = np.zeros((5, 2)), np.ones((5, 2)) * 10
        track = interpolate_track({0: a, 30: b}, 31, max_gap=15)
        assert track.found[0] and track.found[30]
        assert not track.found[1:30].any()
        assert np.allclose(track.points[15], 0.0)  # held, not interpolated

    def test_edges_are_held_but_not_claimed(self):
        mid = np.ones((5, 2)) * 4
        track = interpolate_track({5: mid}, 10, max_gap=15)
        assert track.found[5]
        assert not track.found[0] and not track.found[9]
        assert np.allclose(track.points[0], mid)  # held for shape, flagged unfound

    def test_endpoints_are_marked_found(self):
        track = interpolate_track({0: np.zeros((5, 2)), 4: np.ones((5, 2))}, 5, max_gap=10)
        assert track.found[0] and track.found[4]


class TestFivePoints:
    def test_scales_normalised_landmarks_to_pixels(self):
        class LM:
            def __init__(self, x, y):
                self.x, self.y = x, y

        lms = [LM(0.5, 0.5) for _ in range(478)]
        pts = five_points(lms, 224, 112)
        assert pts.shape == (5, 2)
        assert np.allclose(pts[:, 0], 112.0)
        assert np.allclose(pts[:, 1], 56.0)


class TestFrameCountAssertion:
    def test_exact_match_passes(self):
        assert_frame_count(103, 103, "x.mp4")

    def test_off_by_one_tolerated(self):
        assert_frame_count(102, 103, "x.mp4")

    def test_pf7_sized_error_raises(self):
        """round(duration*25) overshoots video_frames by 2-5 -- that must not pass."""
        with pytest.raises(DecodeError, match="PF-7"):
            assert_frame_count(103, 108, "x.mp4")


class TestCache:
    def test_path_is_content_hashed(self, tmp_path):
        a = cache_root(VideoPreprocessConfig(), tmp_path)
        b = cache_root(VideoPreprocessConfig(crop_size=224), tmp_path)
        assert a != b

    def test_roundtrip(self, tmp_path):
        cfg = VideoPreprocessConfig()
        crops = np.random.default_rng(0).integers(0, 255, (7, 112, 112, 3), dtype=np.uint8)
        found = np.array([True] * 5 + [False] * 2)
        save_faces("v1", crops, found, cfg, tmp_path)
        got_crops, got_found = load_faces("v1", cfg, tmp_path)
        assert np.array_equal(np.asarray(got_crops), crops)
        assert np.array_equal(np.asarray(got_found), found)

    def test_is_cached_reflects_reality(self, tmp_path):
        cfg = VideoPreprocessConfig()
        assert not is_cached("v1", cfg, tmp_path)
        save_faces("v1", np.zeros((2, 112, 112, 3), np.uint8), np.ones(2, bool), cfg, tmp_path)
        assert is_cached("v1", cfg, tmp_path)

    def test_zero_byte_file_is_not_cached(self, tmp_path):
        """A truncated write must not be mistaken for completed work."""
        cfg = VideoPreprocessConfig()
        p = cache_path("v1", cfg, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        assert not is_cached("v1", cfg, tmp_path)

    def test_no_temp_file_survives_a_successful_write(self, tmp_path):
        cfg = VideoPreprocessConfig()
        save_faces("v1", np.zeros((2, 112, 112, 3), np.uint8), np.ones(2, bool), cfg, tmp_path)
        assert not list(cache_root(cfg, tmp_path).glob("*.tmp"))

    def test_different_config_does_not_see_the_other_cache(self, tmp_path):
        crops, found = np.zeros((2, 112, 112, 3), np.uint8), np.ones(2, bool)
        save_faces("v1", crops, found, VideoPreprocessConfig(), tmp_path)
        assert not is_cached("v1", VideoPreprocessConfig(crop_size=224), tmp_path)

    def test_dtype_guards(self, tmp_path):
        cfg = VideoPreprocessConfig()
        with pytest.raises(TypeError, match="uint8"):
            save_faces("v", np.zeros((2, 4, 4, 3), np.float32), np.ones(2, bool), cfg, tmp_path)
        with pytest.raises(TypeError, match="bool"):
            save_faces("v", np.zeros((2, 4, 4, 3), np.uint8), np.ones(2, np.uint8), cfg, tmp_path)

    def test_length_mismatch_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="length mismatch"):
            save_faces(
                "v",
                np.zeros((3, 4, 4, 3), np.uint8),
                np.ones(2, bool),
                VideoPreprocessConfig(),
                tmp_path,
            )

    def test_stats(self, tmp_path):
        cfg = VideoPreprocessConfig()
        assert cache_stats(cfg, tmp_path)["n"] == 0
        save_faces("v1", np.zeros((2, 112, 112, 3), np.uint8), np.ones(2, bool), cfg, tmp_path)
        stats = cache_stats(cfg, tmp_path)
        assert stats["n"] == 1
        assert stats["bytes"] > 0
        assert stats["partial"] == 0
