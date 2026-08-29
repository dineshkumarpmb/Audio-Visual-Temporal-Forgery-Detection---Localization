"""P1-18 / P1-19 / P3-2: config validation and content-hashed cache paths."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    LAVDF_FPS,
    AudioPreprocessConfig,
    DataConfig,
    VideoPreprocessConfig,
    cache_dir,
    config_hash,
)
from src.seed import seed_everything


class TestValidation:
    def test_unknown_key_is_an_error_not_a_silent_default(self):
        with pytest.raises(ValidationError, match="(?i)extra"):
            VideoPreprocessConfig(crop_sixe=112)

    def test_inverted_duration_window_rejected(self):
        with pytest.raises(ValidationError, match="(?i)min_duration"):
            DataConfig(min_duration_s=10.0, max_duration_s=1.0)

    def test_non_positive_crop_rejected(self):
        with pytest.raises(ValidationError):
            VideoPreprocessConfig(crop_size=0)

    def test_frozen(self):
        cfg = VideoPreprocessConfig()
        with pytest.raises(ValidationError):
            cfg.crop_size = 224


class TestAudioGrid:
    """Section C's `hop_length=640` is the alignment contract, so it is enforced."""

    def test_defaults_match_the_plan(self):
        cfg = AudioPreprocessConfig()
        assert (cfg.sample_rate, cfg.n_fft, cfg.hop_length) == (16_000, 1024, 640)
        assert (cfg.n_mels, cfg.fmin, cfg.fmax) == (80, 20.0, 7600.0)

    def test_hop_is_exactly_one_video_frame(self):
        cfg = AudioPreprocessConfig()
        assert cfg.hop_ms == 40.0
        assert cfg.frames_per_second == LAVDF_FPS

    def test_hop_that_breaks_the_grid_is_rejected(self):
        """A 10 ms hop is the conventional choice and would silently misalign everything."""
        with pytest.raises(ValidationError, match="audio frames/s"):
            AudioPreprocessConfig(hop_length=160)

    def test_error_names_the_correct_hop(self):
        with pytest.raises(ValidationError, match="hop_length=640"):
            AudioPreprocessConfig(hop_length=512)

    def test_sample_rate_change_requires_matching_hop(self):
        with pytest.raises(ValidationError, match="audio frames/s"):
            AudioPreprocessConfig(sample_rate=22_050)
        # 22050 is not divisible by 25 into an integer hop, but 8000 is
        AudioPreprocessConfig(sample_rate=8_000, hop_length=320, fmax=3800.0)

    def test_window_shorter_than_hop_rejected(self):
        with pytest.raises(ValidationError, match="leave gaps"):
            AudioPreprocessConfig(n_fft=256)

    def test_fmax_above_nyquist_rejected(self):
        with pytest.raises(ValidationError, match="Nyquist"):
            AudioPreprocessConfig(fmax=9000.0)

    def test_fmin_above_fmax_rejected(self):
        with pytest.raises(ValidationError, match="fmin"):
            AudioPreprocessConfig(fmin=8000.0, fmax=7600.0)

    def test_logmel_is_the_default(self):
        """Decision C-1: MFCC's DCT compresses away where vocoder artefacts live."""
        assert AudioPreprocessConfig().feature == "logmel"

    def test_mfcc_arm_available(self):
        assert AudioPreprocessConfig(feature="mfcc").feature == "mfcc"

    def test_unknown_feature_rejected(self):
        with pytest.raises(ValidationError):
            AudioPreprocessConfig(feature="spectrogram")


class TestContentHash:
    def test_identical_configs_hash_identically(self):
        assert config_hash(VideoPreprocessConfig()) == config_hash(VideoPreprocessConfig())

    def test_field_order_does_not_matter(self):
        a = VideoPreprocessConfig(crop_size=112, detect_every_n=5)
        b = VideoPreprocessConfig(detect_every_n=5, crop_size=112)
        assert config_hash(a) == config_hash(b)

    def test_any_video_change_changes_the_cache_dir(self):
        base = VideoPreprocessConfig()
        for kw in (
            {"crop_size": 224},
            {"target_fps": 12.5},
            {"align": False},
            {"detect_every_n": 3},
            {"face_margin": 0.25},
        ):
            assert cache_dir(VideoPreprocessConfig(**kw)) != cache_dir(base), kw

    def test_any_audio_change_changes_the_cache_dir(self):
        base = AudioPreprocessConfig()
        for kw in (
            {"n_mels": 64},
            {"n_fft": 2048},
            {"fmin": 0.0},
            {"feature": "mfcc"},
            {"cmvn": False},
            {"preemphasis": 0.0},
        ):
            assert cache_dir(AudioPreprocessConfig(**kw)) != cache_dir(base), kw

    def test_audio_and_video_caches_are_independent(self):
        """PF-10: retuning n_mels must not invalidate 7.5 MB/video of face crops."""
        video_before = cache_dir(VideoPreprocessConfig())
        _ = cache_dir(AudioPreprocessConfig(n_mels=64))
        assert cache_dir(VideoPreprocessConfig()) == video_before

    def test_cache_dir_is_under_the_feature_root(self):
        assert cache_dir(VideoPreprocessConfig()).parent.name == "features"


class TestSeed:
    def test_reports_what_it_pinned(self):
        state = seed_everything(1337)
        assert state["seed"] == 1337
        assert state["pythonhashseed_effective"] is True

    def test_python_rng_is_reproducible(self):
        import random

        seed_everything(1337)
        a = [random.random() for _ in range(5)]
        seed_everything(1337)
        assert a == [random.random() for _ in range(5)]

    def test_numpy_rng_is_reproducible(self):
        np = pytest.importorskip("numpy")
        seed_everything(1337)
        a = np.random.rand(5).tolist()
        seed_everything(1337)
        assert a == np.random.rand(5).tolist()
