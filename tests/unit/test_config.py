"""P1-18 / P1-19: config validation and content-hashed cache paths."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import DataConfig, PreprocessConfig, cache_dir, config_hash
from src.seed import seed_everything


class TestValidation:
    def test_unknown_key_is_an_error_not_a_silent_default(self):
        with pytest.raises(Exception, match="(?i)extra"):
            PreprocessConfig(crop_sixe=112)

    def test_hop_wider_than_window_rejected(self):
        with pytest.raises(Exception, match="(?i)overlap|hop"):
            PreprocessConfig(win_length_ms=25.0, hop_length_ms=50.0)

    def test_inverted_duration_window_rejected(self):
        with pytest.raises(Exception, match="(?i)min_duration"):
            DataConfig(min_duration_s=10.0, max_duration_s=1.0)

    def test_non_positive_crop_rejected(self):
        with pytest.raises(ValidationError):
            PreprocessConfig(crop_size=0)

    def test_frozen(self):
        cfg = PreprocessConfig()
        with pytest.raises(ValidationError):
            cfg.crop_size = 224


class TestContentHash:
    def test_identical_configs_hash_identically(self):
        assert config_hash(PreprocessConfig()) == config_hash(PreprocessConfig())

    def test_field_order_does_not_matter(self):
        a = PreprocessConfig(n_mels=64, crop_size=112)
        b = PreprocessConfig(crop_size=112, n_mels=64)
        assert config_hash(a) == config_hash(b)

    def test_any_change_changes_the_cache_dir(self):
        base = PreprocessConfig()
        for kw in ({"n_mels": 80}, {"crop_size": 224}, {"target_fps": 12.5}, {"align": False}):
            assert cache_dir(PreprocessConfig(**kw)) != cache_dir(base), kw

    def test_cache_dir_is_under_the_feature_root(self):
        assert cache_dir(PreprocessConfig()).parent.name == "features"


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
