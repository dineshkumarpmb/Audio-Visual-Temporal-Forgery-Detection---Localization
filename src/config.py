"""Validated configuration (P1-19) + content-hashed caching (P1-18, §3.8 rule 1).

Configs are pydantic models, not raw dicts, so a typo is a startup error rather
than a silently-default-valued run three hours in.

The cache key is a sha256 over the *canonical* dump of a config. Change any
preprocessing parameter and the feature cache path changes with it, which makes
mixing feature versions structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# LAV-DF ground truth, measured in P1-5/P1-13 (see reports/dataset_statistics.md).
# These are constants of the dataset, not tunables — hence module scope, not a config.
LAVDF_FPS = 25.0
LAVDF_SAMPLE_RATE = 16_000
LAVDF_AUDIO_CHANNELS = 1
# metadata `duration` == audio_frames/16000 + 0.128 for all 136,304 entries. It is a
# padded audio-derived figure, NOT a media duration: it exceeds both the video and the
# audio stream duration on every file probed. Use `video_frames` as the video timeline.
LAVDF_DURATION_PAD_S = 0.128


class StrictModel(BaseModel):
    """Reject unknown keys. A misspelled field must fail, not be ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PreprocessConfig(StrictModel):
    """Parameters that change the *content* of cached features (§3.8)."""

    target_fps: float = Field(default=LAVDF_FPS, gt=0)
    crop_size: int = Field(default=112, gt=0)
    detect_every_n: int = Field(default=5, ge=1)
    face_margin: float = Field(default=0.25, ge=0)
    align: bool = True

    audio_sample_rate: int = Field(default=LAVDF_SAMPLE_RATE, gt=0)
    n_mels: int = Field(default=64, gt=0)
    n_mfcc: int = Field(default=40, gt=0)
    win_length_ms: float = Field(default=25.0, gt=0)
    hop_length_ms: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _hop_fits_window(self) -> PreprocessConfig:
        if self.hop_length_ms > self.win_length_ms:
            raise ValueError(
                f"hop_length_ms ({self.hop_length_ms}) > win_length_ms ({self.win_length_ms}): "
                "frames would not overlap and audio would be undersampled"
            )
        return self


class SubsetConfig(StrictModel):
    """§3.9 development subsets. Sizes are targets; stratification may round down."""

    name: Literal["smoke-100", "dev-2k", "dev-10k", "full"] = "smoke-100"
    size: int | None = Field(default=None, gt=0)
    seed: int = 1337


class DataConfig(StrictModel):
    data_root: Path = Path("data/raw")
    manifest_path: Path = Path("data/manifests/manifest_v1.parquet")
    min_duration_s: float = Field(default=1.0, gt=0)
    max_duration_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _duration_window_is_sane(self) -> DataConfig:
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError(
                f"min_duration_s ({self.min_duration_s}) >= max_duration_s ({self.max_duration_s})"
            )
        return self


def config_hash(cfg: BaseModel, length: int = 8) -> str:
    """Content hash of a config — the cache key from §3.8 rule 1.

    Canonical JSON (sorted keys, no whitespace variance) so that logically identical
    configs hash identically regardless of field order or how they were constructed.
    """
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def cache_dir(cfg: PreprocessConfig, root: Path = Path("data/features")) -> Path:
    """features/{sha256(config)[:8]}/ — change a parameter, get a new directory."""
    return root / config_hash(cfg)


def load_yaml(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load and validate a YAML config. Unknown keys raise."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return model.model_validate(data)
