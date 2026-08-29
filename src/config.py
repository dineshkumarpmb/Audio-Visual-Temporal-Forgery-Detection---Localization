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


class VideoPreprocessConfig(StrictModel):
    """Parameters that change the *content* of the cached face crops (section 3.8).

    Split from the audio config on purpose. Both are content-hashed into their own cache
    directory, so tuning `n_mels` does not invalidate 7.5 MB/video of face crops (PF-10)
    and changing `crop_size` does not invalidate the log-mels. A single combined config
    would couple them and force re-extraction of the expensive side for a change to the
    cheap one.
    """

    target_fps: float = Field(default=LAVDF_FPS, gt=0)
    crop_size: int = Field(default=112, gt=0)
    detect_every_n: int = Field(default=5, ge=1)
    # 0.0, not the plan's 0.25 -- decision PF-9. LAV-DF ships 224x224 frames that are
    # already tight VoxCeleb2 face crops, so there is no surrounding context for a
    # margin to include; it only manufactures replicated border. Measured over 12
    # subjects: margin 0.0 -> 11.0% of crop pixels fall outside the source frame,
    # 0.25 -> 21.4%. Kept configurable because a full-frame dataset would want 0.25.
    face_margin: float = Field(default=0.0, ge=0)
    align: bool = True


class AudioPreprocessConfig(StrictModel):
    """PROJECT_PLAN section C, verbatim. **Do not retune `hop_length`.**

    `hop_length=640` at 16 kHz is exactly 40 ms, and one video frame at 25 fps is exactly
    40 ms. That makes audio frame `t` correspond to video frame `t` by array index rather
    than by interpolation, which removes an entire class of alignment bug. Section C calls
    it "the single most important number in the preprocessing" and says to lock it; the
    validator below enforces that rather than trusting it.

    Defaults to log-mel, not MFCC (decision C-1): MFCC's DCT compresses away the fine
    spectral structure where vocoder artefacts live. MFCC stays available as the
    Experiment K arm.
    """

    sample_rate: int = Field(default=LAVDF_SAMPLE_RATE, gt=0)
    n_fft: int = Field(default=1024, gt=0)
    hop_length: int = Field(default=640, gt=0)
    n_mels: int = Field(default=80, gt=0)
    fmin: float = Field(default=20.0, ge=0)
    fmax: float = Field(default=7600.0, gt=0)
    preemphasis: float = Field(default=0.97, ge=0, lt=1)
    cmvn: bool = True
    eps: float = Field(default=1e-10, gt=0)
    feature: Literal["logmel", "mfcc"] = "logmel"
    n_mfcc: int = Field(default=40, gt=0)

    @property
    def frames_per_second(self) -> float:
        """Audio frame rate implied by the hop. Must equal the video fps."""
        return self.sample_rate / self.hop_length

    @property
    def hop_ms(self) -> float:
        return 1000.0 * self.hop_length / self.sample_rate

    @model_validator(mode="after")
    def _grid_locks_to_video(self) -> AudioPreprocessConfig:
        if abs(self.frames_per_second - LAVDF_FPS) > 1e-9:
            raise ValueError(
                f"hop_length={self.hop_length} at sr={self.sample_rate} gives "
                f"{self.frames_per_second:.4f} audio frames/s, but video is {LAVDF_FPS} fps. "
                "Section C requires these to be equal so audio frame t == video frame t. "
                f"For {self.sample_rate} Hz use hop_length={int(self.sample_rate / LAVDF_FPS)}."
            )
        return self

    @model_validator(mode="after")
    def _window_covers_hop(self) -> AudioPreprocessConfig:
        if self.n_fft < self.hop_length:
            raise ValueError(
                f"n_fft ({self.n_fft}) < hop_length ({self.hop_length}): consecutive windows "
                "would leave gaps, so some samples would never be analysed"
            )
        return self

    @model_validator(mode="after")
    def _nyquist(self) -> AudioPreprocessConfig:
        if self.fmax > self.sample_rate / 2:
            raise ValueError(
                f"fmax ({self.fmax}) exceeds Nyquist ({self.sample_rate / 2}) for "
                f"sr={self.sample_rate}"
            )
        if self.fmin >= self.fmax:
            raise ValueError(f"fmin ({self.fmin}) >= fmax ({self.fmax})")
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


def cache_dir(cfg: BaseModel, root: Path = Path("data/features")) -> Path:
    """features/{sha256(config)[:8]}/ — change a parameter, get a new directory."""
    return root / config_hash(cfg)


def load_yaml(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load and validate a YAML config. Unknown keys raise."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return model.model_validate(data)
