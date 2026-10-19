"""API settings, read from `AVTFD_*` environment variables or `.env` (see `.env.example`).

Nothing here is hardcoded at call sites, so the same server runs anywhere (PF-1). The limits
are the plan's (Part 2 §M): 200 MB uploads, four containers, 300 s per job, one job at a time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class ApiSettings(BaseSettings):
    """API limits and paths from `AVTFD_*` environment variables (`.env.example`)."""

    model_config = SettingsConfigDict(env_prefix="AVTFD_", env_file=".env", extra="ignore")

    model_path: Path = Path("models/avtfd_g_seed0.pt")
    upload_dir: Path = Path("data/interim/api_uploads")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    warmup: bool = True  # pay CUDA/cuDNN/numba start-up costs at boot, not on request 1 (P13-8)

    max_upload_mb: float = Field(200.0, gt=0)
    # ⛔ P13-9: 6 GB of RAM holds one loaded analyzer (1.1-2.9 GB working set, PF-30), not
    # two. Not tunable up.
    max_concurrent_jobs: int = Field(1, ge=1, le=1)
    max_queued_jobs: int = Field(4, ge=0)  # waiting behind the running one; beyond -> 503
    job_timeout_s: float = Field(300.0, gt=0)

    min_duration_s: float = Field(0.5, ge=0)
    # ~0.45 s of work per second of video (Table 8.2.3), so 300 s of video fits the timeout.
    max_duration_s: float = Field(300.0, gt=0)
    # A job with no face in any frame fails with `no_face` (P13-10). MediaPipe finds none on
    # 4 of 1,159 cached LAV-DF clips, which the model *can* score from audio alone (their
    # visual stream is zeroed, as in training) -- set False to score them instead.
    require_face: bool = True

    job_ttl_s: float = Field(3600.0, gt=0)  # finished jobs are forgotten after this
    max_jobs_kept: int = Field(200, ge=1)

    @property
    def max_upload_bytes(self) -> int:
        """The upload cap in bytes."""
        return int(self.max_upload_mb * 1024**2)


@lru_cache
def get_settings() -> ApiSettings:
    """The process's settings, read once."""
    return ApiSettings()
