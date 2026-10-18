"""Pydantic response schemas (P13-4). These are what `/docs` publishes and Phase 14 reads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ErrorInfo(BaseModel):
    code: str = Field(examples=["no_audio"])
    message: str
    detail: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


class MediaInfo(BaseModel):
    """What the upload's container says, before any decoding (P13-6)."""

    container: str
    video_codec: str
    audio_codec: str
    width: int
    height: int
    fps: float | None
    duration_s: float
    size_bytes: int


class Segment(BaseModel):
    start_s: float
    end_s: float
    confidence: float


class AnalysisOut(BaseModel):
    verdict: Literal["fake", "real"]
    is_fake: bool
    clip_score: float = Field(description="P(fake) for the whole clip")
    threshold: float
    n_frames: int
    duration_s: float
    fps: float = Field(description="the model's frame grid; frame i covers [i/fps, (i+1)/fps)")
    segments: list[Segment] = Field(description="forged spans, sorted and non-overlapping")
    frame_scores: list[float] = Field(description="per-frame P(fake), one per model frame")
    face_found_fraction: float
    audio_silent: bool
    warnings: list[str]
    timings_s: dict[str, float]


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    queue_position: int


class JobOut(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    queue_position: int | None = Field(None, description="jobs ahead of this one, while queued")
    media: MediaInfo
    result: AnalysisOut | None = None
    error: ErrorInfo | None = None


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    device: str
    busy: bool
    queued: int
    uptime_s: float


class ModelInfo(BaseModel):
    name: str
    file: str
    size_mb: float
    sha256: str
    device: str
    threshold: float
    fps: float
    backbone: str
    audio_feature: str
    model_kwargs: dict[str, Any]
    postprocess: dict[str, Any]
    parameters: dict[str, int]
    source: dict[str, Any]
    warmup_s: float | None
    limits: dict[str, Any]
