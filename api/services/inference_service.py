"""Running analyses: one model, one worker, one job at a time (P13-5, P13-8, P13-9).

The `VideoAnalyzer` is built **once** at startup and shared by every job (P13-8); it is not
thread-safe, and 6 GB of RAM holds one analysis anyway (PF-29, PF-30). So jobs run on
a single worker thread, *and* every run takes a `BoundedSemaphore(1)` -- the executor alone
would serialise them, the semaphore makes the invariant explicit and testable (⛔ P13-9).
Requests beyond `max_queued_jobs` waiting jobs are refused with 503 rather than queued
without limit: each waiting job is a file on disk and a client polling for minutes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from api.errors import ApiError
from api.schemas import ErrorInfo, JobStatus
from api.services.job_store import Job, JobStore
from api.settings import ApiSettings
from src.inference.explain import (
    key_figures,
    modality_summary,
    segment_evidence,
    summary_sentence,
)

log = logging.getLogger("avtfd.api")

LOW_FACE_WARNING = 0.5  # below this fraction of frames with a face, say so in the result


class Analyzer(Protocol):
    """What the service needs from `src.inference.pipeline.VideoAnalyzer` (tests fake it)."""

    threshold: float

    def analyze(
        self,
        path: str | Path,
        *,
        deadline: float | None = None,
        details: bool = False,
        keyframes: bool = False,
    ) -> Any: ...

    def close(self) -> None: ...


def classify_failure(exc: BaseException) -> ErrorInfo:
    """Map an analysis exception to a stable error code (P13-10). Unknown -> internal_error."""
    from src.inference.pipeline import AnalysisTimeoutError
    from src.preprocessing.audio import AudioError
    from src.preprocessing.video import DecodeError

    if isinstance(exc, AnalysisTimeoutError):
        return ErrorInfo(code="timeout", message="the analysis exceeded the job time limit")
    if isinstance(exc, AudioError):
        return ErrorInfo(code="no_audio", message="the audio track could not be decoded")
    if isinstance(exc, DecodeError):
        return ErrorInfo(code="corrupt_video", message="the video could not be fully decoded")
    return ErrorInfo(code="internal_error", message="the analysis failed unexpectedly")


class InferenceService:
    def __init__(
        self,
        analyzer: Analyzer,
        store: JobStore,
        settings: ApiSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.analyzer = analyzer
        self.store = store
        self.settings = settings
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avtfd-infer")
        self._semaphore = threading.BoundedSemaphore(settings.max_concurrent_jobs)
        self._lock = threading.Lock()
        self._pending = 0  # queued + running
        self.peak_concurrency = 0  # observed simultaneous analyses; must stay 1 (P13-9)
        self._active = 0

    # -- admission ---------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._active > 0

    @property
    def queued(self) -> int:
        with self._lock:
            return self._pending - self._active

    def _check_capacity_locked(self) -> None:
        if self._pending >= self.settings.max_concurrent_jobs + self.settings.max_queued_jobs:
            raise ApiError(
                503,
                "queue_full",
                "the server is busy with other analyses; retry shortly",
                {"pending": self._pending},
                headers={"Retry-After": "10"},
            )

    def check_capacity(self) -> None:
        """Refuse before the upload is read, so a full queue never costs 200 MB of disk."""
        with self._lock:
            self._check_capacity_locked()

    def submit(self, job: Job, path: Path) -> None:
        with self._lock:  # re-check and claim atomically: two uploads may finish together
            self._check_capacity_locked()
            self._pending += 1
        self._executor.submit(self._run, job.id, path)

    # -- the worker --------------------------------------------------------

    def _run(self, job_id: str, path: Path) -> None:
        try:
            with self._semaphore:
                with self._lock:
                    self._active += 1
                    self.peak_concurrency = max(self.peak_concurrency, self._active)
                try:
                    self._analyze(job_id, path)
                finally:
                    with self._lock:
                        self._active -= 1
        finally:
            with self._lock:
                self._pending -= 1
            path.unlink(missing_ok=True)

    def _analyze(self, job_id: str, path: Path) -> None:
        self.store.mark_running(job_id)
        deadline = self._clock() + self.settings.job_timeout_s
        try:
            # details + keyframes feed the report; they never change the verdict (pipeline.py).
            res = self.analyzer.analyze(path, deadline=deadline, details=True, keyframes=True)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a structured error
            error = classify_failure(exc)
            if error.code == "internal_error":
                log.exception("job %s failed", job_id)
            self.store.finish(job_id, error=error)
            return

        if self.settings.require_face and res.face_found_fraction == 0.0:
            self.store.finish(
                job_id,
                error=ErrorInfo(
                    code="no_face",
                    message="no face was found in the video; the detector analyses a speaker",
                    detail={"face_found_fraction": round(res.face_found_fraction, 4)},
                ),
            )
            return
        thumbs = [k.jpeg for k in getattr(res, "keyframes", [])]
        self.store.finish(job_id, result=self._result(res, job_id), keyframes=thumbs)

    def _result(self, res: Any, job_id: str) -> dict[str, Any]:
        out = res.as_dict(include_frames=True)
        warnings = []
        if res.face_found_fraction < LOW_FACE_WARNING:
            warnings.append(
                f"a face was found in only {res.face_found_fraction:.0%} of frames; "
                "frames without one are scored from audio alone"
            )
        if res.audio_silent:
            warnings.append("the audio track is silent; the score is unreliable")
        out.update(
            verdict="fake" if res.is_fake else "real",
            threshold=self.analyzer.threshold,
            warnings=warnings,
            evidence=self._evidence(res),
        )
        out["keyframes"] = [
            {**k, "url": f"/api/v1/jobs/{job_id}/keyframes/{i}.jpg"}
            for i, k in enumerate(out.get("keyframes", []))
        ]
        return out

    def _evidence(self, res: Any) -> dict[str, Any]:
        """Key figures, per-span evidence and the modality reading -- shared with the PDF."""
        post = getattr(self.analyzer, "postprocess", None)
        frame_threshold = float(getattr(post, "threshold", 0.5))
        scores = np.asarray(res.frame_scores, dtype=np.float64)
        return {
            "summary": summary_sentence(res.is_fake, res.segments, res.duration_s),
            "frame_threshold": frame_threshold,
            "key_figures": key_figures(
                res.segments, scores, res.duration_s, res.clip_score, res.face_found_fraction
            ),
            "segments": segment_evidence(
                res.segments,
                scores,
                res.fps,
                frame_threshold,
                audio_only=getattr(res, "audio_only_scores", None),
                visual_only=getattr(res, "visual_only_scores", None),
                boundary=getattr(res, "boundary_scores", None),
                face_found=getattr(res, "face_found", None),
            ),
            "modality": modality_summary(
                res.clip_score,
                getattr(res, "audio_only_clip", None),
                getattr(res, "visual_only_clip", None),
            ),
        }

    def job_view(self, job: Job) -> dict[str, Any]:
        return {
            "job_id": job.id,
            "status": job.status,
            "filename": job.filename,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "queue_position": (
                self.store.queue_position(job.id) if job.status is JobStatus.queued else None
            ),
            "media": job.media,
            "result": job.result,
            "error": job.error,
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.analyzer.close()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
