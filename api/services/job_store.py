"""In-process job store (P13-5).

One process, one worker, a few jobs: a locked dict is the whole requirement. Jobs live in
memory and are forgotten after `job_ttl_s` or past `max_jobs_kept`, oldest finished first --
a restart loses them, which is acceptable for a local single-user server (deployment is out
of scope, §0.2).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from api.schemas import ErrorInfo, JobStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    filename: str
    media: dict[str, Any]
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    # In memory only, dropped with the job (job_ttl_s): thumbnails and the rendered report.
    keyframes: list[bytes] = field(default_factory=list)
    report_pdf: bytes | None = None
    _finished_mono: float | None = None

    @property
    def done(self) -> bool:
        return self.status in (JobStatus.succeeded, JobStatus.failed)


class JobStore:
    def __init__(self, *, ttl_s: float, max_jobs: int) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.ttl_s = ttl_s
        self.max_jobs = max_jobs

    def create(self, filename: str, media: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, media=media)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.running
            job.started_at = _now()

    def finish(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: ErrorInfo | None = None,
        keyframes: list[bytes] | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.failed if error is not None else JobStatus.succeeded
            job.result, job.error = result, error
            job.keyframes = list(keyframes or [])
            job.finished_at = _now()
            job._finished_mono = time.monotonic()

    def queue_position(self, job_id: str) -> int | None:
        """Jobs ahead of this one (queued earlier, or running). None once it has started."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.queued:
                return None
            return sum(
                1
                for j in self._jobs.values()
                if j.status is JobStatus.running
                or (j.status is JobStatus.queued and j.created_at < job.created_at)
            )

    def counts(self) -> dict[JobStatus, int]:
        with self._lock:
            out = dict.fromkeys(JobStatus, 0)
            for j in self._jobs.values():
                out[j.status] += 1
            return out

    def _prune(self) -> None:
        """Drop expired finished jobs, then the oldest finished ones past the cap. Never a
        queued or running job -- its worker still holds the id."""
        now = time.monotonic()
        for jid in [
            jid
            for jid, j in self._jobs.items()
            if j._finished_mono is not None and now - j._finished_mono > self.ttl_s
        ]:
            del self._jobs[jid]
        excess = len(self._jobs) - self.max_jobs
        if excess > 0:
            finished = sorted(
                (j for j in self._jobs.values() if j.done), key=lambda j: j._finished_mono or 0
            )
            for j in finished[:excess]:
                del self._jobs[j.id]
