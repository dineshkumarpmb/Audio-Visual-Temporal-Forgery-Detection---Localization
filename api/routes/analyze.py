"""`POST /api/v1/analyze` and `GET /api/v1/jobs/{id}` (P13-1, P13-2)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_app_settings, get_model_info, get_service, get_store
from api.errors import ApiError
from api.schemas import ErrorResponse, JobCreated, JobOut, JobStatus
from api.services.inference_service import InferenceService
from api.services.job_store import JobStore
from api.services.report import build_report
from api.services.upload import FILE_FIELD, inspect_media, receive_upload
from api.settings import ApiSettings

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# The body is parsed by hand (streamed to disk, P13-7), so FastAPI cannot infer it from a
# parameter; this tells /docs what to render instead.
_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": [FILE_FIELD],
                    "properties": {
                        FILE_FIELD: {
                            "type": "string",
                            "format": "binary",
                            "description": "mp4 / avi / mov / mkv, with an audio track",
                        }
                    },
                }
            }
        },
    }
}


def _errors(*codes: int) -> dict:
    return {c: {"model": ErrorResponse} for c in codes}


@router.post(
    "/analyze",
    status_code=202,
    response_model=JobCreated,
    openapi_extra=_UPLOAD_BODY,
    responses=_errors(400, 413, 415, 422, 503),
    summary="Upload a video for analysis; returns a job id to poll",
)
async def analyze(
    request: Request,
    service: InferenceService = Depends(get_service),
    store: JobStore = Depends(get_store),
    settings: ApiSettings = Depends(get_app_settings),
) -> JobCreated:
    """Stream the upload to disk, vet it, queue it; 202 with the job id."""
    service.check_capacity()
    upload = await receive_upload(request, settings)
    try:
        media = await run_in_threadpool(inspect_media, upload, settings)
        job = store.create(upload.filename, media)
        service.submit(job, upload.path)
    except BaseException:
        upload.path.unlink(missing_ok=True)
        raise
    return JobCreated(
        job_id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id).path),
        queue_position=store.queue_position(job.id) or 0,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobOut,
    responses=_errors(404),
    summary="Job status, and the result once it has finished",
)
async def get_job(
    job_id: str,
    service: InferenceService = Depends(get_service),
    store: JobStore = Depends(get_store),
) -> JobOut:
    """The job's status, and its result or structured error once finished."""
    job = store.get(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", f"no job with id '{job_id[:64]}'")
    return JobOut(**service.job_view(job))


def _finished_job(store: JobStore, job_id: str):
    job = store.get(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", f"no job with id '{job_id[:64]}'")
    if job.status is not JobStatus.succeeded:
        raise ApiError(
            409,
            "job_not_ready",
            f"the job is {job.status.value}; a report exists only for a finished analysis",
            {"status": job.status.value},
        )
    return job


@router.get(
    "/jobs/{job_id}/keyframes/{index}.jpg",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, **_errors(404, 409)},
    summary="One keyframe thumbnail (listed in the result's `keyframes`)",
)
async def get_keyframe(job_id: str, index: int, store: JobStore = Depends(get_store)) -> Response:
    """One keyframe thumbnail of a finished job, as JPEG."""
    job = _finished_job(store, job_id)
    if not 0 <= index < len(job.keyframes):
        raise ApiError(404, "keyframe_not_found", f"no keyframe {index} for this job")
    return Response(
        job.keyframes[index],
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get(
    "/jobs/{job_id}/report.pdf",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}, **_errors(404, 409)},
    summary="The analysis as a printable PDF report",
)
async def get_report(
    job_id: str,
    service: InferenceService = Depends(get_service),
    store: JobStore = Depends(get_store),
    model_info: dict = Depends(get_model_info),
) -> Response:
    """The finished job as a PDF report, rendered once and cached with the job."""
    job = _finished_job(store, job_id)
    if job.report_pdf is None:  # rendered once, kept beside the job until it expires
        view = service.job_view(job)
        job.report_pdf = await run_in_threadpool(build_report, view, model_info, job.keyframes)
    stem = Path(job.filename).stem[:80] or "video"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return Response(
        job.report_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe}-forgery-report.pdf"'},
    )
