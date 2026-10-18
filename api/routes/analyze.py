"""`POST /api/v1/analyze` and `GET /api/v1/jobs/{id}` (P13-1, P13-2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_app_settings, get_service, get_store
from api.errors import ApiError
from api.schemas import ErrorResponse, JobCreated, JobOut
from api.services.inference_service import InferenceService
from api.services.job_store import JobStore
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
    job = store.get(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", f"no job with id '{job_id[:64]}'")
    return JobOut(**service.job_view(job))
