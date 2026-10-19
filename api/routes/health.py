"""`GET /health` and `GET /api/v1/model-info` (P13-3)."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_model_info, get_service
from api.schemas import HealthOut, ModelInfo
from api.services.inference_service import InferenceService

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthOut, summary="Liveness and queue state")
async def health(request: Request, service: InferenceService = Depends(get_service)) -> HealthOut:
    """Liveness plus queue state; 200 means the model is loaded."""
    return HealthOut(
        status="ok",
        model_loaded=True,  # the lifespan does not yield until the model is loaded
        device=request.app.state.model_info["device"],
        busy=service.busy,
        queued=service.queued,
        uptime_s=round(time.monotonic() - request.app.state.started, 1),
    )


@router.get(
    "/api/v1/model-info", response_model=ModelInfo, summary="Which model is serving, and limits"
)
async def model_info(info: dict[str, Any] = Depends(get_model_info)) -> ModelInfo:
    """Which model is serving, its post-processing and every limit."""
    return ModelInfo(**info)
