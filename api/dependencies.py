"""FastAPI dependencies: the per-process singletons built in `api.main`'s lifespan."""

from __future__ import annotations

from typing import Any, cast

from fastapi import Request

from api.services.inference_service import InferenceService
from api.services.job_store import JobStore
from api.settings import ApiSettings


def get_service(request: Request) -> InferenceService:
    return cast(InferenceService, request.app.state.service)


def get_store(request: Request) -> JobStore:
    return cast(JobStore, request.app.state.store)


def get_app_settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def get_model_info(request: Request) -> dict[str, Any]:
    return cast(dict[str, Any], request.app.state.model_info)
