"""The AVTFD API: `uvicorn api.main:app --workers 1` (`make api`).

The model is loaded and warmed **once**, in the lifespan, before the server accepts a
request (P13-8): construction plus warm-up is ~40 s of one-off cost (CUDA context, cuDNN
autotuning, numba JIT) that would otherwise land on the first upload. `--workers 1` is not a
default to tune -- a second worker is a second copy of every model, 1.1-2.9 GB of working set (PF-30).

`src/` never imports from `api/` (X-3); this package is a thin shell over
`src.inference.pipeline.VideoAnalyzer`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.errors import install_error_handlers
from api.routes import analyze, health
from api.services.inference_service import Analyzer, InferenceService, file_sha256
from api.services.job_store import JobStore
from api.settings import ALLOWED_EXTENSIONS, ApiSettings, get_settings

log = logging.getLogger("avtfd.api")

AnalyzerFactory = Callable[[ApiSettings], tuple[Analyzer, dict[str, Any]]]

# Vite's dev server (Phase 14). Local only -- deployment is out of scope (§0.2).
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def limits(settings: ApiSettings) -> dict[str, Any]:
    """The upload and job limits, as published by `/api/v1/model-info`."""
    return {
        "max_upload_mb": settings.max_upload_mb,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "min_duration_s": settings.min_duration_s,
        "max_duration_s": settings.max_duration_s,
        "job_timeout_s": settings.job_timeout_s,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "max_queued_jobs": settings.max_queued_jobs,
    }


def load_analyzer(settings: ApiSettings) -> tuple[Analyzer, dict[str, Any]]:
    """Build the one `VideoAnalyzer` this process will ever have, and describe it."""
    import torch

    from src.inference.pipeline import VideoAnalyzer, bundle_parameter_count

    path = settings.model_path
    if not path.is_file():
        raise RuntimeError(f"model file not found: {path} -- run `make export-model` (P12-2)")
    t0 = time.perf_counter()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    analyzer = VideoAnalyzer(bundle, device=None if settings.device == "auto" else settings.device)
    warmup_s = analyzer.warmup() if settings.warmup else None
    kwargs = bundle["model_kwargs"]
    info = {
        "name": path.stem,
        "file": path.as_posix(),
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "sha256": file_sha256(path),
        "device": analyzer.device,
        "threshold": analyzer.threshold,
        "fps": analyzer.video_cfg.target_fps,
        "backbone": bundle["backbone_name"],
        "audio_feature": kwargs["feature"],
        "model_kwargs": kwargs,
        "postprocess": bundle["postprocess"],
        "parameters": bundle_parameter_count(bundle),
        "source": bundle.get("source", {}),
        "warmup_s": None if warmup_s is None else round(warmup_s, 2),
    }
    del bundle  # the weights now live in the analyzer's modules
    log.info(
        "model %s loaded on %s in %.1f s", info["name"], info["device"], time.perf_counter() - t0
    )
    return analyzer, info


def create_app(
    settings: ApiSettings | None = None, *, analyzer_factory: AnalyzerFactory = load_analyzer
) -> FastAPI:
    """Build the app; the model loads in the lifespan, once (P13-8). Tests pass a factory."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load and warm the analyzer before serving; close it on shutdown."""
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        for stale in settings.upload_dir.iterdir():  # left by a crash or a hard stop
            if stale.is_file():
                stale.unlink(missing_ok=True)
        analyzer, info = analyzer_factory(settings)
        store = JobStore(ttl_s=settings.job_ttl_s, max_jobs=settings.max_jobs_kept)
        app.state.settings = settings
        app.state.store = store
        app.state.service = InferenceService(analyzer, store, settings)
        app.state.model_info = {**info, "limits": limits(settings)}
        app.state.started = time.monotonic()
        try:
            yield
        finally:
            app.state.service.shutdown()

    app = FastAPI(
        title="AVTFD API",
        version="0.1.0",
        summary="Audio-visual temporal forgery detection and localization",
        description=(
            "Upload a talking-head video; the job result says whether it is fake and *where*: "
            "forged segments in seconds plus a per-frame score. One analysis runs at a time."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(analyze.router)
    return app


# Cheap: the model loads in the lifespan, when uvicorn starts serving, not on import.
app = create_app()
