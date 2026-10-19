"""Structured errors (P13-10): every failure is `{"error": {"code", "message", "detail"}}`.

No path through the API returns a stack trace. Expected failures raise `ApiError` with a
stable machine-readable `code`; anything unexpected is logged server-side and answered with a
generic `internal_error`.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("avtfd.api")


class ApiError(Exception):
    """An expected failure with a stable machine-readable code (P13-10)."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail
        self.headers = headers

    def body(self) -> dict[str, Any]:
        """This error as the API's JSON error body."""
        return error_body(self.code, self.message, self.detail)


def error_body(code: str, message: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """The one error shape every response uses: `{error: {code, message, detail}}`."""
    return {"error": {"code": code, "message": message, "detail": detail}}


_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


def install_error_handlers(app: FastAPI) -> None:
    """Route every exception to a structured JSON error -- never a stack trace."""

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, f"http_{exc.status_code}")
        return JSONResponse(
            error_body(code, str(exc.detail)),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            error_body("invalid_request", "request failed validation", {"errors": errors}),
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            error_body("internal_error", "the server hit an unexpected error"), status_code=500
        )
