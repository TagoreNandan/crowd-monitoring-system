"""Structured error handling for the API.

Every error response uses the same envelope:

    {"error": {"code": "...", "message": "..."}}

so frontend/API clients can branch on `error.code` instead of parsing
strings. Unexpected exceptions are logged with a full traceback server-side
but never leak internal details to the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ApiError(Exception):
    """Base class for errors that should be reported to the client as a
    clean, typed JSON error rather than an opaque 500."""

    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class InvalidUploadError(ApiError):
    def __init__(self, message: str):
        super().__init__(400, "invalid_upload", message)


class JobNotFoundError(ApiError):
    def __init__(self, job_id: str):
        super().__init__(404, "job_not_found", f"No job found with id '{job_id}'.")


class PipelineUnavailableError(ApiError):
    def __init__(self, message: str):
        super().__init__(503, "pipeline_unavailable", message)


class VideoNotAvailableError(ApiError):
    def __init__(self, job_id: str, message: str):
        super().__init__(404, "video_not_available", message)
        self.job_id = job_id


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        logger.warning(
            "API error on %s %s [%d %s]: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(code="validation_error", message="Request validation failed."),
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        logger.warning(
            "HTTP exception on %s %s [%d]: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(code=f"http_{exc.status_code}", message=str(exc.detail)),
            ).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error processing %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(code="internal_error", message="An unexpected error occurred."),
            ).model_dump(),
        )
