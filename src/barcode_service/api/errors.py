"""Mapping from domain exceptions to HTTP errors (400 / 413 / 503, 500 for the rest)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from barcode_service.types import ImageDecodeError, ImageTooLargeError

logger = logging.getLogger(__name__)


class UnavailableError(Exception):
    """The service cannot take the request now (no pipeline loaded, or at capacity)."""

    def __init__(self, detail: str, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


def _json(status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    """Install exception handlers on ``app``."""

    @app.exception_handler(ImageDecodeError)
    async def _bad_image(request: Request, exc: ImageDecodeError) -> JSONResponse:
        return _json(400, str(exc))

    @app.exception_handler(ImageTooLargeError)
    async def _too_large(request: Request, exc: ImageTooLargeError) -> JSONResponse:
        return _json(413, str(exc))

    @app.exception_handler(UnavailableError)
    async def _unavailable(request: Request, exc: UnavailableError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        return _json(503, exc.detail, headers)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Keep the traceback in the log only; clients get a generic message.
        logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return _json(500, "internal server error")
