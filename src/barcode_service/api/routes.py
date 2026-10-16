"""Routes: POST /v1/decode, GET /health, GET /ready."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from barcode_service.api.errors import UnavailableError
from barcode_service.api.schemas import BarcodeOut, DecodeRequest, DecodeResponse
from barcode_service.config import Settings
from barcode_service.imaging import decode_base64_image, decode_image
from barcode_service.types import ImageDecodeError, ImageTooLargeError

router = APIRouter()

# Slack for JSON keys / data-URI prefix (JSON) and multipart boundaries and headers.
_JSON_OVERHEAD = 1024
_MULTIPART_OVERHEAD = 4096


async def _bounded_stream(request: Request, limit: int) -> AsyncIterator[bytes]:
    """Yield the request body, failing as soon as it exceeds ``limit`` bytes."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise ImageTooLargeError("request body exceeds the size limit")
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > limit:
            raise ImageTooLargeError("request body exceeds the size limit")
        yield chunk


async def read_payload(request: Request) -> bytes | str:
    """Extract the image: raw bytes from multipart ``file``, or the base64 string from JSON.

    Async only for body reading; the route itself stays sync so inference runs in the threadpool.
    """
    settings: Settings = request.app.state.settings
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    if content_type == "application/json":
        # Base64 inflates the payload by 4/3.
        limit = settings.max_image_bytes * 4 // 3 + _JSON_OVERHEAD
        body = b"".join([chunk async for chunk in _bounded_stream(request, limit)])
        try:
            return DecodeRequest.model_validate_json(body).image
        except ValidationError as exc:
            raise HTTPException(422, "body must be JSON with a string field 'image'") from exc

    if content_type == "multipart/form-data":
        limit = settings.max_image_bytes + _MULTIPART_OVERHEAD
        parser = MultiPartParser(request.headers, _bounded_stream(request, limit))
        try:
            form = await parser.parse()
        except MultiPartException as exc:
            raise ImageDecodeError(f"malformed multipart body: {exc.message}") from exc
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(422, "multipart body must contain a file field 'file'")
            return await upload.read()
        finally:
            await form.close()

    raise HTTPException(415, "use application/json or multipart/form-data")


@router.post("/v1/decode", response_model=DecodeResponse)
def decode(request: Request, payload: Annotated[bytes | str, Depends(read_payload)]):
    state = request.app.state
    pipeline = state.pipeline
    if pipeline is None:
        raise UnavailableError("models are not loaded")

    settings: Settings = state.settings
    limits = {"max_bytes": settings.max_image_bytes, "max_pixels": settings.max_image_pixels}
    if isinstance(payload, str):
        img = decode_base64_image(payload, **limits)
    else:
        img = decode_image(payload, **limits)

    # Non-blocking: shed load instead of queueing behind inference.
    if not state.slots.acquire(blocking=False):
        raise UnavailableError("server is at capacity", retry_after=1)
    try:
        barcodes = pipeline.decode(img)
    finally:
        state.slots.release()
    return DecodeResponse(
        barcodes=[BarcodeOut(text=b.text, format=b.format, strategy=b.strategy) for b in barcodes]
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> dict[str, str]:
    if request.app.state.pipeline is None:
        raise UnavailableError("models are not loaded")
    return {"status": "ready"}
