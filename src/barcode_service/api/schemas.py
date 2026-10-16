"""Request and response models."""

from __future__ import annotations

from pydantic import BaseModel


class DecodeRequest(BaseModel):
    """JSON body: base64 string or data URI."""

    image: str


class BarcodeOut(BaseModel):
    text: str
    format: str
    strategy: str


class DecodeResponse(BaseModel):
    barcodes: list[BarcodeOut]
