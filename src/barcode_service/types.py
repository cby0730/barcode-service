"""Shared contracts: value objects, protocols and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Barcode:
    """A decoded barcode."""

    text: str
    format: str
    strategy: str


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in original-image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@runtime_checkable
class Detector(Protocol):
    """Locates barcode regions in a BGR image."""

    def detect(self, img_bgr: np.ndarray) -> list[Box]: ...


@runtime_checkable
class Upscaler(Protocol):
    """Super-resolution; returns None when the input cannot be upscaled."""

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray | None: ...


@runtime_checkable
class Decoder(Protocol):
    """Decodes barcodes from an image (BGR or grayscale)."""

    name: str

    def decode(self, img: np.ndarray) -> list[Barcode]: ...


class ImageDecodeError(ValueError):
    """The payload is not a decodable image."""


class ImageTooLargeError(ValueError):
    """The payload exceeds the configured byte or pixel limit."""


class BackendUnavailableError(RuntimeError):
    """The requested inference backend is not installed or not usable."""


class ModelNotFoundError(FileNotFoundError):
    """A required model file is missing."""
