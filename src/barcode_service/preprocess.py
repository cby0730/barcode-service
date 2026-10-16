"""Pure image preprocessing functions."""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from barcode_service.types import Box


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR to single-channel uint8; grayscale input is returned as is."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    img: np.ndarray, clip_limit: float = 2.0, grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation; returns grayscale."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return clahe.apply(to_gray(img))


def apply_sauvola(img: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola local binarisation; returns a uint8 image with values 0 or 255."""
    gray = to_gray(img)
    # skimage requires an odd window.
    thresh = threshold_sauvola(gray, window_size=window | 1, k=k)
    return np.where(gray > thresh, 255, 0).astype(np.uint8)


def resize_max_side(img: np.ndarray, max_side: int = 800) -> np.ndarray:
    """Downscale so the longer side is at most ``max_side``; never upscales."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def crop_box(img: np.ndarray, box: Box, margin: float = 0.05) -> np.ndarray:
    """Crop ``box`` expanded by ``margin`` (fraction of its size), clipped to the image."""
    h, w = img.shape[:2]
    mx = (box.x2 - box.x1) * margin
    my = (box.y2 - box.y1) * margin
    x1 = min(max(math.floor(box.x1 - mx), 0), w)
    x2 = min(max(math.ceil(box.x2 + mx), 0), w)
    y1 = min(max(math.floor(box.y1 - my), 0), h)
    y2 = min(max(math.ceil(box.y2 + my), 0), h)
    return np.ascontiguousarray(img[y1:y2, x1:x2])
