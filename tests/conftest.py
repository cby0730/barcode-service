"""Shared fixtures and image helpers: synthetic barcodes and deterministic degradations."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
import zxingcpp

_FORMATS = {
    "ean13": zxingcpp.BarcodeFormat.EAN13,
    "code128": zxingcpp.BarcodeFormat.Code128,
    "qr": zxingcpp.BarcodeFormat.QRCode,
}

EAN13_TEXT = "5901234123457"
CODE128_TEXT = "BARCODE-0042"
QR_TEXT = "https://example.com/item/42"

# Fixed seed so noise is reproducible across runs and machines.
NOISE_SEED = 1234


def make_barcode(kind: str = "ean13", text: str = EAN13_TEXT, scale: int = 3) -> np.ndarray:
    """Render a barcode as a BGR image: black on white with a quiet zone."""
    barcode = zxingcpp.create_barcode(text, _FORMATS[kind])
    gray = np.array(zxingcpp.write_barcode_to_image(barcode, scale=scale, add_quiet_zones=True))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def downscale(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def gaussian_noise(img: np.ndarray, sigma: float = 10.0, seed: int = NOISE_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate(img: np.ndarray, angle: float = 10.0) -> np.ndarray:
    """Rotate about the centre, keeping the canvas size and filling with white."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def low_contrast(img: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Compress the dynamic range around mid-grey by ``factor``."""
    out = (img.astype(np.float32) - 127.5) * factor + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int = 40) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def to_data_uri(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode("ascii")


def _side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    height = max(a.shape[0], b.shape[0])
    pad = lambda im: cv2.copyMakeBorder(  # noqa: E731
        im, 0, height - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return np.hstack([pad(a), pad(b)])


@pytest.fixture
def ean13_img() -> np.ndarray:
    return make_barcode("ean13", EAN13_TEXT)


@pytest.fixture
def code128_img() -> np.ndarray:
    return make_barcode("code128", CODE128_TEXT)


@pytest.fixture
def qr_img() -> np.ndarray:
    return make_barcode("qr", QR_TEXT, scale=4)


@pytest.fixture
def multi_barcode_img(ean13_img: np.ndarray, qr_img: np.ndarray) -> np.ndarray:
    """An EAN-13 and a QR code side by side."""
    return _side_by_side(ean13_img, qr_img)


@pytest.fixture
def png_bytes():
    return to_png_bytes


@pytest.fixture
def data_uri():
    return to_data_uri
