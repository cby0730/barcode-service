import base64

import cv2
import numpy as np
import pytest

from barcode_service.imaging import decode_base64_image, decode_image
from barcode_service.types import ImageDecodeError, ImageTooLargeError

LIMITS = {"max_bytes": 1_000_000, "max_pixels": 1_000_000}


def test_decode_png_roundtrip(ean13_img, png_bytes):
    out = decode_image(png_bytes(ean13_img), **LIMITS)
    assert out.dtype == np.uint8 and out.ndim == 3
    assert np.array_equal(out, ean13_img)


def test_decode_jpeg(ean13_img):
    ok, buf = cv2.imencode(".jpg", ean13_img)
    assert ok
    assert decode_image(buf.tobytes(), **LIMITS).shape == ean13_img.shape


def test_grayscale_png_becomes_bgr(ean13_img):
    gray = cv2.cvtColor(ean13_img, cv2.COLOR_BGR2GRAY)
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    assert decode_image(buf.tobytes(), **LIMITS).shape == (*gray.shape, 3)


def test_too_many_bytes(ean13_img, png_bytes):
    data = png_bytes(ean13_img)
    with pytest.raises(ImageTooLargeError):
        decode_image(data, max_bytes=len(data) - 1, max_pixels=10**9)


def test_too_many_pixels(ean13_img, png_bytes):
    h, w = ean13_img.shape[:2]
    with pytest.raises(ImageTooLargeError):
        decode_image(png_bytes(ean13_img), max_bytes=10**9, max_pixels=h * w - 1)


def test_pixel_limit_is_inclusive(ean13_img, png_bytes):
    h, w = ean13_img.shape[:2]
    decode_image(png_bytes(ean13_img), max_bytes=10**9, max_pixels=h * w)


def test_decompression_bomb_rejected_without_full_decode(png_bytes):
    # 30000 x 30000 white pixels compress to a tiny PNG; decoding it would need ~2.7 GB.
    huge = np.full((30000, 30000), 255, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", huge)
    assert ok and len(buf) < 5_000_000
    with pytest.raises(ImageTooLargeError):
        decode_image(buf.tobytes(), max_bytes=10_000_000, max_pixels=25_000_000)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_undecodable_bytes(data):
    with pytest.raises(ImageDecodeError):
        decode_image(data, **LIMITS)


def test_truncated_png(ean13_img, png_bytes):
    data = png_bytes(ean13_img)
    with pytest.raises(ImageDecodeError):
        decode_image(data[: len(data) // 2], **LIMITS)


def test_base64_raw_and_data_uri(ean13_img, png_bytes, data_uri):
    raw = base64.b64encode(png_bytes(ean13_img)).decode()
    assert np.array_equal(decode_base64_image(raw, **LIMITS), ean13_img)
    assert np.array_equal(decode_base64_image(data_uri(ean13_img), **LIMITS), ean13_img)


def test_base64_tolerates_whitespace(ean13_img, png_bytes):
    raw = base64.encodebytes(png_bytes(ean13_img)).decode()  # contains newlines
    assert np.array_equal(decode_base64_image(raw, **LIMITS), ean13_img)


@pytest.mark.parametrize("s", ["", "!!!not base64!!!", "data:image/png;base64,", "abc"])
def test_invalid_base64(s):
    with pytest.raises(ImageDecodeError):
        decode_base64_image(s, **LIMITS)


def test_valid_base64_of_non_image():
    with pytest.raises(ImageDecodeError):
        decode_base64_image(base64.b64encode(b"hello world").decode(), **LIMITS)


def test_base64_oversize_rejected_before_decoding(ean13_img, png_bytes):
    raw = base64.b64encode(png_bytes(ean13_img)).decode()
    with pytest.raises(ImageTooLargeError):
        decode_base64_image(raw, max_bytes=100, max_pixels=10**9)


@pytest.mark.parametrize("ext", [".bmp", ".webp", ".jpg"])
def test_other_formats_decode_through_header_sniffing(ean13_img, ext):
    ok, buf = cv2.imencode(ext, ean13_img)
    assert ok
    assert decode_image(buf.tobytes(), **LIMITS).shape == ean13_img.shape


def _bomb_headers() -> dict[str, bytes]:
    huge = 40_000
    return {
        "bmp": b"BM" + bytes(16) + huge.to_bytes(4, "little") * 2 + bytes(8),
        "gif": b"GIF89a" + huge.to_bytes(2, "little") * 2 + bytes(8),
        "webp-vp8x": (
            b"RIFF" + bytes(4) + b"WEBPVP8X" + bytes(8)
            + (huge - 1).to_bytes(3, "little") * 2
        ),
    }  # fmt: skip


@pytest.mark.parametrize("kind", ["bmp", "gif", "webp-vp8x"])
def test_forged_headers_rejected_by_pixel_limit(kind):
    with pytest.raises(ImageTooLargeError):
        decode_image(_bomb_headers()[kind], **LIMITS)


def test_forged_jpeg_header_rejected_by_pixel_limit(ean13_img):
    ok, buf = cv2.imencode(".jpg", ean13_img)
    data = bytearray(buf.tobytes())
    sof = data.index(b"\xff\xc0")
    data[sof + 5 : sof + 9] = (40_000).to_bytes(2, "big") * 2
    with pytest.raises(ImageTooLargeError):
        decode_image(bytes(data), **LIMITS)


def test_data_uri_without_comma_is_rejected():
    with pytest.raises(ImageDecodeError, match="data URI"):
        decode_base64_image("data:image/png;base64", **LIMITS)
