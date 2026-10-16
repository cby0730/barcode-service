"""Bytes / base64 to BGR ndarray, with size limits."""

from __future__ import annotations

import base64
import binascii
import struct

import cv2
import numpy as np

from barcode_service.types import ImageDecodeError, ImageTooLargeError

# JPEG start-of-frame markers; C4 (DHT), C8 (JPG) and CC (DAC) share the range but are not frames.
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _png_size(d: bytes) -> tuple[int, int] | None:
    if d[:8] == b"\x89PNG\r\n\x1a\n" and len(d) >= 24:
        return struct.unpack(">II", d[16:24])
    return None


def _jpeg_size(d: bytes) -> tuple[int, int] | None:
    if d[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 4 <= len(d):
        if d[i] != 0xFF:
            return None
        marker = d[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
        elif marker in _JPEG_SOF:
            if i + 9 > len(d):
                return None
            h, w = struct.unpack(">HH", d[i + 5 : i + 9])
            return w, h
        elif marker == 0x01 or 0xD0 <= marker <= 0xD9:  # markers without a length field
            i += 2
        else:
            i += 2 + struct.unpack(">H", d[i + 2 : i + 4])[0]
    return None


def _bmp_size(d: bytes) -> tuple[int, int] | None:
    if d[:2] == b"BM" and len(d) >= 26:
        w, h = struct.unpack("<ii", d[18:26])
        return abs(w), abs(h)  # negative height means top-down
    return None


def _gif_size(d: bytes) -> tuple[int, int] | None:
    if d[:4] == b"GIF8" and len(d) >= 10:
        return struct.unpack("<HH", d[6:10])
    return None


def _webp_size(d: bytes) -> tuple[int, int] | None:
    if d[:4] != b"RIFF" or d[8:12] != b"WEBP" or len(d) < 30:
        return None
    kind = d[12:16]
    if kind == b"VP8 ":
        w, h = struct.unpack("<HH", d[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L":
        bits = int.from_bytes(d[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if kind == b"VP8X":
        return (
            int.from_bytes(d[24:27], "little") + 1,
            int.from_bytes(d[27:30], "little") + 1,
        )
    return None


def _sniff_size(data: bytes) -> tuple[int, int] | None:
    """Read (width, height) from the container header without decoding pixels."""
    for parse in (_png_size, _jpeg_size, _webp_size, _bmp_size, _gif_size):
        try:
            size = parse(data)
        except struct.error:
            return None
        if size is not None:
            return size
    return None


def decode_image(data: bytes, *, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Decode encoded image bytes to a BGR array.

    Raises ImageTooLargeError (bytes or pixels over limit) or ImageDecodeError.
    """
    if len(data) > max_bytes:
        raise ImageTooLargeError(f"image is {len(data)} bytes, limit is {max_bytes}")
    # A tiny file can declare enormous dimensions (decompression bomb), so check the
    # header first. Formats we cannot size are refused rather than decoded blindly.
    size = _sniff_size(data)
    if size is None:
        raise ImageDecodeError(
            "unsupported or corrupt image (expected PNG, JPEG, WebP, BMP or GIF)"
        )
    pixels = size[0] * size[1]
    if pixels > max_pixels:
        raise ImageTooLargeError(f"image has {pixels} pixels, limit is {max_pixels}")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageDecodeError(f"cannot decode image: {exc}") from exc
    if img is None:
        raise ImageDecodeError("cannot decode image")
    return img


def decode_base64_image(s: str, *, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Like ``decode_image`` for a data URI or raw base64 string."""
    if s.startswith("data:"):
        _, sep, s = s.partition(",")
        if not sep:
            raise ImageDecodeError("malformed data URI")
    s = "".join(s.split())
    # Reject before allocating the decoded buffer.
    if len(s) * 3 // 4 > max_bytes:
        raise ImageTooLargeError(f"base64 payload exceeds {max_bytes} bytes once decoded")
    try:
        data = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError(f"invalid base64: {exc}") from exc
    return decode_image(data, max_bytes=max_bytes, max_pixels=max_pixels)
