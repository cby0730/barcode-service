"""Barcode decoders and a first-hit-wins combinator."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import zxingcpp
from pyzbar import pyzbar

from barcode_service.types import Barcode, Decoder

logger = logging.getLogger(__name__)

# Canonical format names are upper-case zxing names; map the pyzbar spellings that differ.
_PYZBAR_FORMATS = {"I25": "ITF", "DATABAR_EXP": "DATABAREXPANDED"}


def _bytes_to_text(raw: bytes) -> str:
    """UTF-8 when valid, otherwise latin-1 (lossless, so binary payloads never crash)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


class PyzbarDecoder:
    """Decoder using pyzbar."""

    name = "pyzbar"

    def decode(self, img: np.ndarray) -> list[Barcode]:
        return [
            Barcode(
                text=_bytes_to_text(r.data),
                format=_PYZBAR_FORMATS.get(r.type, r.type),
                strategy="",
            )
            for r in pyzbar.decode(img)
        ]


class ZxingDecoder:
    """Decoder using zxing-cpp."""

    name = "zxingcpp"

    def decode(self, img: np.ndarray) -> list[Barcode]:
        return [
            Barcode(text=r.text, format=r.format.name.upper(), strategy="")
            for r in zxingcpp.read_barcodes(np.ascontiguousarray(img))
        ]


def decode_all(decoders: Sequence[Decoder], img: np.ndarray) -> list[Barcode]:
    """Return the results of the first decoder that finds anything.

    An exception in one decoder is logged at debug level and that decoder is skipped.
    """
    for decoder in decoders:
        try:
            found = decoder.decode(img)
        except Exception:
            logger.debug("decoder %s failed", decoder.name, exc_info=True)
            continue
        if found:
            return found
    return []
