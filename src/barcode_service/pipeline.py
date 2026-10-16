"""Strategy-driven decoding pipeline with early exit."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from barcode_service.decoders import decode_all
from barcode_service.preprocess import (
    apply_clahe,
    apply_sauvola,
    crop_box,
    resize_max_side,
    to_gray,
)
from barcode_service.types import Barcode, Decoder, Detector, Upscaler

logger = logging.getLogger(__name__)

RAW_STRATEGY = "raw"
# Allowed step names: "detect", "upscale", "clahe", "sauvola".
STEPS = frozenset({"detect", "upscale", "clahe", "sauvola"})


@dataclass(frozen=True)
class Strategy:
    """A named, ordered list of steps applied before decoding."""

    name: str
    steps: tuple[str, ...]


# Trimmed with scripts/benchmark.py (synthetic data). "detect",
# "detect+sauvola" and "detect+upscale+clahe" never solved an image the others had not already
# solved, on any backend, so they are gone. Cheap strategies run first; detect+upscale is the
# expensive one but the only one with a sizeable gain.
DEFAULT_STRATEGIES: tuple[Strategy, ...] = (
    Strategy("clahe", ("clahe",)),
    Strategy("sauvola", ("sauvola",)),
    Strategy("detect+clahe", ("detect", "clahe")),
    Strategy("detect+upscale", ("detect", "upscale")),
    Strategy("detect+upscale+sauvola", ("detect", "upscale", "sauvola")),
)


class Pipeline:
    """Tries the raw image first, then each strategy in order, stopping at the first hit."""

    def __init__(
        self,
        decoders: Sequence[Decoder],
        detector: Detector | None = None,
        upscaler: Upscaler | None = None,
        strategies: Sequence[Strategy] = DEFAULT_STRATEGIES,
        max_side: int = 800,
    ) -> None:
        for strategy in strategies:
            unknown = set(strategy.steps) - STEPS
            if unknown:
                raise ValueError(f"strategy {strategy.name!r}: unknown steps {sorted(unknown)}")
        self._decoders = tuple(decoders)
        self._detector = detector
        self._upscaler = upscaler
        self._strategies = tuple(strategies)
        self._max_side = max_side

    def decode(self, img: np.ndarray) -> list[Barcode]:
        """Decode barcodes from a BGR image; per-strategy exceptions are logged and skipped."""
        # Fast path on the untouched image: most clean photos decode here.
        found = decode_all(self._decoders, img)
        if found:
            return _tag(found, RAW_STRATEGY)

        base = resize_max_side(img, self._max_side)
        for strategy in self._strategies:
            try:
                found = decode_all(self._decoders, self._apply(base, strategy.steps))
            except Exception:
                logger.warning("strategy %r failed; skipping", strategy.name, exc_info=True)
                continue
            if found:
                return _tag(found, strategy.name)
        return []

    def _apply(self, base: np.ndarray, steps: Sequence[str]) -> np.ndarray:
        img = base.copy()
        for step in steps:
            if step == "detect":
                if self._detector is not None:
                    boxes = self._detector.detect(img)
                    if boxes:
                        img = crop_box(img, max(boxes, key=lambda b: b.score))
            elif step == "upscale":
                if self._upscaler is not None:
                    upscaled = self._upscaler.upscale(img)
                    if upscaled is not None:
                        img = upscaled
            elif step == "clahe":
                img = apply_clahe(img)
            elif step == "sauvola":
                img = apply_sauvola(to_gray(img))
        return img


def _tag(found: Sequence[Barcode], strategy: str) -> list[Barcode]:
    """Label results with the winning strategy and drop duplicates (same text and format)."""
    unique: dict[tuple[str, str], Barcode] = {}
    for barcode in found:
        unique.setdefault((barcode.text, barcode.format), replace(barcode, strategy=strategy))
    return list(unique.values())
