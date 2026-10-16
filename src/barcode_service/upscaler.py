"""x2 super-resolution backed by an ONNX model."""

from __future__ import annotations

import logging

import numpy as np

from barcode_service.runtime import Session

logger = logging.getLogger(__name__)


class OnnxUpscaler:
    """Upscaler backed by a ``Session``."""

    def __init__(self, session: Session, scale: int = 2) -> None:
        self._session = session
        self._input_name = session.input_names[0]
        self._scale = scale

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """Return the upscaled BGR image, or None if it cannot be upscaled."""
        try:
            if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.size == 0:
                raise ValueError(f"expected a non-empty HxWx3 image, got shape {img_bgr.shape}")
            rgb = img_bgr[:, :, ::-1]
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            out = self._session.run({self._input_name: tensor})[0]
            hwc = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
            h, w = img_bgr.shape[:2]
            if hwc.shape[:2] != (h * self._scale, w * self._scale):
                raise ValueError(f"unexpected output shape {hwc.shape} for input {(h, w)}")
            return np.ascontiguousarray((hwc * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1])
        except Exception:  # noqa: BLE001 - the pipeline treats any failure as "skip this step"
            logger.warning("Upscaling failed; skipping", exc_info=True)
            return None
