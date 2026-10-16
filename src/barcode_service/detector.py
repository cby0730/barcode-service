"""YOLOv8-format ONNX barcode detector (letterbox + NMS implemented here)."""

from __future__ import annotations

import cv2
import numpy as np

from barcode_service.runtime import Session
from barcode_service.types import Box

_PAD_VALUE = 114  # the grey ultralytics pads with; the model was trained on it


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with aspect ratio onto a ``size`` x ``size`` gray canvas.

    Returns (NCHW float32 RGB tensor in [0, 1], scale ratio, (pad_x, pad_y)).
    """
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), _PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = canvas[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, ratio, (float(pad_x), float(pad_y))


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression over (N, 4) xyxy boxes; returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        inter_w = np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0])
        inter_h = np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1])
        inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
        union = areas[i] + areas[rest] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[overlap <= iou]
    return keep


def postprocess(
    raw: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    conf: float,
    iou: float,
    orig_shape: tuple[int, ...],
) -> list[Box]:
    """Turn raw YOLOv8 output into boxes in original-image coordinates, sorted by score desc."""
    # (1, 4+nc, N) -> (N, 4+nc); class-agnostic NMS, so the best class score per anchor is enough.
    preds = raw[0].T
    scores = preds[:, 4:].max(axis=1)
    mask = scores >= conf
    preds, scores = preds[mask], scores[mask]
    if not len(scores):
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
    img_h, img_w = orig_shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, img_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, img_h)

    keep = nms(xyxy, scores, iou)
    return [Box(*(float(v) for v in xyxy[i]), float(scores[i])) for i in keep]


class OnnxDetector:
    """Detector backed by a ``Session``."""

    def __init__(
        self, session: Session, input_size: int = 640, conf: float = 0.25, iou: float = 0.45
    ) -> None:
        self._session = session
        self._input_name = session.input_names[0]
        self._size = input_size
        self._conf = conf
        self._iou = iou

    def detect(self, img_bgr: np.ndarray) -> list[Box]:
        """Return detected boxes sorted by score, highest first."""
        tensor, ratio, pad = letterbox(img_bgr, self._size)
        raw = self._session.run({self._input_name: tensor})[0]
        return postprocess(raw, ratio, pad, self._conf, self._iou, img_bgr.shape)
