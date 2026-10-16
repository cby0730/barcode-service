"""Real-model checks: detector localisation, exact x2 upscaling and cross-backend agreement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from barcode_service.detector import OnnxDetector
from barcode_service.runtime import create_session
from barcode_service.types import BackendUnavailableError, Box
from barcode_service.upscaler import OnnxUpscaler
from tests.conftest import gaussian_noise, make_barcode

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DETECTOR_ONNX = MODELS_DIR / "yolov8s-barcode.onnx"
UPSCALER_ONNX = MODELS_DIR / "pan-x2.onnx"

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        not (DETECTOR_ONNX.is_file() and UPSCALER_ONNX.is_file()),
        reason="models/*.onnx not present (see scripts/download_models.py, export_models.py)",
    ),
]

CANVAS = (480, 640)  # (height, width)
BACKENDS = [
    pytest.param(
        "ort-cpu",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("onnxruntime") is None, reason="onnxruntime not installed"
        ),
    ),
    pytest.param(
        "openvino",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("openvino") is None, reason="openvino not installed"
        ),
    ),
    pytest.param("ort-cuda", marks=pytest.mark.gpu),
]


def _make_scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """An EAN-13 pasted on a larger noisy canvas; returns (image, (x1, y1, x2, y2) of the paste)."""
    code = make_barcode("ean13", scale=3)
    h, w = code.shape[:2]
    rng = np.random.default_rng(7)
    canvas = np.clip(rng.normal(150, 25, (*CANVAS, 3)), 0, 255).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 3)
    x1, y1 = 220, 180
    canvas[y1 : y1 + h, x1 : x1 + w] = code
    return gaussian_noise(canvas, sigma=4.0), (x1, y1, x1 + w, y1 + h)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def _coords(box: Box) -> tuple[float, float, float, float]:
    return (box.x1, box.y1, box.x2, box.y2)


def _detector(backend: str) -> OnnxDetector:
    try:
        return OnnxDetector(create_session(backend, DETECTOR_ONNX))
    except BackendUnavailableError as exc:
        if backend == "ort-cuda":
            pytest.skip(f"CUDA unavailable: {exc}")
        raise


@pytest.fixture(scope="module")
def scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    return _make_scene()


@pytest.mark.parametrize("backend", BACKENDS)
def test_detector_finds_the_pasted_barcode(backend, scene):
    img, truth = scene
    boxes = _detector(backend).detect(img)
    assert boxes, "no detection"
    assert _iou(_coords(boxes[0]), truth) >= 0.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_upscaler_output_is_exactly_twice_the_input(backend):
    try:
        upscaler = OnnxUpscaler(create_session(backend, UPSCALER_ONNX))
    except BackendUnavailableError as exc:
        if backend == "ort-cuda":
            pytest.skip(f"CUDA unavailable: {exc}")
        raise
    crop = make_barcode("ean13", scale=1)  # odd sizes exercise the dynamic axes
    out = upscaler.upscale(crop)
    assert out is not None
    assert out.shape == (crop.shape[0] * 2, crop.shape[1] * 2, 3)
    assert out.dtype == np.uint8


def test_backends_agree_on_the_same_image(scene):
    img, _ = scene
    results: dict[str, Box] = {}
    for backend in ("ort-cpu", "openvino", "ort-cuda"):
        try:
            boxes = OnnxDetector(create_session(backend, DETECTOR_ONNX)).detect(img)
        except BackendUnavailableError:
            continue  # this backend is not usable in the current environment
        assert boxes, f"{backend} found nothing"
        results[backend] = boxes[0]

    if len(results) < 2:
        pytest.skip(f"need at least two usable backends, got {sorted(results)}")
    names = sorted(results)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert _iou(_coords(results[a]), _coords(results[b])) >= 0.8, (a, b)
