"""Letterbox, NMS and YOLOv8 postprocessing, exercised with hand-built tensors."""

from __future__ import annotations

import numpy as np
import pytest

from barcode_service.detector import OnnxDetector, letterbox, nms, postprocess


def _raw(rows: list[tuple[float, ...]]) -> np.ndarray:
    """Build a (1, 4+nc, N) tensor from per-anchor (cx, cy, w, h, *class_scores) rows."""
    return np.array(rows, dtype=np.float32).T[None]


class TestLetterbox:
    def test_wide_image_is_padded_top_and_bottom(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, ratio, (pad_x, pad_y) = letterbox(img, 640)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert ratio == pytest.approx(3.2)
        assert (pad_x, pad_y) == (0.0, 160.0)
        assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)  # padding
        assert tensor[0, 0, 320, 320] == 0.0  # image content

    def test_tall_image_is_padded_left_and_right(self):
        _, ratio, (pad_x, pad_y) = letterbox(np.zeros((200, 100, 3), dtype=np.uint8), 640)
        assert ratio == pytest.approx(3.2)
        assert (pad_x, pad_y) == (160.0, 0.0)

    def test_channels_are_converted_from_bgr_to_rgb(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # blue in BGR
        tensor, _, _ = letterbox(img, 64)
        assert tensor[0, 2].min() == 1.0  # blue ends up in the last RGB channel
        assert tensor[0, 0].max() == 0.0

    def test_large_image_is_downscaled(self):
        _, ratio, _ = letterbox(np.zeros((1280, 1280, 3), dtype=np.uint8), 640)
        assert ratio == pytest.approx(0.5)


class TestNms:
    def test_suppresses_overlapping_lower_score(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.6, 0.9, 0.7], dtype=np.float32)
        assert nms(boxes, scores, 0.45) == [1, 2]

    def test_keeps_boxes_below_iou_threshold(self):
        boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], dtype=np.float32)  # IoU = 2/18
        assert sorted(nms(boxes, np.array([0.9, 0.8], dtype=np.float32), 0.45)) == [0, 1]

    def test_empty_input(self):
        assert nms(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.45) == []

    def test_degenerate_zero_area_boxes_do_not_divide_by_zero(self):
        boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], dtype=np.float32)
        assert nms(boxes, np.array([0.9, 0.8], dtype=np.float32), 0.45) == [0, 1]


class TestPostprocess:
    def test_undoes_pad_and_ratio(self):
        # Letterboxed box centred at (320, 320), 160 wide, 80 tall; ratio 3.2, pad_y 160.
        raw = _raw([(320, 320, 160, 80, 0.9, 0.1)])
        (box,) = postprocess(raw, 3.2, (0.0, 160.0), 0.25, 0.45, (100, 200, 3))
        assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((75.0, 37.5, 125.0, 62.5))
        assert box.score == pytest.approx(0.9)

    def test_filters_by_confidence_using_best_class(self):
        raw = _raw([(100, 100, 20, 20, 0.1, 0.2), (300, 300, 20, 20, 0.1, 0.8)])
        boxes = postprocess(raw, 1.0, (0.0, 0.0), 0.25, 0.45, (640, 640, 3))
        assert len(boxes) == 1
        assert boxes[0].score == pytest.approx(0.8)

    def test_clips_to_image_bounds(self):
        raw = _raw([(5, 5, 40, 40, 0.9, 0.0)])
        (box,) = postprocess(raw, 1.0, (0.0, 0.0), 0.25, 0.45, (30, 50, 3))
        assert (box.x1, box.y1) == (0.0, 0.0)
        assert box.x2 == pytest.approx(25.0)
        assert box.y2 == pytest.approx(25.0)
        raw = _raw([(45, 25, 40, 40, 0.9, 0.0)])
        (box,) = postprocess(raw, 1.0, (0.0, 0.0), 0.25, 0.45, (30, 50, 3))
        assert (box.x2, box.y2) == (50.0, 30.0)

    def test_merges_duplicates_and_sorts_by_score(self):
        raw = _raw(
            [
                (100, 100, 50, 50, 0.6, 0.0),
                (102, 101, 50, 50, 0.9, 0.0),  # duplicate of the first, higher score
                (400, 400, 30, 30, 0.7, 0.0),
            ]
        )
        boxes = postprocess(raw, 1.0, (0.0, 0.0), 0.25, 0.45, (640, 640, 3))
        assert [round(b.score, 1) for b in boxes] == [0.9, 0.7]

    def test_nothing_above_threshold(self):
        raw = _raw([(100, 100, 20, 20, 0.1, 0.1)])
        assert postprocess(raw, 1.0, (0.0, 0.0), 0.25, 0.45, (640, 640, 3)) == []


class FakeSession:
    input_names = ["images"]

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.seen: dict[str, np.ndarray] = {}

    def run(self, inputs):
        self.seen = inputs
        return [self.output]


def test_onnx_detector_end_to_end_with_fake_session():
    session = FakeSession(_raw([(320, 320, 160, 80, 0.9, 0.1)]))
    detector = OnnxDetector(session)
    boxes = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert session.seen["images"].shape == (1, 3, 640, 640)
    assert len(boxes) == 1
    assert boxes[0].x1 == pytest.approx(75.0)
