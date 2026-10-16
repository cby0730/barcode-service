"""Upscaler pre/post-processing with fake sessions (no model needed)."""

from __future__ import annotations

import logging

import numpy as np

from barcode_service.upscaler import OnnxUpscaler


class NearestSession:
    """Stand-in x2 model: nearest-neighbour upsampling of the NCHW input."""

    input_names = ["input"]

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def run(self, inputs):
        x = inputs["input"]
        self.seen = x
        return [x.repeat(2, axis=2).repeat(2, axis=3)]


class BrokenSession:
    input_names = ["input"]

    def run(self, inputs):
        raise RuntimeError("boom")


def test_output_is_twice_the_size_and_round_trips_pixels():
    img = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    session = NearestSession()
    out = OnnxUpscaler(session).upscale(img)
    assert out is not None
    assert out.shape == (10, 14, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[::2, ::2], img)  # BGR order and scaling survive
    assert session.seen is not None
    assert session.seen.shape == (1, 3, 5, 7)
    assert session.seen.dtype == np.float32
    assert session.seen.min() >= 0.0 and session.seen.max() <= 1.0


def test_input_is_fed_as_rgb():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue
    session = NearestSession()
    OnnxUpscaler(session).upscale(img)
    assert session.seen is not None
    assert session.seen[0, 2].min() == 1.0


def test_values_outside_unit_range_are_clipped():
    class Overshoot(NearestSession):
        def run(self, inputs):
            return [super().run(inputs)[0] * 2.0 - 0.5]

    img = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = OnnxUpscaler(Overshoot()).upscale(img)
    assert out is not None
    assert out.max() == 255 and out.min() == 255


def test_returns_none_and_logs_when_the_model_fails(caplog):
    with caplog.at_level(logging.WARNING):
        assert OnnxUpscaler(BrokenSession()).upscale(np.zeros((4, 4, 3), np.uint8)) is None
    assert "Upscaling failed" in caplog.text


def test_returns_none_for_grayscale_input():
    assert OnnxUpscaler(NearestSession()).upscale(np.zeros((4, 4), np.uint8)) is None


def test_returns_none_when_output_shape_is_wrong():
    class WrongScale(NearestSession):
        def run(self, inputs):
            return [inputs["input"].repeat(3, axis=2).repeat(3, axis=3)]

    assert OnnxUpscaler(WrongScale()).upscale(np.zeros((4, 4, 3), np.uint8)) is None
