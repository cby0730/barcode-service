"""Pipeline behaviour with fake detector, upscaler and decoders (no models needed)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pytest

from barcode_service.pipeline import DEFAULT_STRATEGIES, Pipeline, Strategy
from barcode_service.types import Barcode, Box


class FakeDecoder:
    """Returns ``result`` when ``accept(img)`` is true; records every image it sees."""

    def __init__(self, accept: Callable[[np.ndarray], bool], result: list[Barcode] | None = None):
        self.name = "fake"
        self.accept = accept
        self.result = result if result is not None else [Barcode("ok", "QRCode", "")]
        self.seen: list[np.ndarray] = []

    def decode(self, img: np.ndarray) -> list[Barcode]:
        self.seen.append(img)
        return list(self.result) if self.accept(img) else []


class FakeDetector:
    def __init__(self, boxes: list[Box]) -> None:
        self.boxes = boxes
        self.calls = 0

    def detect(self, img_bgr: np.ndarray) -> list[Box]:
        self.calls += 1
        return self.boxes


class FakeUpscaler:
    def __init__(self, out: np.ndarray | None = "double") -> None:
        self.out = out
        self.calls = 0

    def upscale(self, img_bgr: np.ndarray) -> np.ndarray | None:
        self.calls += 1
        if isinstance(self.out, str):
            return np.repeat(np.repeat(img_bgr, 2, axis=0), 2, axis=1)
        return self.out


@pytest.fixture
def img() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (100, 200, 3), dtype=np.uint8)


def never(_: np.ndarray) -> bool:
    return False


def always(_: np.ndarray) -> bool:
    return True


def is_binary(x: np.ndarray) -> bool:
    return x.ndim == 2 and set(np.unique(x)) <= {0, 255}


def test_raw_hit_skips_every_strategy(img):
    detector = FakeDetector([Box(0, 0, 10, 10, 0.9)])
    decoder = FakeDecoder(always)
    result = Pipeline([decoder], detector).decode(img)
    assert [b.strategy for b in result] == ["raw"]
    assert len(decoder.seen) == 1
    assert detector.calls == 0


def test_first_matching_strategy_wins_and_stops(img):
    decoder = FakeDecoder(is_binary)
    strategies = [
        Strategy("clahe", ("clahe",)),
        Strategy("sauvola", ("sauvola",)),
        Strategy("never-reached", ("clahe", "sauvola")),
    ]
    result = Pipeline([decoder], strategies=strategies).decode(img)
    assert [b.strategy for b in result] == ["sauvola"]
    assert len(decoder.seen) == 3  # raw, clahe, sauvola


def test_strategy_order_comes_from_data(img):
    decoder = FakeDecoder(always)
    # Shrunk below the raw hit: make raw fail so the first listed strategy decides.
    decoder.accept = lambda x: x.ndim == 2
    first = Pipeline([decoder], strategies=[Strategy("b", ("sauvola",)), Strategy("a", ("clahe",))])
    assert first.decode(img)[0].strategy == "b"
    second = Pipeline(
        [decoder], strategies=[Strategy("a", ("clahe",)), Strategy("b", ("sauvola",))]
    )
    assert second.decode(img)[0].strategy == "a"


def test_all_fail_returns_empty_list(img):
    pipeline = Pipeline([FakeDecoder(never)], FakeDetector([]), FakeUpscaler())
    assert pipeline.decode(img) == []


def test_strategy_failure_is_logged_and_skipped(img, caplog):
    class ExplodingDetector:
        def detect(self, img_bgr):
            raise RuntimeError("inference blew up")

    decoder = FakeDecoder(lambda x: x.ndim == 2)
    strategies = [Strategy("detect", ("detect",)), Strategy("clahe", ("clahe",))]
    with caplog.at_level(logging.WARNING):
        result = Pipeline([decoder], ExplodingDetector(), strategies=strategies).decode(img)
    assert [b.strategy for b in result] == ["clahe"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "detect" in warnings[0].getMessage()


def test_detect_crops_best_box(img):
    detector = FakeDetector([Box(0, 0, 10, 10, 0.3), Box(50, 20, 150, 60, 0.9)])
    decoder = FakeDecoder(lambda x: x.shape[0] < 100)
    result = Pipeline([decoder], detector, strategies=[Strategy("detect", ("detect",))]).decode(img)
    assert result[0].strategy == "detect"
    crop = decoder.seen[-1]
    # 100x40 box plus a 5% margin on each side.
    assert crop.shape[:2] == (44, 110)


def test_detect_without_boxes_keeps_image(img):
    decoder = FakeDecoder(never)
    Pipeline([decoder], FakeDetector([]), strategies=[Strategy("detect", ("detect",))]).decode(img)
    assert decoder.seen[-1].shape == img.shape


def test_steps_skipped_when_detector_and_upscaler_missing(img):
    decoder = FakeDecoder(never)
    strategies = [Strategy("full", ("detect", "upscale"))]
    assert Pipeline([decoder], None, None, strategies).decode(img) == []
    assert decoder.seen[-1].shape == img.shape


def test_upscale_applied_and_none_result_ignored(img):
    decoder = FakeDecoder(never)
    strategies = [Strategy("up", ("upscale",))]
    Pipeline([decoder], upscaler=FakeUpscaler(), strategies=strategies).decode(img)
    assert decoder.seen[-1].shape[:2] == (200, 400)

    decoder = FakeDecoder(never)
    upscaler = FakeUpscaler(out=None)
    Pipeline([decoder], upscaler=upscaler, strategies=strategies).decode(img)
    assert upscaler.calls == 1
    assert decoder.seen[-1].shape == img.shape


def test_steps_run_in_order_on_a_copy(img):
    original = img.copy()
    decoder = FakeDecoder(never)
    strategies = [Strategy("s", ("clahe", "sauvola"))]
    Pipeline([decoder], strategies=strategies).decode(img)
    assert is_binary(decoder.seen[-1])  # sauvola last
    np.testing.assert_array_equal(img, original)


def test_resize_applied_once_before_steps(img):
    decoder = FakeDecoder(never)
    detector = FakeDetector([])
    strategies = [Strategy("d", ("detect",)), Strategy("d2", ("detect",))]
    Pipeline([decoder], detector, strategies=strategies, max_side=50).decode(img)
    assert decoder.seen[0].shape == img.shape  # raw path is untouched
    assert all(max(s.shape[:2]) == 50 for s in decoder.seen[1:])


def test_results_are_deduplicated_and_tagged(img):
    dup = [Barcode("a", "EAN13", ""), Barcode("a", "EAN13", ""), Barcode("b", "EAN13", "")]
    result = Pipeline([FakeDecoder(always, dup)]).decode(img)
    assert [(b.text, b.strategy) for b in result] == [("a", "raw"), ("b", "raw")]


def test_unknown_step_rejected_early():
    with pytest.raises(ValueError, match="bogus"):
        Pipeline([], strategies=[Strategy("x", ("bogus",))])


def test_default_strategies_are_valid_data():
    names = [s.name for s in DEFAULT_STRATEGIES]
    assert len(names) == len(set(names))
    Pipeline([], strategies=DEFAULT_STRATEGIES)
