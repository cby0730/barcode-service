import numpy as np
import pytest

from barcode_service.preprocess import (
    apply_clahe,
    apply_sauvola,
    crop_box,
    resize_max_side,
    to_gray,
)
from barcode_service.types import Box


def test_to_gray_from_bgr(ean13_img):
    g = to_gray(ean13_img)
    assert g.shape == ean13_img.shape[:2] and g.dtype == np.uint8


def test_to_gray_is_idempotent(ean13_img):
    g = to_gray(ean13_img)
    assert np.array_equal(to_gray(g), g)


def test_clahe_returns_gray_uint8(ean13_img):
    out = apply_clahe(ean13_img)
    assert out.ndim == 2 and out.dtype == np.uint8 and out.shape == ean13_img.shape[:2]


def test_clahe_accepts_gray(ean13_img):
    assert apply_clahe(to_gray(ean13_img)).ndim == 2


def test_sauvola_is_binary_uint8(ean13_img):
    out = apply_sauvola(ean13_img)
    assert out.dtype == np.uint8 and out.shape == ean13_img.shape[:2]
    assert set(np.unique(out)) <= {0, 255}
    assert len(np.unique(out)) == 2  # a barcode has both ink and paper


def test_sauvola_blank_image_does_not_crash():
    out = apply_sauvola(np.full((40, 40), 255, np.uint8))
    assert set(np.unique(out)) <= {0, 255}


def test_resize_shrinks_to_max_side():
    img = np.zeros((400, 1000, 3), np.uint8)
    out = resize_max_side(img, 500)
    assert out.shape == (200, 500, 3) and out.dtype == np.uint8


def test_resize_never_upsamples():
    img = np.zeros((50, 60, 3), np.uint8)
    assert resize_max_side(img, 800).shape == img.shape


def test_resize_gray_keeps_ndim():
    assert resize_max_side(np.zeros((300, 600), np.uint8), 100).shape == (50, 100)


def test_crop_box_with_margin():
    img = np.arange(100 * 100, dtype=np.uint32).reshape(100, 100).astype(np.uint8)
    out = crop_box(img, Box(20, 20, 40, 60, 0.9), margin=0.1)
    # width 20 -> 2px margin each side, height 40 -> 4px
    assert out.shape == (48, 24)


def test_crop_box_clips_to_image():
    img = np.zeros((50, 80, 3), np.uint8)
    out = crop_box(img, Box(-10, -10, 200, 200, 0.5), margin=0.2)
    assert out.shape == img.shape


def test_crop_box_outside_image_gives_empty_crop():
    out = crop_box(np.zeros((50, 50, 3), np.uint8), Box(100, 100, 120, 120, 0.5))
    assert out.size == 0


@pytest.mark.parametrize("fn", [to_gray, apply_clahe, apply_sauvola])
def test_inputs_are_not_mutated(fn, ean13_img):
    before = ean13_img.copy()
    fn(ean13_img)
    assert np.array_equal(ean13_img, before)
