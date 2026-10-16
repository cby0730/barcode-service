"""Fixtures for API tests: fake pipelines and tiny real PNG payloads."""

from __future__ import annotations

import base64
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from barcode_service.api.app import create_app
from barcode_service.config import Settings
from barcode_service.types import Barcode


def png_bytes(value: int = 0, size: int = 8) -> bytes:
    """A lossless size x size PNG whose pixels all equal ``value``."""
    ok, buf = cv2.imencode(".png", np.full((size, size, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def noisy_png(size: int) -> bytes:
    """An incompressible PNG, large in bytes."""
    rng = np.random.default_rng(0)
    ok, buf = cv2.imencode(".png", rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class EchoPipeline:
    """Fake pipeline that echoes the image's pixel value, so responses identify their request."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def decode(self, img: np.ndarray) -> list[Barcode]:
        if self.delay:
            time.sleep(self.delay)
        return [Barcode(text=str(int(img[0, 0, 0])), format="QRCode", strategy="raw")]


class EmptyPipeline:
    def decode(self, img: np.ndarray) -> list[Barcode]:
        return []


class BrokenPipeline:
    def decode(self, img: np.ndarray) -> list[Barcode]:
        raise RuntimeError("secret internal detail")


class BlockingPipeline:
    """Holds every call until ``release`` is set; ``entered`` fires once a call is inside."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def decode(self, img: np.ndarray) -> list[Barcode]:
        self.entered.set()
        assert self.release.wait(timeout=10)
        return []


@pytest.fixture
def make_client():
    def _make(pipeline=None, **settings) -> TestClient:
        app = create_app(Settings(_env_file=None, **settings), pipeline=pipeline)
        return TestClient(app, raise_server_exceptions=False)

    return _make
