"""Settings validation and caching."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from barcode_service.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # No stray BARCODE_* variables or .env file may leak into these tests.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BARCODE_PORT", raising=False)


def test_defaults():
    s = Settings()
    assert (s.port, s.workers, s.max_concurrency) == (8080, 1, 4)
    assert s.detector_backend == "ort-cpu"
    assert s.models_dir == Path("models")
    assert s.cors_origins == []


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("BARCODE_PORT", "9000")
    monkeypatch.setenv("BARCODE_DETECTOR_BACKEND", "openvino")
    s = Settings()
    assert s.port == 9000
    assert s.detector_backend == "openvino"


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_port_range(port):
    with pytest.raises(ValidationError):
        Settings(port=port)


def test_backend_must_be_known():
    with pytest.raises(ValidationError):
        Settings(detector_backend="tpu")
    with pytest.raises(ValidationError):
        Settings(upscaler_backend="tpu")


@pytest.mark.parametrize("field", ["max_concurrency", "workers"])
def test_at_least_one(field):
    with pytest.raises(ValidationError):
        Settings(**{field: 0})


def test_log_level_normalised_and_validated():
    assert Settings(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(log_level="loud")


def test_cors_origins_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("BARCODE_CORS_ORIGINS", "https://a.example, https://b.example")
    assert Settings().cors_origins == ["https://a.example", "https://b.example"]


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
