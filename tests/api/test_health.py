"""Liveness, readiness and lifespan model loading."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from barcode_service.api import app as app_module
from barcode_service.api.app import create_app
from barcode_service.config import Settings
from tests.api.conftest import EmptyPipeline


def test_health_is_always_200(make_client):
    assert make_client(None).get("/health").status_code == 200
    assert make_client(EmptyPipeline()).get("/health").status_code == 200


def test_ready_503_without_pipeline(make_client):
    assert make_client(None).get("/ready").status_code == 503


def test_ready_200_with_pipeline(make_client):
    assert make_client(EmptyPipeline()).get("/ready").status_code == 200


def test_lifespan_loads_pipeline_when_not_injected(monkeypatch):
    loaded = EmptyPipeline()
    monkeypatch.setattr(app_module, "build_pipeline", lambda settings: loaded)
    app = create_app(Settings(_env_file=None))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert app.state.pipeline is loaded


def test_injected_pipeline_never_loads_models(monkeypatch):
    def boom(settings):
        raise AssertionError("models must not be loaded")

    monkeypatch.setattr(app_module, "build_pipeline", boom)
    with TestClient(create_app(Settings(_env_file=None), pipeline=EmptyPipeline())) as client:
        assert client.get("/ready").status_code == 200


def test_model_load_failure_aborts_startup(monkeypatch):
    def boom(settings):
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(app_module, "build_pipeline", boom)
    with pytest.raises(FileNotFoundError), TestClient(create_app(Settings(_env_file=None))):
        pass


def test_build_pipeline_with_models_disabled_needs_no_files(tmp_path):
    settings = Settings(
        _env_file=None, models_dir=tmp_path, detector_enabled=False, upscaler_enabled=False
    )
    assert app_module.build_pipeline(settings) is not None


def test_build_pipeline_fails_clearly_when_model_missing(tmp_path):
    from barcode_service.types import ModelNotFoundError

    settings = Settings(_env_file=None, models_dir=tmp_path)
    with pytest.raises(ModelNotFoundError, match="download_models"):
        app_module.build_pipeline(settings)


def test_build_pipeline_wires_enabled_models(tmp_path, monkeypatch):
    calls = []

    class FakeSession:
        input_names = ["x"]

    def fake_create(backend, path):
        calls.append((backend, path.name))
        return FakeSession()

    monkeypatch.setattr(app_module, "create_session", fake_create)
    pipeline = app_module.build_pipeline(Settings(_env_file=None, models_dir=tmp_path))
    assert pipeline is not None
    assert [name for _, name in calls] == ["yolov8s-barcode.onnx", "pan-x2.onnx"]
