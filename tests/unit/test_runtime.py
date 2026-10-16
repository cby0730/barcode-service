"""Backend selection and error reporting; no model files required."""

from __future__ import annotations

import sys

import pytest

from barcode_service import runtime
from barcode_service.runtime import create_session
from barcode_service.types import BackendUnavailableError, ModelNotFoundError


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"not a real model")  # existence is checked before parsing
    return path


@pytest.mark.parametrize("backend", ["ort-cpu", "ort-cuda", "openvino"])
def test_missing_model_file_points_to_download_script(backend, tmp_path):
    with pytest.raises(ModelNotFoundError, match=r"scripts/download_models\.py"):
        create_session(backend, tmp_path / "absent.onnx")


def test_unknown_backend_is_rejected(model_file):
    with pytest.raises(BackendUnavailableError, match="Unknown backend"):
        create_session("tensorrt", model_file)  # type: ignore[arg-type]


def test_missing_openvino_package_names_the_extra(model_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino", None)  # makes `import openvino` fail
    with pytest.raises(BackendUnavailableError, match=r"\[openvino\]"):
        create_session("openvino", model_file)


@pytest.mark.parametrize(("backend", "extra"), [("ort-cpu", "cpu"), ("ort-cuda", "gpu")])
def test_missing_onnxruntime_package_names_the_extra(backend, extra, model_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(BackendUnavailableError, match=rf"\[{extra}\]"):
        create_session(backend, model_file)


def test_cuda_requested_without_provider_raises_instead_of_falling_back(model_file, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(BackendUnavailableError, match="onnxruntime-gpu"):
        create_session("ort-cuda", model_file)


def test_module_imports_without_inference_packages():
    # Lazy imports: the module-level namespace must not hold either package.
    assert not hasattr(runtime, "ort")
    assert not hasattr(runtime, "ov")


def test_cuda_provider_silently_dropped_by_ort_is_an_error(model_file, monkeypatch):
    ort = pytest.importorskip("onnxruntime")

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    with pytest.raises(BackendUnavailableError, match="fell back"):
        create_session("ort-cuda", model_file)
