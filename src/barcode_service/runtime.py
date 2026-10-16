"""Inference backends behind a minimal ``Session`` protocol."""

from __future__ import annotations

import threading
from importlib import metadata
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from barcode_service.types import BackendUnavailableError, ModelNotFoundError

Backend = Literal["ort-cpu", "ort-cuda", "openvino"]


class Session(Protocol):
    """A loaded model that maps named input tensors to output tensors."""

    input_names: list[str]

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...


def _installed(*dists: str) -> list[str]:
    found = []
    for dist in dists:
        try:
            metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
        found.append(dist)
    return found


class OrtSession:
    """onnxruntime session; ``InferenceSession.run`` is thread-safe, so no lock is needed."""

    def __init__(self, model_path: Path, device: Literal["cpu", "cuda"] = "cpu") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            extra = "gpu" if device == "cuda" else "cpu"
            raise BackendUnavailableError(
                f"onnxruntime is not installed; install the '{extra}' extra "
                f"(pip install 'barcode-service[{extra}]')."
            ) from exc

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                installed = _installed("onnxruntime-gpu", "onnxruntime") or ["no onnxruntime"]
                raise BackendUnavailableError(
                    "CUDAExecutionProvider is not available in the installed package "
                    f"({', '.join(installed)}); install onnxruntime-gpu (the 'gpu' extra, "
                    "which conflicts with 'cpu') and make sure CUDA libraries are present."
                )
            providers = ["CUDAExecutionProvider"]

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        # ORT drops an unloadable provider with only a warning; refuse to run on CPU silently.
        if self._session.get_providers()[0] != providers[0]:
            raise BackendUnavailableError(
                f"{providers[0]} was requested but onnxruntime fell back to "
                f"{self._session.get_providers()}; check the CUDA/cuDNN installation."
            )
        self.input_names = [i.name for i in self._session.get_inputs()]

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self._session.run(None, inputs)


class OpenVinoSession:
    """OpenVINO CPU session.

    A compiled model owns one infer request here, which is not safe to share between threads,
    so calls are serialised with a lock. Run several processes (workers) for parallelism.
    """

    def __init__(self, model_path: Path) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise BackendUnavailableError(
                "openvino is not installed; install the 'openvino' extra "
                "(pip install 'barcode-service[openvino]')."
            ) from exc

        core = ov.Core()
        compiled = core.compile_model(
            core.read_model(str(model_path)),
            "CPU",
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        self._request = compiled.create_infer_request()
        self._lock = threading.Lock()
        self._n_outputs = len(compiled.outputs)
        self.input_names = [i.get_any_name() for i in compiled.inputs]

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        with self._lock:
            self._request.infer(inputs)
            # Copy: the request reuses its output buffers on the next call.
            return [self._request.get_output_tensor(i).data.copy() for i in range(self._n_outputs)]


def create_session(backend: Backend, model_path: Path) -> Session:
    """Load ``model_path`` on ``backend``.

    Raises BackendUnavailableError or ModelNotFoundError with an actionable message;
    never silently falls back to another backend.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"Model file not found: {model_path}. Run scripts/download_models.py and "
            "scripts/export_models.py, or point BARCODE_MODELS_DIR at your own models."
        )
    if backend == "ort-cpu":
        return OrtSession(model_path, "cpu")
    if backend == "ort-cuda":
        return OrtSession(model_path, "cuda")
    if backend == "openvino":
        return OpenVinoSession(model_path)
    raise BackendUnavailableError(
        f"Unknown backend {backend!r}; use ort-cpu, ort-cuda or openvino."
    )
