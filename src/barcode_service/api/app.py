"""Application factory."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from anyio import to_thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from barcode_service import __version__
from barcode_service.api.errors import register_error_handlers
from barcode_service.api.routes import router
from barcode_service.config import Settings, get_settings
from barcode_service.decoders import PyzbarDecoder, ZxingDecoder
from barcode_service.detector import OnnxDetector
from barcode_service.pipeline import Pipeline
from barcode_service.runtime import create_session
from barcode_service.upscaler import OnnxUpscaler


def build_pipeline(settings: Settings) -> Pipeline:
    """Load the enabled models and assemble a pipeline; raises if a model or backend is missing."""
    detector = upscaler = None
    if settings.detector_enabled:
        path = settings.models_dir / settings.detector_model
        detector = OnnxDetector(create_session(settings.detector_backend, path))
    if settings.upscaler_enabled:
        path = settings.models_dir / settings.upscaler_model
        upscaler = OnnxUpscaler(create_session(settings.upscaler_backend, path))
    return Pipeline([ZxingDecoder(), PyzbarDecoder()], detector, upscaler)


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    """Build the app. An injected ``pipeline`` skips model loading; it is kept on ``app.state``."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load in a worker thread so startup does not block the event loop; a failure here
        # aborts startup with the loader's message instead of serving 503s forever.
        if app.state.pipeline is None:
            app.state.pipeline = await to_thread.run_sync(build_pipeline, settings)
        yield

    app = FastAPI(title="barcode-service", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.slots = threading.BoundedSemaphore(settings.max_concurrency)

    if settings.cors_origins:
        # Credentials stay off so an explicit origin list can never be combined with a wildcard.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            allow_credentials=False,
        )
    register_error_handlers(app)
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    """Zero-argument factory for uvicorn; workers rebuild settings from the environment."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    return create_app(settings)
