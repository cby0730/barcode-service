"""Environment-driven settings (prefix ``BARCODE_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from barcode_service.runtime import Backend

_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


class Settings(BaseSettings):
    """Service configuration; precedence is init kwargs > env > .env > defaults."""

    model_config = SettingsConfigDict(env_prefix="BARCODE_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535)
    workers: int = Field(1, ge=1)
    log_level: str = "INFO"

    models_dir: Path = Path("models")
    detector_model: str = "yolov8s-barcode.onnx"
    upscaler_model: str = "pan-x2.onnx"
    detector_backend: Backend = "ort-cpu"
    upscaler_backend: Backend = "ort-cpu"
    detector_enabled: bool = True
    upscaler_enabled: bool = True

    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_concurrency: int = Field(4, ge=1)
    # NoDecode: accept a plain comma-separated env value instead of requiring JSON.
    cors_origins: Annotated[list[str], NoDecode] = []

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"must be one of {', '.join(_LOG_LEVELS)}")
        return level

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
