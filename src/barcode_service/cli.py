"""Command line entry point: ``barcode-service serve``."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import get_args

import uvicorn
from pydantic import ValidationError

from barcode_service import __version__
from barcode_service.config import Settings
from barcode_service.runtime import Backend

# Settings that can be given on the command line; the rest are env-only.
_CLI_FIELDS = (
    "host",
    "port",
    "workers",
    "log_level",
    "detector_backend",
    "upscaler_backend",
    "models_dir",
)
_FACTORY = "barcode_service.api.app:create_app_from_env"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="barcode-service")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--workers", type=int)
    serve.add_argument("--log-level")
    serve.add_argument("--detector-backend", choices=get_args(Backend))
    serve.add_argument("--upscaler-backend", choices=get_args(Backend))
    serve.add_argument("--models-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments (CLI overrides env), export them to the environment and run uvicorn."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Unset options stay out of the kwargs so env / .env / defaults still apply to them.
    overrides = {k: v for k in _CLI_FIELDS if (v := getattr(args, k)) is not None}
    try:
        settings = Settings(**overrides)
    except ValidationError as exc:
        parser.error(str(exc))

    # Worker subprocesses build their own Settings from the environment, so the effective
    # values must be visible there even when they came from the command line or .env.
    for field in _CLI_FIELDS:
        os.environ[f"BARCODE_{field.upper()}"] = str(getattr(settings, field))

    uvicorn.run(
        _FACTORY,
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )
