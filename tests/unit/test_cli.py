"""CLI precedence (CLI > env > .env > defaults) and the environment write-back."""

from __future__ import annotations

import os

import pytest

from barcode_service import __version__, cli

_FIELDS = (
    "HOST",
    "PORT",
    "WORKERS",
    "LOG_LEVEL",
    "DETECTOR_BACKEND",
    "UPSCALER_BACKEND",
    "MODELS_DIR",
)


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run ``main`` with uvicorn stubbed out; returns the kwargs uvicorn would get."""
    monkeypatch.chdir(tmp_path)  # isolates the .env lookup
    for name in _FIELDS:
        # setenv then delenv makes monkeypatch restore the variable after main() writes it back.
        monkeypatch.setenv(f"BARCODE_{name}", "x")
        monkeypatch.delenv(f"BARCODE_{name}")
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

    def _run(*argv: str) -> dict:
        cli.main(["serve", *argv])
        assert len(calls) == 1
        return calls[0][1]

    return _run


def test_defaults(run):
    kwargs = run()
    assert kwargs["port"] == 8080
    assert kwargs["workers"] == 1


def test_cli_beats_env_beats_default(run, monkeypatch):
    monkeypatch.setenv("BARCODE_PORT", "9000")
    monkeypatch.setenv("BARCODE_WORKERS", "3")
    kwargs = run("--port", "9100")
    assert kwargs["port"] == 9100  # CLI over env
    assert kwargs["workers"] == 3  # env over default
    assert kwargs["host"] == "0.0.0.0"  # default


def test_env_beats_dotenv(run, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("BARCODE_PORT=9001\nBARCODE_WORKERS=2\n")
    monkeypatch.setenv("BARCODE_PORT", "9002")
    kwargs = run()
    assert kwargs["port"] == 9002
    assert kwargs["workers"] == 2


def test_effective_values_written_back_to_environment(run, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("BARCODE_WORKERS=2\n")
    monkeypatch.setenv("BARCODE_PORT", "9000")
    run(
        "--port",
        "9100",
        "--detector-backend",
        "openvino",
        "--models-dir",
        "m",
        "--log-level",
        "debug",
    )
    assert os.environ["BARCODE_PORT"] == "9100"
    assert os.environ["BARCODE_WORKERS"] == "2"  # came from .env, workers must still see it
    assert os.environ["BARCODE_DETECTOR_BACKEND"] == "openvino"
    assert os.environ["BARCODE_UPSCALER_BACKEND"] == "ort-cpu"
    assert os.environ["BARCODE_MODELS_DIR"] == "m"
    assert os.environ["BARCODE_LOG_LEVEL"] == "DEBUG"


def test_runs_factory_import_string(run):
    kwargs = run()
    assert kwargs["factory"] is True
    assert kwargs["workers"] == 1


def test_factory_target_is_importable():
    module, _, attr = cli._FACTORY.partition(":")
    assert callable(getattr(__import__(module, fromlist=[attr]), attr))


def test_invalid_value_exits_with_usage_error(run):
    with pytest.raises(SystemExit) as exc:
        run("--port", "70000")
    assert exc.value.code == 2


def test_unknown_backend_rejected(run):
    with pytest.raises(SystemExit):
        run("--detector-backend", "tpu")


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
