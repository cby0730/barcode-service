# Contributing

```bash
uv sync --extra cpu            # dev tools come from the default `dev` group
uv run pre-commit install
uv run pytest                  # unit + api tests, no models needed
uv run ruff check . && uv run ruff format --check .
```

- Tests that need real models are marked `models` and run with `uv run pytest -m models`.
- Keep the service runtime free of torch and ultralytics; those belong to the offline `export` group only.
- Do not commit model weights, `.env` files or anything under `tmp/`.
- Keep changes small and typed, with comments that explain why rather than what.
- Update `uv.lock` (`uv lock`) whenever dependencies change; the Docker build uses `--frozen`.
