# barcode-service

A barcode decoding service. An ONNX YOLO detector finds barcode regions, an optional ONNX x2
super-resolution model sharpens small crops, and several decoders (pyzbar, zxing-cpp) are tried
over a list of image-processing strategies until one succeeds. Everything is served through a
small FastAPI app.

## Features

- Runtime depends on onnxruntime (or OpenVINO) only: no torch and no ultralytics in the service image.
- Three inference backends, selectable independently for the detector and the upscaler:
  `ort-cpu`, `openvino`, `ort-cuda`. An unavailable backend fails at startup; there is no silent fallback.
- JSON (base64 / data URI) and multipart upload.
- Responses carry `text`, `format` and the `strategy` that produced the result.
- Request size and pixel limits, a concurrency limit that sheds load with `503` + `Retry-After`.
- CORS is off unless origins are listed explicitly.

## Architecture

```
            +-------------------------- barcode-service ---------------------------+
 client --> | FastAPI  POST /v1/decode                                              |
 (JSON /    |   |  size + pixel limits, concurrency slots                           |
 multipart) |   v                                                                   |
            | imaging (bytes/base64 -> BGR ndarray)                                 |
            |   |                                                                   |
            |   v                                                                   |
            | pipeline: for each strategy (data) until a decoder succeeds           |
            |     preprocess (grayscale / clahe / sauvola / resize)                 |
            |       -> Detector (YOLO ONNX)  -> crop -> Upscaler (PAN x2 ONNX)      |
            |       -> Decoders (pyzbar, zxing-cpp) -> Barcode(text, format, ...)   |
            |                                                                       |
            | runtime: Session = onnxruntime CPU | OpenVINO | onnxruntime CUDA     |
            +-----------------------------------------------------------------------+
                                        ^
                                  models/ (ONNX files, mounted, not bundled)
```

## Quickstart

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). Pick exactly one of the `cpu` and
`gpu` extras (they conflict); `openvino` can be added on top.

```bash
uv sync --extra cpu --extra openvino     # CPU + OpenVINO
# uv sync --extra gpu                    # CUDA (onnxruntime-gpu)
```

### Models

Weights are not included. Choose one:

1. Download and export the recommended weights (the export step needs torch/ultralytics, kept in
   separate dependency groups so they never reach the service image):

   ```bash
   uv run --group models python scripts/download_models.py
   uv run --group export python scripts/export_models.py \
       --yolo-weights models/pretrained/yolov8s-barcode.pt \
       --pan-dir models/pretrained/pan
   ```

   This produces `models/yolov8s-barcode.onnx` and `models/pan-x2.onnx`.
2. Bring your own compatible ONNX files and point `BARCODE_MODELS_DIR`, `BARCODE_DETECTOR_MODEL` and `BARCODE_UPSCALER_MODEL` at them.

### Run

```bash
uv run barcode-service serve                                   # defaults: 0.0.0.0:8080
uv run barcode-service serve --port 9000 --detector-backend openvino
BARCODE_PORT=9000 uv run barcode-service serve --port 8081      # listens on 8081
```

Precedence is CLI flags > environment variables > `.env` > defaults. `host`, `port`, `workers`,
`log-level`, `detector-backend`, `upscaler-backend` and `models-dir` can be given on the command
line; the other settings are environment-only. The CLI writes the effective values back to the
environment before starting uvicorn, so `--workers N` child processes see the same configuration.

### Docker

```bash
cp .env.example .env      # optional
docker compose up --build                                          # CPU image (default)
docker compose -f compose.yaml -f compose.gpu.yaml up --build      # GPU image
```

- `./models` is mounted read-only at `/app/models`; the root filesystem is read-only with a tmpfs `/tmp`.
- The container always listens on 8080. Change the published host port with `BARCODE_HOST_PORT`
  (for example `BARCODE_HOST_PORT=9000 docker compose up`).
- The GPU file needs the NVIDIA Container Toolkit on the host and a backend set to `ort-cuda`
  (`BARCODE_DETECTOR_BACKEND=ort-cuda`, `BARCODE_UPSCALER_BACKEND=ort-cuda`, for example in `.env`).
- Behind a proxy, export `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` in your shell; compose forwards
  them as build args. Nothing is hardcoded.
- Build targets can also be used directly: `docker build --target cpu .` / `--target gpu .`

## API

### `POST /v1/decode`

JSON body with a base64 string or data URI:

```bash
curl -s http://localhost:8080/v1/decode \
  -H 'Content-Type: application/json' \
  -d "{\"image\": \"$(base64 -w0 sample.png)\"}"
```

Multipart upload (raw bytes, about 33% smaller than base64):

```bash
curl -s http://localhost:8080/v1/decode -F file=@sample.png
```

Response:

```json
{"barcodes": [{"text": "4006381333931", "format": "EAN13", "strategy": "raw"}]}
```

An image with no readable barcode returns `{"barcodes": []}` with status 200.

| Status | Meaning |
|---|---|
| 400 | not a decodable image or malformed body |
| 413 | body or decoded image exceeds `BARCODE_MAX_IMAGE_BYTES` / `BARCODE_MAX_IMAGE_PIXELS` |
| 415 | content type is neither `application/json` nor `multipart/form-data` |
| 422 | JSON lacks a string `image`, or multipart lacks a `file` field |
| 503 | models not loaded, or at capacity (`Retry-After` is set in the latter case) |

### `GET /health` and `GET /ready`

`/health` returns `{"status": "ok"}` whenever the process is up (liveness). `/ready` returns
`{"status": "ready"}` once the models are loaded and `503` otherwise.

## Backends

Backends are chosen per model with `BARCODE_DETECTOR_BACKEND` and `BARCODE_UPSCALER_BACKEND`.

| Backend | Runs on | Install |
|---|---|---|
| `ort-cpu` | onnxruntime, CPU | `uv sync --extra cpu` |
| `openvino` | OpenVINO, CPU (reads the ONNX file directly) | `uv sync --extra openvino` (plus `--extra cpu` if you also use `ort-cpu`) |
| `ort-cuda` | onnxruntime, NVIDIA GPU | `uv sync --extra gpu` (conflicts with `cpu`); the `gpu` Docker target ships the CUDA libraries |

Docker target `cpu` installs `cpu` + `openvino`; target `gpu` installs `gpu`.

## Configuration

All settings are environment variables with the prefix `BARCODE_` (see [.env.example](.env.example)).

| Variable | Default | Description |
|---|---|---|
| `BARCODE_HOST` | `0.0.0.0` | bind address (CLI `--host`) |
| `BARCODE_PORT` | `8080` | listen port (CLI `--port`) |
| `BARCODE_WORKERS` | `1` | worker processes, each loads its own models (CLI `--workers`) |
| `BARCODE_LOG_LEVEL` | `INFO` | log level (CLI `--log-level`) |
| `BARCODE_MODELS_DIR` | `models` | directory with the ONNX files (CLI `--models-dir`) |
| `BARCODE_DETECTOR_MODEL` | `yolov8s-barcode.onnx` | detector file name inside the models dir |
| `BARCODE_UPSCALER_MODEL` | `pan-x2.onnx` | upscaler file name inside the models dir |
| `BARCODE_DETECTOR_BACKEND` | `ort-cpu` | `ort-cpu`, `openvino` or `ort-cuda` (CLI `--detector-backend`) |
| `BARCODE_UPSCALER_BACKEND` | `ort-cpu` | same choices (CLI `--upscaler-backend`) |
| `BARCODE_DETECTOR_ENABLED` | `true` | set `false` to skip the detector |
| `BARCODE_UPSCALER_ENABLED` | `true` | set `false` to skip the upscaler |
| `BARCODE_MAX_IMAGE_BYTES` | `10485760` | maximum upload size in bytes |
| `BARCODE_MAX_IMAGE_PIXELS` | `40000000` | maximum decoded width x height |
| `BARCODE_MAX_CONCURRENCY` | `4` | concurrent decodes per worker; more get `503` |
| `BARCODE_CORS_ORIGINS` | `[]` | JSON list of allowed origins; CORS is off when empty |

## Testing

| Layer | Command | Needs models | Runs in CI |
|---|---|---|---|
| unit (`tests/unit`) | `uv run pytest` | no | yes |
| api (`tests/api`) | `uv run pytest` | no (fake pipeline, includes concurrency checks) | yes |
| integration | `uv run pytest -m models` | yes, skipped when absent | no |
| gpu | `uv run pytest -m gpu` | CUDA GPU, skipped when absent | no |
| e2e / load | `scripts/e2e_smoke.py`, `scripts/loadtest.py` against a running service | yes | no |

Lint: `uv run ruff check . && uv run ruff format --check .`

## Benchmarks

The strategy list in the pipeline is data, and `scripts/benchmark.py` measures hit rate and
latency per strategy. Treat the numbers with care: the dataset is synthetic (generated with
zxing-cpp and then degraded), so it only shows relative differences between strategies and
backends, not hit rates on real photographs.

Measured on a 224-CPU Xeon with an H100 (shared machine), 280 synthetic images:

| backend | hit | mean ms | p95 ms | detector ms/call | upscaler ms/call |
|---|---|---|---|---|---|
| `ort-cpu` | 95.4% | 58.4 | 287.3 | 47.4 | 107.8 |
| `openvino` | 95.4% | 17.3 | 45.2 | 7.6 | 17.3 |
| `ort-cuda` | 95.4% | 15.2 | 45.0 | 4.0 | 8.9 |

- The untouched image decodes 90.4% of the set; the fallback strategies add 5 points, almost all
  from `detect+upscale` on downscaled images. `detect`, `detect+sauvola` and
  `detect+upscale+clahe` added nothing and were dropped from `DEFAULT_STRATEGIES`.
- OpenVINO is about 6x faster than onnxruntime CPU and is the better default without a GPU.
  CUDA is faster per call, but paid about 1.3 s the first time it saw each new input shape in
  the upscaler, which shows up in the tail latency with variable-size inputs.
- Synthetic data cannot show real-photo hit rates (blur, glare, perspective); ten images per
  cell also means a one-image difference is below noise.

## Load test

`scripts/loadtest.py` against a local service (OpenVINO, loopback, client on the same machine):

- A clean image (raw pass) reaches about 550 RPS at concurrency 16 with p95 of about 80 ms; an image
  that needs the detector and upscaler reaches about 45 RPS per process with p50 of about 85 ms
  at concurrency 4.
- Adding workers helped only modestly in this environment (about +25% for 2 workers on the heavy
  path, nothing consistent for 4), probably because each process already uses all CPUs. This
  was not tuned.
- Overload (`BARCODE_MAX_CONCURRENCY=2`, 64 concurrent clients, 10 s): 2483 of 2878 requests were
  rejected with 503 and `Retry-After`, 395 succeeded, with no timeouts, 500s or crashes.
- `scripts/e2e_smoke.py <url>` checks health, readiness, JSON and multipart decoding and the
  400 / 413 / 422 paths against a running service (13 checks passed).

## Licensing

- This project is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
- Model weights are not included in this repository and are git-ignored. You download or supply them yourself.
- The recommended detector weights, `Piero2411/YOLOV8s-Barcode-Detection`, are published under AGPL-3.0.
  An ONNX file exported from them is derived from those weights.
- The recommended upscaler weights, `eugenesiow/pan`, are published under Apache-2.0.
- The service does not import ultralytics at runtime; ultralytics and torch are only used by the
  offline export script (the `export` dependency group).
- No legal conclusions are drawn here. Check each model's license before use or redistribution.
