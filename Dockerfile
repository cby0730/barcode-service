# syntax=docker/dockerfile:1
#
# Targets:
#   cpu  onnxruntime (CPU) + OpenVINO on a slim Debian base. This is the default target.
#   gpu  onnxruntime-gpu on an NVIDIA CUDA runtime base.
#
#   docker build --target cpu -t barcode-service:cpu .
#   docker build --target gpu -t barcode-service:gpu .
#
# Behind a proxy, pass --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=... --build-arg
# NO_PROXY=...; these are Docker's predefined proxy args, so they are unset (and absent from the
# final image) unless you provide them. Models are not baked in: mount them at /app/models.

# uv is installed from PyPI inside the builder stages only (it never reaches the final images).
ARG UV_VERSION=0.12.18
ARG CPU_BASE=python:3.12.12-slim-bookworm
ARG GPU_BASE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# ---------------------------------------------------------------- gpu
FROM ${GPU_BASE} AS builder-gpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG UV_VERSION
# A throwaway venv keeps pip away from the system interpreter (PEP 668 on Ubuntu 24.04).
RUN python3.12 -m venv /opt/uv-venv \
    && /opt/uv-venv/bin/pip install --no-cache-dir "uv==${UV_VERSION}" \
    && ln -s /opt/uv-venv/bin/uv /usr/local/bin/uv
# The venv must point at the same interpreter path the runtime image has.
ENV UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first so this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --frozen --no-dev --no-install-project --extra gpu
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra gpu

FROM ${GPU_BASE} AS gpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 libzbar0 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder-gpu --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    BARCODE_MODELS_DIR=/app/models BARCODE_PORT=8080
WORKDIR /app
USER app
EXPOSE 8080
# python instead of curl keeps the image free of extra tools.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['BARCODE_PORT'], timeout=3)"]
ENTRYPOINT ["barcode-service", "serve"]

# ---------------------------------------------------------------- cpu
FROM ${CPU_BASE} AS builder-cpu
ARG UV_VERSION
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --frozen --no-dev --no-install-project --extra cpu --extra openvino
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra cpu --extra openvino

# Last stage, so a plain `docker build .` produces the cpu image.
FROM ${CPU_BASE} AS cpu
# libzbar0: pyzbar loads it at runtime. libglib2.0-0 / libgomp1: cheap insurance for the OpenCV
# and OpenMP-linked wheels on a slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzbar0 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder-cpu --chown=app:app /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    BARCODE_MODELS_DIR=/app/models BARCODE_PORT=8080
WORKDIR /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['BARCODE_PORT'], timeout=3)"]
ENTRYPOINT ["barcode-service", "serve"]
