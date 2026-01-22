# syntax=docker/dockerfile:1.7

# Pin the base image by digest for reproducibility.
# (Digest taken from your earlier build logs.)
FROM python:3.11-slim-bookworm@sha256:bcbbec29f7a3f9cbee891e3cd69d7fe4dec7e281daf36cbd415ddd8ee2ba0077

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_MANAGED_PYTHON=1 \
    INSTALLER_NO_MODIFY_PATH=1 \
    PATH="/app/.venv/bin:/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pin uv version (change this only when you intentionally upgrade)
ARG UV_VERSION=0.9.26
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

WORKDIR /app

# The train group includes torch with platform markers
ARG UV_GROUPS="--group train"

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_GROUPS}

COPY configs configs/
COPY src src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen ${UV_GROUPS}

CMD ["uv", "run", "python", "src/audio_emotion/train.py"]
