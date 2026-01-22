FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 🔑 INSTALL PYTHON 3.11 VIA UV
RUN uv python install 3.11

WORKDIR /app

# The train group includes torch with platform markers that automatically
# select CPU (macOS/Darwin) or CUDA (Linux/Windows) versions
ARG UV_GROUPS="--group train"

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_GROUPS}

COPY configs configs/
COPY src src/

# IMPORTANT: keep the same groups here too
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen ${UV_GROUPS}

CMD ["uv", "run", "python", "src/audio_emotion/train.py"]
