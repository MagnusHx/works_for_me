FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Install uv
RUN apt-get update && apt-get install -y curl \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY configs configs/
COPY src src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["uv", "run", "python", "src/audio_emotion/train.py"]
