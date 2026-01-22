FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv sync --frozen --no-install-project --extra api

COPY src src/

RUN uv sync --frozen --extra api

ENTRYPOINT ["uv", "run", "uvicorn", "audio_emotion.api:app", "--host", "0.0.0.0", "--port", "8000"]
