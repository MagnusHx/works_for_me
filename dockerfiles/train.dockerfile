FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Example:
#   --group train --group cpu
#   --group train --group cu117
ARG UV_GROUPS="--group train --group cpu"

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_GROUPS}

COPY src/ src/

# IMPORTANT: keep the same groups here too
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen ${UV_GROUPS}

CMD ["uv", "run", "--frozen", "audio-emotion-train"]
