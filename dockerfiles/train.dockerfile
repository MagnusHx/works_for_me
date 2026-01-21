# dockerfiles/train.dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# System deps (libsndfile er vigtig for soundfile/librosa; ffmpeg er handy til flere lydformater)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Vælg extras ved build (fx "train,cpu" eller "train,cu117")
ARG UV_EXTRAS="train,cpu"

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    sh -c 'set -eu; \
      EXTRAS_FLAGS=""; \
      IFS=,; for e in $UV_EXTRAS; do EXTRAS_FLAGS="$EXTRAS_FLAGS --extra $e"; done; \
      uv sync --frozen --no-install-project $EXTRAS_FLAGS'

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    sh -c 'set -eu; \
      EXTRAS_FLAGS=""; \
      IFS=,; for e in $UV_EXTRAS; do EXTRAS_FLAGS="$EXTRAS_FLAGS --extra $e"; done; \
      uv sync --frozen $EXTRAS_FLAGS'

# Default (compose overskriver typisk command alligevel)
CMD ["uv", "run", "audio-emotion-train"]
