# dockerfiles/train.dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# fx: "train,cpu" eller "train,cu117"
ARG UV_EXTRAS="train,cpu"

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    sh -c 'set -eu; \
      EXTRA_ARGS=""; \
      OLDIFS="$IFS"; IFS=,; \
      for e in $UV_EXTRAS; do EXTRA_ARGS="$EXTRA_ARGS --extra $e"; done; \
      IFS="$OLDIFS"; \
      uv sync --frozen --no-install-project $EXTRA_ARGS'

COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    sh -c 'set -eu; \
      EXTRA_ARGS=""; \
      OLDIFS="$IFS"; IFS=,; \
      for e in $UV_EXTRAS; do EXTRA_ARGS="$EXTRA_ARGS --extra $e"; done; \
      IFS="$OLDIFS"; \
      uv sync --frozen $EXTRA_ARGS'

CMD ["uv", "run", "audio-emotion-train"]
