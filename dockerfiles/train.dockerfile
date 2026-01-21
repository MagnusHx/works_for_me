# dockerfiles/train.dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# System dependencies (audio + build basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      libsndfile1 \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Build-time selection of dependency groups
# Example values:
#   "--group train --group cpu"
#   "--group train --group cu117"
ARG UV_GROUPS=""

# Copy dependency manifests
COPY pyproject.toml uv.lock README.md LICENSE ./

# Install dependencies (without installing the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${UV_GROUPS}

# Copy source code
COPY src/ src/

# Install the project (entry points, scripts, etc.)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Default command (can be overridden by docker-compose)
CMD ["uv", "run", "audio-emotion-train"]
