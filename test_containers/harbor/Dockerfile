# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.13

# Create virtual environment
RUN uv venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install harbor

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Install Docker with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/tmp/docker-cache \
    apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Copy venv and Python installation from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /python /python
ENV PATH="/app/.venv/bin:$PATH"

COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]