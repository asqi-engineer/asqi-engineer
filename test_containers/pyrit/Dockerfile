# Stage 1: Build dependencies
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

# Install build tools and system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        unixodbc-dev && \
    rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python 3.12
RUN uv python install 3.12

# Create virtual environment
RUN uv venv

# Install PyRIT and dependencies
# We install the core pyrit package without heavy optional dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install pyrit==0.10.0rc0 ipython

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        unixodbc \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy venv, Python installation and set path
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /python /python
ENV PATH="/app/.venv/bin:$PATH"

# Copy entrypoint and manifest
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
