# Stage 1: Builder - Install dependencies and create virtual environment
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

# Configure uv for faster builds and smaller images
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.11

# Create virtual environment
RUN uv venv

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install litellm httpx

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application files
COPY entrypoint.py /app/entrypoint.py
COPY manifest.yaml /app/manifest.yaml

RUN chmod +x /app/entrypoint.py

# Entrypoint for the container
ENTRYPOINT ["python", "/app/entrypoint.py"]