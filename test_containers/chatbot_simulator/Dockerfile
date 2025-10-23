# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

# Install build tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.12

# Create virtual environment and install dependencies
RUN uv venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install openevals==0.1.0 rich openai langchain-openai langchain-core pydantic jinja2

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Copy venv, Python installation and set path
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /python /python
ENV PATH="/app/.venv/bin:$PATH"

# Copy your app code
COPY manifest.yaml .
COPY entrypoint.py .
COPY simulation.py .
COPY evaluator.py .


RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]
