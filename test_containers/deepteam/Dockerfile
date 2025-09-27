# Stage 1: Build dependencies
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder
WORKDIR /app

# Install build tools and Rust
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.12

# Create virtual environment
RUN uv venv

# Install Rust and Python packages in one step
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=cache,target=/root/.cargo,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rustup default stable && \
    uv pip install deepteam==0.2.4 deepeval==3.3.9 litellm==1.76.0

# Stage 2: Runtime (Python 3.12 slim)
FROM python:3.12-slim
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
COPY manifest.yaml .
COPY entrypoint.py .

ENV DEEPTEAM_TELEMETRY_OPT_OUT="YES"
RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]
