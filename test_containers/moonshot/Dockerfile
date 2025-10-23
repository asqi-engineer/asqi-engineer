# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /tmp

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.12

# Create virtual environment
RUN uv venv

# Install Moonshot and data
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "aiverify-moonshot[all]" openai && \
    git clone https://github.com/aiverify-foundation/moonshot-data.git && \
    cd moonshot-data && \
    uv pip install -r requirements.txt

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Copy venv and Python installation
COPY --from=builder /tmp/.venv /app/.venv
COPY --from=builder /python /python
COPY --from=builder /tmp/moonshot-data /app/moonshot-data
ENV PATH="/app/.venv/bin:$PATH"

COPY manifest.yaml entrypoint.py ./

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
