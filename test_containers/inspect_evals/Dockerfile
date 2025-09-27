# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

# Install system dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Configure uv for faster builds
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python
RUN uv python install 3.11

# Set environment variable to allow remote code execution for datasets
ENV HF_DATASETS_TRUST_REMOTE_CODE=1

# Clone inspect_evals 
RUN git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd inspect_evals && \
    git checkout 0d37467

# Create virtual environment
RUN cd inspect_evals && uv venv

RUN --mount=type=cache,target=/root/.cache/uv \
    cd inspect_evals && \
    uv sync --extra test --extra math --extra mle_bench --extra worldsense --extra scicode --extra ifeval --extra personality --extra stealth --extra bbq && \
    uv add tf-keras && \
    uv add google-genai

# Stage 2: Runtime
FROM python:3.11-slim

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

# Copy installed packages and venv from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/inspect_evals /app/inspect_evals

ENV PATH="/app/inspect_evals/.venv/bin:/root/.local/bin:$PATH"

# Set environment variables
ENV HF_DATASETS_TRUST_REMOTE_CODE=1
ENV CYBENCH_ACKNOWLEDGE_RISKS=1

# Copy test container files
COPY manifest.yaml .
COPY entrypoint.py .

# TODO: Find a better solution for this
# Create symlink for evaluator_prompt.yaml to fix sciknoweval path issue
RUN mkdir -p /app/src/inspect_evals/sciknoweval && \
    ln -sf /app/inspect_evals/src/inspect_evals/sciknoweval/evaluator_prompt.yaml /app/src/inspect_evals/sciknoweval/evaluator_prompt.yaml

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
