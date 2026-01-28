# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.10

# Clone llmperf v2.0
RUN git config --global http.sslverify false && \
    git clone --branch v2.0 --depth 1 https://github.com/ray-project/llmperf.git

WORKDIR /app/llmperf

RUN uv venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e . openai

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /app/llmperf/.venv /app/.venv
COPY --from=builder /python /python
COPY --from=builder /app/llmperf /app/llmperf
ENV PATH="/app/.venv/bin:$PATH"

COPY manifest.yaml .
COPY entrypoint.py .
COPY openai_client.py .
COPY utils.py .

RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]

