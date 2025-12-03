# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

RUN uv venv
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /python /python
ENV PATH="/app/.venv/bin:$PATH"

# Copy validation code from main library
# Note: rag_response_schema.py is copied from src/asqi during package development
# This ensures the test container uses the latest validation schema from development
# Alternative: Install asqi-engineer package and import from asqi.rag_response_schema
# RUN uv pip install --no-cache-dir --system asqi-engineer
COPY rag_response_schema.py .
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]


