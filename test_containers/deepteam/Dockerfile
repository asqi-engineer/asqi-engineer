# Stage 1: Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache \
    curl https://sh.rustup.rs -sSf | sh -s -- -y

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo,sharing=locked \
    . "$HOME/.cargo/env" && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install deepteam==0.2.4 litellm==1.76.0

# Stage 2: Runtime (Python 3.11 slim)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY manifest.yaml .
COPY entrypoint.py .

ENV DEEPTEAM_TELEMETRY_OPT_OUT="YES"
RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]
