FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY manifest.yaml .
COPY entrypoint.py .

COPY asqi_pkg/ /tmp/asqi_pkg/
RUN uv pip install --system --no-cache /tmp/asqi_pkg/ && rm -rf /tmp/asqi_pkg/
RUN uv pip install --system --no-cache -e .

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

ENTRYPOINT ["python", "./entrypoint.py"]
