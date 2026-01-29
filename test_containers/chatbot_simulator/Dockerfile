# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies (git required for uv to clone from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY manifest.yaml .
COPY entrypoint.py .
COPY simulation.py .
COPY evaluator.py .
COPY j2_utils.py .
COPY prompts/ ./prompts

RUN uv pip install --system --no-cache -e .

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed Python packages and application files from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
