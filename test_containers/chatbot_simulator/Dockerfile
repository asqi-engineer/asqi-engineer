# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools (only in builder)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a temporary /install directory
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install openevals==0.1.0 rich

# Stage 2: Runtime (minimal, no build tools)
FROM python:3.11-slim

WORKDIR /app

# Copy just the built Python packages
COPY --from=builder /install /usr/local

# Copy your app code
COPY manifest.yaml .
COPY entrypoint.py .
COPY simulation.py .
COPY evaluator.py .


RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]
