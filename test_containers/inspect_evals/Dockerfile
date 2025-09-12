# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set environment variable to allow remote code execution for datasets
ENV HF_DATASETS_TRUST_REMOTE_CODE=1

# Clone inspect_evals and install with uv sync
RUN git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd inspect_evals && \
    git checkout 479a4685 && \
    uv venv && \
    uv sync --all-extras

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install Docker
RUN apt-get update && apt-get install -y \
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

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
