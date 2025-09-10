FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Docker
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

WORKDIR /app

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

RUN pip install --no-cache-dir openai==1.104.1

# Copy test container files
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

# Set environment variable to acknowledge Cybench security risks
ENV CYBENCH_ACKNOWLEDGE_RISKS=1

ENTRYPOINT ["/app/inspect_evals/.venv/bin/python", "./entrypoint.py"]
