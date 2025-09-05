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

# Install inspect_ai from PyPI
RUN pip install --no-cache-dir inspect-ai==0.3.129

# Clone inspect_evals and install with extras using uv pip
RUN git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd inspect_evals && \
    git checkout 479a4685 && \
    uv pip install --system -e .[core_bench,livebench,agentdojo,ifeval,cybench,bold,bbq,math,sevenllm,mle_bench]

RUN pip install --no-cache-dir openai==1.104.1

# Copy test container files
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

# Set environment variable to allow remote code execution for datasets
ENV HF_DATASETS_TRUST_REMOTE_CODE=1

# Set environment variable to acknowledge Cybench security risks
ENV CYBENCH_ACKNOWLEDGE_RISKS=1

ENTRYPOINT ["python", "./entrypoint.py"]
