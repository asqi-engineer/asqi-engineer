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

# Install inspect_ai from PyPI and inspect_evals from GitHub
RUN pip install --no-cache-dir inspect-ai==0.3.129
RUN pip install --no-cache-dir git+https://github.com/UKGovernmentBEIS/inspect_evals.git@main
# Install additional dependencies needed for OpenAI
RUN pip install --no-cache-dir openai
# Install sympy for math evaluation
RUN pip install --no-cache-dir sympy
# Install datasets with trust_remote_code support
RUN pip install --no-cache-dir datasets
RUN pip install --no-cache-dir git+https://github.com/josejg/instruction_following_eval
RUN pip install --no-cache-dir inspect-cyber==0.1.0

# Copy test container files
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

# Set environment variable to allow remote code execution for datasets
ENV HF_DATASETS_TRUST_REMOTE_CODE=1

# Set environment variable to acknowledge Cybench security risks
ENV CYBENCH_ACKNOWLEDGE_RISKS=1

ENTRYPOINT ["python", "./entrypoint.py"]
