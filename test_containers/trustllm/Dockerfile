FROM python:3.11-slim

WORKDIR /app

# Install git, build tools, and unzip for building packages and extracting datasets
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone TrustLLM repository, install package, and extract dataset
RUN git clone https://github.com/timlrx/TrustLLM.git /app/trustllm && \
    cd /app/trustllm/trustllm_pkg && \
    pip install --no-cache-dir . && \
    cd /app/trustllm && \
    if [ -f "dataset/dataset.zip" ]; then \
        unzip dataset/dataset.zip -d /app/dataset/; \
        echo "Dataset extracted successfully"; \
    else \
        mkdir -p /app/dataset/dataset && \
        echo "Dataset zip not found, created empty dataset directory"; \
    fi && \
    cd /app && \
    rm -rf /app/trustllm

# Pre-download HuggingFace models used by TrustLLM evaluators
RUN python -c "\
import os; \
os.environ['HF_HOME'] = '/app/.cache/huggingface'; \
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
print('Downloading LibrAI/longformer-harmful-ro model...'); \
model = AutoModelForSequenceClassification.from_pretrained('LibrAI/longformer-harmful-ro'); \
tokenizer = AutoTokenizer.from_pretrained('LibrAI/longformer-harmful-ro'); \
print('Model download completed'); \
"

# Set HuggingFace cache directory for runtime
ENV HF_HOME=/app/.cache/huggingface

COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
