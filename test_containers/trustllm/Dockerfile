# Multi-stage build to completely eliminate build dependencies
FROM python:3.11-slim AS builder

WORKDIR /tmp

# Install build dependencies only in builder stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Clone and install TrustLLM package with --user flag
RUN git clone https://github.com/timlrx/TrustLLM.git && \
    cd TrustLLM/trustllm_pkg && \
    pip install --no-cache-dir --user . && \
    cd /tmp/TrustLLM && \
    if [ -f "dataset/dataset.zip" ]; then \
        unzip dataset/dataset.zip -d /tmp/dataset/; \
    else \
        mkdir -p /tmp/dataset/dataset; \
    fi

# Final runtime stage - completely clean
FROM python:3.11-slim

WORKDIR /app

# Copy only the compiled packages and dataset
COPY --from=builder /root/.local /root/.local
COPY --from=builder /tmp/dataset /app/dataset

# Ensure installed packages are in PATH
ENV PATH=/root/.local/bin:$PATH

# CRITICAL: Pre-download model with optimizations
RUN python -c "\
import os; \
# Force only PyTorch format, disable other formats \
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; \
os.environ['HF_HUB_CACHE'] = '/app/.cache/huggingface'; \
os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/huggingface'; \
os.environ['HF_HOME'] = '/app/.cache/huggingface'; \
# Disable unnecessary model format downloads \
os.environ['HUGGINGFACE_HUB_CACHE'] = '/app/.cache/huggingface'; \
print('Downloading LibrAI/longformer-harmful-ro model - PyTorch only...'); \
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
import torch; \
# Use float16 to halve model size in memory and on disk \
model = AutoModelForSequenceClassification.from_pretrained(\
    'LibrAI/longformer-harmful-ro', \
    torch_dtype=torch.float16, \
    use_safetensors=True \
); \
tokenizer = AutoTokenizer.from_pretrained('LibrAI/longformer-harmful-ro'); \
# Force garbage collection \
del model, tokenizer; \
import gc; \
gc.collect(); \
print('Model download completed'); \
" && \
# AGGRESSIVE CLEANUP - Remove ALL unnecessary files \
pip cache purge && \
# Remove all __pycache__ directories \
find /app/.cache -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
# Remove temporary files \
find /app/.cache -name "*.tmp" -delete 2>/dev/null || true && \
find /app/.cache -name "*.temp" -delete 2>/dev/null || true && \
# Remove duplicate model formats if they exist \
find /app/.cache -name "*.onnx" -delete 2>/dev/null || true && \
find /app/.cache -name "*.h5" -delete 2>/dev/null || true && \
find /app/.cache -name "tf_model*" -delete 2>/dev/null || true && \
find /app/.cache -name "rust_model*" -delete 2>/dev/null || true && \
# Remove git-related files from cache \
find /app/.cache -name ".git*" -exec rm -rf {} + 2>/dev/null || true && \
# Remove any logs \
find /app/.cache -name "*.log" -delete 2>/dev/null || true

# Set optimized environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface  
ENV HF_HUB_CACHE=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false

# Copy only essential files
COPY manifest.yaml entrypoint.py ./

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
