FROM python:3.11-slim AS builder

WORKDIR /tmp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/timlrx/TrustLLM.git && \
    cd TrustLLM/trustllm_pkg && \
    pip install --no-cache-dir --user . && \
    cd /tmp/TrustLLM && \
    if [ -f "dataset/dataset.zip" ]; then \
        unzip dataset/dataset.zip -d /tmp/dataset/; \
    else \
        mkdir -p /tmp/dataset/dataset; \
    fi

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /tmp/dataset /app/dataset

ENV PATH=/root/.local/bin:$PATH

RUN python -c "\
import os; \
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'; \
os.environ['HF_HUB_CACHE'] = '/app/.cache/huggingface'; \
os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/huggingface'; \
os.environ['HF_HOME'] = '/app/.cache/huggingface'; \
os.environ['HUGGINGFACE_HUB_CACHE'] = '/app/.cache/huggingface'; \
print('Downloading LibrAI/longformer-harmful-ro model - PyTorch only...'); \
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
import torch; \
model = AutoModelForSequenceClassification.from_pretrained(\
    'LibrAI/longformer-harmful-ro', \
    torch_dtype=torch.float16, \
    use_safetensors=True \
); \
tokenizer = AutoTokenizer.from_pretrained('LibrAI/longformer-harmful-ro'); \
del model, tokenizer; \
import gc; \
gc.collect(); \
print('Model download completed'); \
" && \
# AGGRESSIVE CLEANUP - Remove ALL unnecessary files \
pip cache purge && \
find /app/.cache -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
find /root/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
find /app/.cache -name "*.tmp" -delete 2>/dev/null || true && \
find /app/.cache -name "*.temp" -delete 2>/dev/null || true && \
find /app/.cache -name "*.onnx" -delete 2>/dev/null || true && \
find /app/.cache -name "*.h5" -delete 2>/dev/null || true && \
find /app/.cache -name "tf_model*" -delete 2>/dev/null || true && \
find /app/.cache -name "rust_model*" -delete 2>/dev/null || true && \
find /app/.cache -name ".git*" -exec rm -rf {} + 2>/dev/null || true && \
find /app/.cache -name "*.log" -delete 2>/dev/null || true

# Set optimized environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface  
ENV HF_HUB_CACHE=/app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TOKENIZERS_PARALLELISM=false

COPY manifest.yaml entrypoint.py ./

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
