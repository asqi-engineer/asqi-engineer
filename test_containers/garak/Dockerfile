
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir --user garak==0.12.0

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH="/root/.local/bin:${PATH}"

COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]