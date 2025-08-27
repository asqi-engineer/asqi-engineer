FROM python:3.11-slim

WORKDIR /app

# Install Rust and Cargo
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure cargo is on PATH for subsequent RUN commands
ENV PATH="/root/.cargo/bin:${PATH}"

# Install garak and its dependencies
RUN pip install --no-cache-dir garak==0.12.0

COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
