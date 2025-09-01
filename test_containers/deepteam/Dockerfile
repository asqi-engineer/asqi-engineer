FROM python:3.13-slim

WORKDIR /app

# Install Rust and Cargo
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure cargo is on PATH for subsequent RUN commands
ENV PATH="/root/.cargo/bin:${PATH}"

# Install deepteam and its dependencies
RUN pip install --no-cache-dir deepteam==0.2.4 litellm==1.76.0
COPY manifest.yaml .
COPY entrypoint.py .

ENV DEEPTEAM_TELEMETRY_OPT_OUT="YES"
RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
