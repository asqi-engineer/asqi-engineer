FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (git required for uv to clone from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY manifest.yaml .
COPY entrypoint.py .

RUN uv pip install --system --no-cache -e .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
