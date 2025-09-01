FROM python:3.13-slim

WORKDIR /app

# Copy code & manifest
COPY manifest.yaml .
COPY entrypoint.py .

# Install build dependencies and pycocotools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && pip install --no-cache-dir pycocotools \
    && apt-get remove -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
