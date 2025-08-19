FROM python:3.11-slim

WORKDIR /app

COPY manifest.yaml .
COPY entrypoint.py .

# Install system dependencies for OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
        ultralytics \
        opencv-python-headless \
        numpy

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
