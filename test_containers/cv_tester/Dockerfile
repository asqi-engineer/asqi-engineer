FROM python:3.11-slim

# Make Python logs unbuffered and avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy app files
COPY manifest.yaml .
COPY entrypoint.py .
COPY evaluator.py .
COPY detectors.py .

# System dependencies for OpenCV/TorchVision
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
        ultralytics \
        opencv-python-headless \
        numpy \
        python-dotenv \
        torch \
        torchvision

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
