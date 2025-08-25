FROM python:3.11-slim

# Make Python logs unbuffered and avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy app files
COPY manifest.yaml .
COPY entrypoint.py .
COPY evaluator.py .

# Minimal runtime deps for opencv-python-headless + HTTPS
# (libglib and image codecs are typically required by OpenCV;
# ca-certificates for requests over HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libjpeg62-turbo \
        libpng16-16 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies actually used by the scripts
RUN pip install --no-cache-dir \
        opencv-python-headless \
        numpy \
        requests \
        python-dotenv

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
