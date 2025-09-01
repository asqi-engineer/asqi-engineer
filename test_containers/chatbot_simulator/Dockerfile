FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openevals==0.1.0 rich

COPY manifest.yaml .
COPY entrypoint.py .
COPY simulation.py .
COPY evaluator.py .

RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]
