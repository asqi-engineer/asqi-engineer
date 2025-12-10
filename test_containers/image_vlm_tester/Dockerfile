# Use the official Python image as a base
FROM python:3.11-slim-bookworm

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entrypoint script, manifest, and evaluator
COPY entrypoint.py /app/entrypoint.py
COPY manifest.yaml /app/manifest.yaml
COPY evaluator.py /app/evaluator.py

RUN chmod +x /app/entrypoint.py
RUN ls -la /app

# Entrypoint for the container
ENTRYPOINT ["python", "/app/entrypoint.py"]
