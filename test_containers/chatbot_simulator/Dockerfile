FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openevals==0.1.0 rich Jinja2

COPY manifest.yaml .
COPY entrypoint.py .
COPY simulation.py .
COPY evaluator.py .
COPY prompt_loader.py .
COPY prompts prompts/

RUN chmod +x entrypoint.py
ENTRYPOINT ["python", "./entrypoint.py"]
