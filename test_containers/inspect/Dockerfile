FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install inspect_ai from PyPI and inspect_evals from GitHub
RUN pip install --no-cache-dir inspect-ai
RUN pip install --no-cache-dir git+https://github.com/UKGovernmentBEIS/inspect_evals.git
# Install additional dependencies needed for OpenAI
RUN pip install --no-cache-dir openai
# Install sympy for math evaluation
RUN pip install --no-cache-dir sympy

# Copy test container files
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]
