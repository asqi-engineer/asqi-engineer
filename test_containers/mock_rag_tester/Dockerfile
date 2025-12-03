FROM python:3.12-slim

WORKDIR /app

# Install uv for faster Python dependency management
RUN pip install --no-cache-dir uv

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copy validation code from main library
# Note: rag_response_schema.py is copied from src/asqi during package development
# Keep this file in sync with src/asqi/rag_response_schema.py
# Alternative: Install asqi-engineer package and import from asqi.rag_response_schema
# RUN uv pip install --no-cache-dir --system asqi-engineer
COPY rag_response_schema.py .
COPY manifest.yaml .
COPY entrypoint.py .

RUN chmod +x entrypoint.py

ENTRYPOINT ["python", "./entrypoint.py"]


