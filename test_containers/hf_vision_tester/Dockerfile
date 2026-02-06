FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY test_containers/hf_vision_tester/pyproject.toml .
COPY test_containers/hf_vision_tester/manifest.yaml .
COPY test_containers/hf_vision_tester/entrypoint.py .

COPY src/asqi /tmp/asqi_src/src/asqi
RUN printf '[project]\nname = "asqi-engineer"\nversion = "0.0.0"\nrequires-python = ">=3.12"\ndependencies = ["datasets[vision]>=4.4.1","pydantic>=2.11.7","pyyaml>=6.0.2"]\n\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n[tool.hatch.build.targets.wheel]\npackages = ["src/asqi"]\n' > /tmp/asqi_src/pyproject.toml \
    && uv pip install --system --no-cache /tmp/asqi_src/ \
    && rm -rf /tmp/asqi_src/

RUN uv pip install --system --no-cache -e .

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

ENTRYPOINT ["python", "./entrypoint.py"]
