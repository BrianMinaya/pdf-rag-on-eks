# =============================================================================
# Embedding Server Dockerfile -- ARM64 (AWS Graviton)
#
# Lightweight FastAPI server wrapping sentence-transformers for Nomic Embed
# Text V1.5. This replaces the official TEI image which lacks ARM64 support.
#
# The model is downloaded from HuggingFace on first container start and
# cached in the container. For faster restarts, consider mounting a
# persistent volume at /root/.cache/huggingface.
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
