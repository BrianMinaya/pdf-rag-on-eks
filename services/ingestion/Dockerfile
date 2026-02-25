# =============================================================================
# Ingestion Service Dockerfile -- Multi-stage build for x86_64 (AWS t3.xlarge)
#
# This Dockerfile builds the PDF ingestion pipeline. It parses PDFs, chunks
# the text, generates embeddings via the TEI server, and stores vectors in
# Qdrant. Runs as a one-shot Kubernetes Job (not a long-running service).
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/

# The ingestion service runs as a script, not a server
# It processes all PDFs in the /data directory and exits
CMD ["python", "-m", "app.main"]
