# =============================================================================
# Chat API Dockerfile -- Multi-stage build for x86_64 (AWS t3.xlarge)
#
# This Dockerfile builds the FastAPI RAG orchestrator service. It runs on
# x86 CPU nodes (t3.xlarge) in EKS.
#
# Multi-stage builds keep the final image small by separating the build
# environment (with compilers, headers) from the runtime environment.
# =============================================================================

# Stage 1: Build dependencies
# We use a slim Python image to install packages. Some Python packages
# (like uvloop) need C compilation, so we do that here.
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies that some Python packages need for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image
# This stage only contains the Python runtime and our installed packages.
# No compilers, no build tools -- keeps the image small and secure.
FROM python:3.12-slim

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# FastAPI runs on port 8000 by default with uvicorn
EXPOSE 8000

# Run the FastAPI app with uvicorn
# --host 0.0.0.0: Listen on all interfaces (required in containers)
# --workers 2: Run 2 worker processes for concurrency
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
