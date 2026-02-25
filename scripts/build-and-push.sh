#!/usr/bin/env bash
# =============================================================================
# Build and Push Docker Images to ECR
#
# Builds the chat-api and ingestion service Docker images for x86_64 (amd64)
# matching the t3.large CPU nodes, and pushes them to ECR.
#
# Since we're building on Apple Silicon (ARM) but targeting x86 nodes, we use
# docker buildx for cross-platform compilation. Docker Desktop includes QEMU
# which handles the architecture translation automatically.
#
# Prerequisites:
#   - AWS CLI configured
#   - Docker Desktop running (provides buildx + QEMU)
#   - Terraform has been applied (ECR repos exist)
#
# Usage: ./scripts/build-and-push.sh
# =============================================================================
set -euo pipefail

# --- Configuration ---
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-pdf-rag-chatbot}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "=== Build and Push Docker Images ==="
echo "Account:  ${ACCOUNT_ID}"
echo "Region:   ${AWS_REGION}"
echo "ECR Base: ${ECR_BASE}"
echo "Platform: linux/amd64 (x86_64)"
echo ""

# --- Step 1: Authenticate Docker to ECR ---
echo ">>> Authenticating Docker to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_BASE}"
echo ""

# --- Step 2: Build and push each service ---
# We use docker buildx to cross-compile from ARM (Apple Silicon) to x86_64
# (AWS t3.large nodes). The --push flag pushes directly to ECR after building.
SERVICES=("chat-api" "ingestion")

for SERVICE in "${SERVICES[@]}"; do
  IMAGE="${ECR_BASE}/${PROJECT_NAME}/${SERVICE}:latest"
  echo ">>> Building ${SERVICE} (linux/amd64) -> ${IMAGE}"

  docker buildx build \
    --platform linux/amd64 \
    --tag "${IMAGE}" \
    --push \
    "services/${SERVICE}"

  echo ">>> ${SERVICE} pushed successfully"
  echo ""
done

echo "=== All images built and pushed ==="
echo ""
echo "Verify with:"
echo "  aws ecr list-images --repository-name ${PROJECT_NAME}/chat-api --region ${AWS_REGION}"
echo "  aws ecr list-images --repository-name ${PROJECT_NAME}/ingestion --region ${AWS_REGION}"
