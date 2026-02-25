#!/usr/bin/env bash
# =============================================================================
# Deploy All Services to EKS
#
# Applies Kubernetes manifests in dependency order and waits for each
# service to become ready before proceeding to the next.
#
# Dependency order matters because:
#   1. Namespace must exist before anything else
#   2. ConfigMap/Secrets must exist before pods that reference them
#   3. Qdrant must be ready before ingestion or chat-api can connect
#   4. Embedding server must be ready before ingestion or chat-api can embed
#   5. vLLM must be ready before chat-api can generate answers
#
# Usage: ./scripts/deploy-all.sh
# =============================================================================
set -euo pipefail

NAMESPACE="pdf-rag-chatbot"

echo "=== Deploying PDF RAG Chatbot to EKS ==="
echo "Namespace: ${NAMESPACE}"
echo ""

# --- Step 1: Namespace and configuration ---
echo ">>> Applying namespace..."
kubectl apply -f k8s/namespace.yaml

echo ">>> Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml

echo ">>> Applying Secrets..."
kubectl apply -f k8s/secrets.yaml
echo ""

# --- Step 2: Storage layer (Qdrant + PostgreSQL) ---
echo ">>> Deploying Qdrant..."
kubectl apply -f k8s/qdrant/
echo "Waiting for Qdrant to be ready..."
kubectl -n "${NAMESPACE}" rollout status statefulset/qdrant --timeout=120s
echo ""

echo ">>> Deploying PostgreSQL..."
kubectl apply -f k8s/postgres/
echo "Waiting for PostgreSQL to be ready..."
kubectl -n "${NAMESPACE}" rollout status statefulset/postgres --timeout=120s
echo ""

# --- Step 3: Embedding server (must be ready before ingestion or chat-api) ---
echo ">>> Deploying Embedding Server..."
kubectl apply -f k8s/embedding/
echo "Waiting for Embedding Server to be ready (model download may take a few minutes)..."
kubectl -n "${NAMESPACE}" rollout status deployment/embedding --timeout=300s
echo ""

# --- Step 4: NVIDIA Device Plugin (must exist before vLLM) ---
echo ">>> Deploying NVIDIA Device Plugin..."
kubectl apply -f k8s/nvidia-device-plugin.yaml
echo ""

# --- Step 5: vLLM (must be ready before chat-api) ---
echo ">>> Deploying vLLM..."
kubectl apply -f k8s/vllm/
echo "Waiting for vLLM to be ready (model loading takes 2-3 minutes)..."
kubectl -n "${NAMESPACE}" rollout status deployment/vllm --timeout=600s
echo ""

# --- Step 6: Chat API ---
echo ">>> Deploying Chat API..."
kubectl apply -f k8s/chat-api/
echo "Waiting for Chat API to be ready..."
kubectl -n "${NAMESPACE}" rollout status deployment/chat-api --timeout=120s
echo ""

echo "=== All services deployed ==="
echo ""
echo "Check status:"
echo "  kubectl -n ${NAMESPACE} get pods"
echo ""
echo "To run ingestion:"
echo "  kubectl apply -f k8s/ingestion/job.yaml"
echo ""
echo "To test the chat API:"
echo "  kubectl port-forward svc/chat-api 8000:8000 -n ${NAMESPACE}"
echo "  curl http://localhost:8000/health"
