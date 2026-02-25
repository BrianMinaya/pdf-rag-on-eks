# PDF RAG on EKS

A production-style Retrieval-Augmented Generation (RAG) chatbot deployed on AWS EKS. Upload PDFs, ask questions in natural language, and get grounded answers with source citations — powered by Llama 3.1 8B and a fully orchestrated Kubernetes pipeline.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Chat API (FastAPI)                  │
│                                                 │
│  1. Embed question ──► Embedding Server          │
│  2. Search vectors  ──► Qdrant                   │
│  3. Generate answer ──► vLLM (Llama 3.1 8B)     │
│  4. Return answer + citations                    │
└─────────────────────────────────────────────────┘

PDF Ingestion (batch job):
  PDF ──► Parse ──► Chunk (512 tokens) ──► Embed ──► Store in Qdrant
```

## Tech Stack

| Component | Technology | Runs On |
|-----------|-----------|---------|
| **RAG Orchestrator** | FastAPI (Python) | CPU node (t3.xlarge) |
| **Ingestion Pipeline** | Custom Python service | CPU node |
| **LLM Inference** | vLLM + Llama 3.1 8B Instruct AWQ INT4 | GPU node (g4dn.xlarge, spot) |
| **Embeddings** | HuggingFace TEI + Nomic Embed Text V1.5 | CPU node |
| **Vector Database** | Qdrant (StatefulSet) | CPU node |
| **Conversation Store** | PostgreSQL (StatefulSet) | CPU node |
| **Infrastructure** | AWS EKS, VPC, ECR via OpenTofu | us-east-1 |

## What This Demonstrates

- **RAG pipeline from scratch** — no LangChain, direct API calls for full transparency and debuggability
- **Kubernetes orchestration** — 6 services across CPU and GPU node groups with health checks, resource limits, and persistent storage
- **GPU workload scheduling** — spot instances, node taints/tolerations, scale-to-zero for cost control
- **Infrastructure as Code** — full AWS environment (VPC, EKS, ECR, IAM/IRSA) via OpenTofu modules
- **Cost optimization** — GPU spot instances (~70% savings), startup/shutdown scripts, scale-to-zero GPU nodes
- **Production patterns** — idempotent ingestion, deterministic vector IDs, incremental updates via content hashing

## Project Structure

```
terraform/                  AWS infrastructure (VPC, EKS, ECR)
k8s/                        Kubernetes manifests
  chat-api/                   Chat API deployment + service
  embedding/                  Embedding server deployment + service
  ingestion/                  Ingestion job + PVC
  postgres/                   PostgreSQL StatefulSet + service
  qdrant/                     Qdrant StatefulSet + service
  vllm/                       vLLM deployment + service
  configmap.yaml              Shared environment config
  namespace.yaml              Namespace definition
  secrets.yaml.example        Template for K8s secrets (copy to secrets.yaml)
services/
  chat-api/                 FastAPI RAG orchestrator + web chat UI
  ingestion/                PDF parsing + chunking + embedding pipeline
scripts/
  build-and-push.sh           Build Docker images and push to ECR
  startup.sh                  Scale up node groups
  shutdown.sh                 Scale down node groups to zero
  chat.py                     Interactive CLI chat client (pip install httpx)
  test-rag.sh                 Quick RAG smoke test
docs/diagrams/              Interactive HTML architecture diagrams
data/                       PDF files (gitignored)
```

## Prerequisites

- **AWS Account** with permissions for EKS, EC2, ECR, VPC, IAM
- **GPU quota**: at least 4 vCPUs for G-family spot instances in us-east-1
- **CLI tools**: AWS CLI, OpenTofu v1.5+, kubectl, Docker
- **HuggingFace account**: Llama 3.1 is a gated model — [accept Meta's license](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) on HuggingFace, then [create an access token](https://huggingface.co/settings/tokens)

## Quick Start

### 1. Provision infrastructure

```bash
cd terraform
tofu init && tofu apply
```

### 2. Configure kubectl

```bash
aws eks update-kubeconfig --region us-east-1 --name pdf-rag-chatbot
```

### 3. Build and push Docker images

```bash
./scripts/build-and-push.sh
```

This builds `chat-api` and `ingestion` for linux/amd64 and pushes to ECR. The account ID is detected automatically.

### 4. Update K8s image references

Replace the `<YOUR_AWS_ACCOUNT_ID>` placeholder in these two files with your AWS account ID:

- `k8s/chat-api/deployment.yaml` (line 52)
- `k8s/ingestion/job.yaml` (line 87)

```bash
# Find your account ID
aws sts get-caller-identity --query Account --output text

# Replace in both files (macOS sed)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
sed -i '' "s/<YOUR_AWS_ACCOUNT_ID>/$ACCOUNT_ID/g" k8s/chat-api/deployment.yaml k8s/ingestion/job.yaml
```

### 5. Create secrets

```bash
cp k8s/secrets.yaml.example k8s/secrets.yaml
```

Edit `k8s/secrets.yaml` — uncomment `HF_TOKEN` and paste your HuggingFace token. Change `POSTGRES_PASSWORD` to something secure.

### 6. Deploy services

Apply manifests in dependency order:

```bash
# Namespace and config first
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Storage layer
kubectl apply -f k8s/qdrant/
kubectl apply -f k8s/postgres/

# Embedding server (takes 1-2 min to download model)
kubectl apply -f k8s/embedding/

# NVIDIA device plugin (required before vLLM)
kubectl apply -f k8s/nvidia-device-plugin.yaml

# Chat API (works without vLLM — will just fail on LLM calls until vLLM is up)
kubectl apply -f k8s/chat-api/
```

### 7. Scale up GPU and deploy vLLM

```bash
# Scale up the GPU node (takes ~90 seconds)
./scripts/startup.sh

# Wait for the GPU node to join
kubectl get nodes -w

# Deploy vLLM (takes 3-5 min to download and load model)
kubectl apply -f k8s/vllm/

# Monitor progress
kubectl -n pdf-rag-chatbot get pods -w
```

### 8. Upload PDFs and run ingestion

The ingestion job reads PDFs from a PersistentVolumeClaim (PVC) inside the cluster, not from your local filesystem.

```bash
# Create the PVC
kubectl apply -f k8s/ingestion/pvc.yaml

# Spin up a temporary pod to copy files into the PVC
kubectl run pdf-loader --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"pdf-loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"pdf-data","mountPath":"/data"}]}],"volumes":[{"name":"pdf-data","persistentVolumeClaim":{"claimName":"pdf-data"}}]}}' \
  -n pdf-rag-chatbot

# Copy your PDF into the PVC
kubectl cp my-document.pdf pdf-rag-chatbot/pdf-loader:/data/my-document.pdf

# Clean up the loader pod
kubectl delete pod pdf-loader -n pdf-rag-chatbot

# Run ingestion
kubectl apply -f k8s/ingestion/job.yaml

# Watch progress
kubectl logs -f job/ingestion -n pdf-rag-chatbot
```

To re-run ingestion (e.g., after adding new PDFs), delete the old job first:

```bash
kubectl delete job ingestion -n pdf-rag-chatbot
kubectl apply -f k8s/ingestion/job.yaml
```

### 9. Use the chatbot

```bash
# Port-forward the Chat API to your local machine
kubectl port-forward svc/chat-api 8000:8000 -n pdf-rag-chatbot &

# Web UI
open http://localhost:8000

# Or use the CLI chat client (requires: pip install httpx)
python scripts/chat.py

# Or hit the API directly
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Cost Management

GPU instances are expensive. The included scripts make it easy to spin up only when needed:

```bash
# Done for the day? Scale all nodes to zero
./scripts/shutdown.sh

# Ready to work again? Scale nodes back up
./scripts/startup.sh
```

Additional cost measures:
- GPU nodes use **spot instances** (up to 70% cheaper than on-demand)
- GPU desired size defaults to **0** — no GPU cost until you explicitly scale up
- CPU nodes use affordable **t3.xlarge** on-demand instances (~$0.17/hr)
- EKS control plane runs at ~$0.10/hr regardless of node count

## How It Works

### Ingestion (batch)
1. **Parse** — Extract text from PDFs as Markdown (preserving structure)
2. **Chunk** — Split into 512-token windows with 50-token overlap
3. **Embed** — Convert each chunk to a 768-dim vector via Nomic Embed V1.5
4. **Store** — Upsert vectors + metadata into Qdrant with deterministic IDs

### Query (real-time)
1. **Embed** — Convert the user's question to a vector (same model, "search_query:" prefix)
2. **Retrieve** — Find the top-5 most similar chunks in Qdrant (cosine similarity)
3. **Generate** — Send question + retrieved chunks to Llama 3.1 8B via vLLM
4. **Respond** — Return the answer with source citations (filename, page number)

## Swapping Data Sources

The ingestion pipeline is designed to be data-source agnostic. Only `services/ingestion/app/pdf_parser.py` is PDF-specific — the rest of the pipeline (chunking, embedding, vector storage) works identically regardless of input source.

To ingest from a different source, replace `pdf_parser.py` with your own fetcher that returns `Document` objects (text + metadata). Examples:
- REST API (Notion, Confluence, SharePoint, etc.)
- Web scraper
- S3 bucket of documents

## Architecture Diagrams

Interactive HTML diagrams are available in `docs/diagrams/`:
- **[Application Topology](docs/diagrams/application-topology.html)** — service interactions, data flow, K8s resources
- **[Infrastructure Topology](docs/diagrams/infrastructure-topology.html)** — AWS VPC, EKS, node groups, networking
