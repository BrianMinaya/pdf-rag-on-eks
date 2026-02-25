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
services/
  chat-api/                 FastAPI RAG orchestrator + web chat UI
  embedding/                Nomic V1.5 embedding wrapper
  ingestion/                PDF parsing + chunking + embedding pipeline
scripts/
  build-and-push.sh           Build Docker images and push to ECR
  deploy-all.sh               Apply all K8s manifests
  startup.sh                  Scale up GPU node + port-forward
  shutdown.sh                 Scale down GPU node to zero
  chat.py                     Interactive CLI chat client
  test-rag.sh                 Quick RAG smoke test
docs/diagrams/              Interactive HTML architecture diagrams
data/                       PDF files (gitignored)
```

## Prerequisites

- **AWS Account** with permissions for EKS, EC2, ECR, VPC, IAM
- **GPU quota**: at least 4 vCPUs for G-family spot instances in us-east-1
- **CLI tools**: AWS CLI, OpenTofu v1.5+, kubectl, Docker
- **HuggingFace token**: for pulling gated model weights (Llama 3.1)

## Quick Start

```bash
# 1. Provision AWS infrastructure
cd terraform
tofu init && tofu apply

# 2. Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name pdf-rag-chatbot

# 3. Build and push Docker images to ECR
./scripts/build-and-push.sh

# 4. Create your secrets file
cp k8s/secrets.yaml.example k8s/secrets.yaml
# Edit k8s/secrets.yaml with your actual credentials

# 5. Deploy all Kubernetes services
./scripts/deploy-all.sh

# 6. Place your PDF(s) in data/, then run ingestion
kubectl apply -f k8s/ingestion/job.yaml

# 7. Start up (scales GPU node, sets up port-forwarding)
./scripts/startup.sh

# 8. Open the web UI
open http://localhost:8000
```

## Cost Management

GPU instances are expensive. The included scripts make it easy to spin up only when needed:

```bash
# Done for the day? Scale GPU to zero (~$0.21/hr saved)
./scripts/shutdown.sh

# Ready to work again? Scale GPU back up
./scripts/startup.sh
```

Additional cost measures:
- GPU nodes use **spot instances** (up to 70% cheaper than on-demand)
- GPU desired size defaults to **0** — no GPU cost until you explicitly scale up
- CPU nodes use affordable **t3.xlarge** on-demand instances (~$0.17/hr)

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
