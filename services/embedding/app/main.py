"""
Embedding Server -- FastAPI wrapper around sentence-transformers.

This is a lightweight alternative to HuggingFace's TEI (Text Embeddings Inference)
server. We use this because the official TEI Docker image does not ship pre-built
ARM64 images, and our CPU nodes are ARM-based (Graviton t4g.large).

This server provides the same /embed endpoint that TEI exposes, so the rest of
our codebase (ingestion service, chat API) works without any changes.

The embedding model (Nomic Embed Text V1.5) is downloaded from HuggingFace on
first startup and cached locally. Subsequent starts reuse the cached model.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model instance. Loaded once on startup, reused for all requests.
# sentence-transformers handles tokenization, inference, and pooling for us.
# ---------------------------------------------------------------------------
model: SentenceTransformer | None = None

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model on startup."""
    global model
    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    logger.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())
    yield
    logger.info("Shutting down embedding server.")


app = FastAPI(
    title="Embedding Server",
    description="Serves Nomic Embed Text V1.5 embeddings",
    version="0.1.0",
    lifespan=lifespan,
)


class EmbedRequest(BaseModel):
    """Request body matching the TEI /embed endpoint format."""
    inputs: list[str]


@app.post("/embed")
async def embed(request: EmbedRequest) -> list[list[float]]:
    """
    Generate embeddings for a list of input texts.

    This endpoint matches the TEI /embed API format so the rest of our
    codebase (ingestion, chat-api) works without modification.

    The caller is responsible for adding the Nomic prefix
    ("search_document: " or "search_query: ") before calling this endpoint.
    """
    embeddings = model.encode(request.inputs, convert_to_numpy=True)
    return embeddings.tolist()


@app.get("/")
async def health():
    """Health check endpoint for Kubernetes readiness probe."""
    return {"status": "ready", "model": MODEL_NAME}
