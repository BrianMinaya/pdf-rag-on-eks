"""
Ingestion Service -- Main Orchestrator

This is the entry point for the RAG ingestion pipeline. It runs as a
ONE-SHOT batch job (not a long-running server). In Kubernetes, this is
deployed as a CronJob that runs on a schedule (e.g., every 6 hours) or
can be triggered manually.

=== The RAG Ingestion Pipeline (what this script does) ===

RAG stands for "Retrieval-Augmented Generation." It's a technique where we:
  1. INGEST: Pre-process documents into searchable chunks (this script)
  2. RETRIEVE: At query time, find the most relevant chunks for a question
  3. GENERATE: Send those chunks + the question to an LLM to get an answer

This script handles step 1 -- the ingestion pipeline:

  PDF Files                          Qdrant Vector Database
  (or other data sources)              (searchable knowledge base)
       |                                      ^
       v                                      |
  [1. PARSE]                           [5. STORE]
  Extract text as Markdown              Upsert vectors + metadata
       |                                      ^
       v                                      |
  [2. CHUNK]                           [4. EMBED]
  Split into 512-token windows          Convert text to 768-dim vectors
  with 50-token overlap                 via Nomic Embed Text V1.5
       |                                      ^
       v                                      |
  [3. COLLECT] -----> texts -----> [TEI Server]

After this pipeline runs, the Qdrant collection contains all document chunks
as searchable vectors. The Chat API service can then query this collection
to find relevant context for user questions.

=== Idempotency ===
This pipeline is designed to be idempotent -- running it multiple times on
the same documents produces the same result. This is achieved through:
  - Deterministic UUIDs based on content hashes (same content = same ID)
  - Upsert operations (insert-or-update, not insert-only)
So if a CronJob fires twice by mistake, no harm done.
"""

import asyncio
import logging
import sys
from pathlib import Path

from app.chunker import chunk_documents
from app.config import Config
from app.embedder import EmbeddingClient
from app.pdf_parser import parse_pdf
from app.vector_store import VectorStore

# Configure logging for the entire ingestion run.
# We use structured log messages with consistent formatting so they're
# easy to parse in Kubernetes log aggregation tools (e.g., CloudWatch,
# Datadog, Grafana Loki).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Run the full ingestion pipeline:
      1. Load configuration
      2. Discover and parse PDF files
      3. Chunk documents into fixed-size token windows
      4. Initialize the vector store (create collection if needed)
      5. Embed all chunks via the TEI server
      6. Store embeddings + metadata in Qdrant
      7. Log summary statistics
    """

    # --- Step 1: Load configuration from environment variables ---
    # All settings (Qdrant host, embedding URL, chunk sizes, etc.) come from
    # environment variables, with defaults for local development. See config.py.
    config = Config()
    logger.info("Configuration loaded:")
    logger.info("  Qdrant: %s:%d, collection='%s'", config.qdrant_host, config.qdrant_port, config.qdrant_collection)
    logger.info("  Embedding: %s (dim=%d, batch=%d)", config.embedding_url, config.embedding_dimension, config.embedding_batch_size)
    logger.info("  Chunking: size=%d, overlap=%d", config.chunk_size, config.chunk_overlap)
    logger.info("  PDF directory: %s", config.pdf_directory)

    # --- Step 2: Discover PDF files ---
    # In this project, we read PDFs from a local directory. To use a different
    # data source, replace the PDF parser with your own fetcher (e.g. REST API,
    # web scraper, S3 bucket).
    pdf_dir = Path(config.pdf_directory)
    if not pdf_dir.exists():
        logger.error("PDF directory does not exist: %s", pdf_dir)
        sys.exit(1)

    # Find all PDF files (case-insensitive matching for .pdf and .PDF).
    pdf_files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    if not pdf_files:
        logger.warning("No PDF files found in %s -- nothing to ingest", pdf_dir)
        sys.exit(0)

    logger.info("Found %d PDF file(s) to process", len(pdf_files))

    # --- Step 3: Parse and chunk all PDFs ---
    # We process all PDFs first, then embed all chunks together. This is
    # more efficient than processing one PDF at a time because we can send
    # larger batches to the embedding server.
    all_chunks = []

    for pdf_path in pdf_files:
        logger.info("--- Processing: %s ---", pdf_path.name)

        # Parse: Extract text from PDF as Markdown, one Document per page.
        documents = parse_pdf(str(pdf_path))

        # Chunk: Split each page's text into overlapping 512-token windows.
        chunks = chunk_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        logger.info(
            "Parsed %d pages, created %d chunks from %s",
            len(documents),
            len(chunks),
            pdf_path.name,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks generated from any PDF -- nothing to ingest")
        sys.exit(0)

    logger.info("Total chunks to embed and store: %d", len(all_chunks))

    # --- Step 4: Initialize the vector store ---
    # Connect to Qdrant and ensure our collection exists. If this is the
    # first run, the collection is created with the correct vector dimension
    # and distance metric. On subsequent runs, this is a no-op.
    vector_store = VectorStore(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.qdrant_collection,
        vector_dimension=config.embedding_dimension,
    )
    vector_store.ensure_collection()

    # --- Step 5: Embed all chunks ---
    # This is typically the most time-consuming step. Each chunk's text is
    # sent to the TEI server (running Nomic Embed Text V1.5), which returns
    # a 768-dimensional vector representing the text's semantic meaning.
    #
    # These vectors are what enable semantic search: at query time, the
    # user's question is embedded with the same model, and Qdrant finds the
    # stored vectors closest to the query vector.
    embedding_client = EmbeddingClient(
        base_url=config.embedding_url,
        batch_size=config.embedding_batch_size,
    )

    # Collect just the text strings from all chunks for embedding.
    # The prefix "search_document: " is prepended by the embedding client
    # (required by Nomic V1.5 to distinguish documents from queries).
    chunk_texts = [chunk.text for chunk in all_chunks]

    logger.info("Embedding %d chunks via TEI server...", len(chunk_texts))
    embeddings = await embedding_client.embed_texts(chunk_texts)

    logger.info(
        "Embedding complete: %d vectors of dimension %d",
        len(embeddings),
        len(embeddings[0]) if embeddings else 0,
    )

    # --- Step 6: Store embeddings in Qdrant ---
    # Each chunk's vector and metadata (source, page number, text) are
    # upserted into Qdrant. "Upsert" means existing vectors with the same
    # deterministic ID are overwritten, making this pipeline idempotent.
    logger.info("Upserting %d chunks into Qdrant...", len(all_chunks))
    vector_store.upsert_chunks(all_chunks, embeddings)

    # --- Step 7: Log summary ---
    # Fetch and display collection statistics to verify the ingestion worked.
    # This is useful for monitoring and debugging in production.
    collection_info = vector_store.get_collection_info()
    logger.info(
        "Collection '%s' now contains %d points",
        config.qdrant_collection,
        collection_info.get("points_count", "unknown"),
    )

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    # asyncio.run() creates an event loop, runs the main() coroutine, and
    # then closes the loop. We use async because the embedding step benefits
    # from async HTTP calls (non-blocking I/O while waiting for the TEI
    # server to respond).
    asyncio.run(main())
