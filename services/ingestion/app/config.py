"""
Configuration module for the Ingestion Service.

Uses pydantic-settings to load all configuration from environment variables.
This means every setting below can be overridden by setting an environment
variable with the same name (case-insensitive). For example, setting
QDRANT_HOST=my-qdrant-server in the environment will override the default
"qdrant" value below.

Why environment variables?
--------------------------
In Kubernetes (where this service runs), configuration is injected via
environment variables or ConfigMaps. This pattern keeps secrets and
environment-specific values out of the codebase entirely.
"""

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    All configuration for the ingestion pipeline, loaded from environment
    variables with sensible defaults for local development.
    """

    # -------------------------------------------------------------------------
    # Qdrant (Vector Database) Connection
    # -------------------------------------------------------------------------
    # Qdrant is the vector database where we store embeddings (numeric
    # representations of text meaning). The ingestion service writes to it;
    # the chat API reads from it at query time.
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    # The "collection" in Qdrant is like a table in a relational database.
    # Each collection holds vectors of a single dimensionality.
    qdrant_collection: str = "pdf_chunks"

    # -------------------------------------------------------------------------
    # Embedding Model Server
    # -------------------------------------------------------------------------
    # The embedding server runs Nomic Embed Text V1.5, which converts text
    # into 768-dimensional vectors. We call it over HTTP so the model runs
    # on its own pod and can be scaled independently.
    embedding_url: str = "http://embedding:8080"
    # Nomic Embed Text V1.5 outputs 768-dimensional vectors. This MUST match
    # the model -- if you swap to a different embedding model, update this.
    embedding_dimension: int = 768
    # How many texts to send to the embedding server in one HTTP request.
    # Larger batches = better GPU utilization but more memory. 32 is a safe
    # default for most hardware.
    embedding_batch_size: int = 32

    # -------------------------------------------------------------------------
    # Chunking Parameters
    # -------------------------------------------------------------------------
    # chunk_size: How many tokens each chunk should contain. 512 tokens is a
    # sweet spot -- small enough to be specific (so the retriever finds
    # relevant content), but large enough to contain meaningful context.
    #
    # chunk_overlap: How many tokens overlap between consecutive chunks. This
    # ensures that if an important concept spans a chunk boundary, it still
    # appears fully in at least one chunk. 50 tokens (~10% of 512) is a
    # common starting point.
    chunk_size: int = 512
    chunk_overlap: int = 50

    # -------------------------------------------------------------------------
    # PDF Source Directory
    # -------------------------------------------------------------------------
    # PDFs are read from a local directory. To ingest from a different source,
    # replace the PDF parser with your own fetcher.
    pdf_directory: str = "/data"

    # pydantic-settings model configuration. env_file lets you use a .env
    # file during local development instead of exporting variables manually.
    model_config = {"env_file": ".env"}
