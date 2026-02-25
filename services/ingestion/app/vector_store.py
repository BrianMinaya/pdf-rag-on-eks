"""
Vector Store Module -- Manages the Qdrant vector database.

=== What is a vector database? ===
A vector database is a specialized database designed to store and search
high-dimensional vectors (lists of numbers). Unlike a traditional SQL
database that searches by exact column values (WHERE name = 'Alice'), a
vector database searches by SIMILARITY -- finding vectors that are closest
to a query vector.

In our RAG pipeline:
  1. The ingestion service converts each text chunk into a 768-dimensional
     vector (embedding) and stores it in Qdrant.
  2. At query time, the Chat API converts the user's question into a vector
     and asks Qdrant: "Find me the 10 stored vectors most similar to this
     query vector."
  3. Qdrant returns those 10 chunks, which are then sent to the LLM as
     context for generating an answer.

=== Why Qdrant specifically? ===
We chose Qdrant over alternatives like pgvector or Pinecone because it offers:
  - Fast similarity search with HNSW indexing (a graph-based algorithm)
  - Rich metadata filtering (e.g., filter by source document)
  - A built-in web dashboard for debugging and inspecting vectors
  - A Python client with both sync and async APIs

=== What is cosine distance? ===
When comparing two vectors, we need a way to measure how "similar" they are.
Cosine distance measures the ANGLE between two vectors:
  - Distance = 0: Vectors point in the same direction (identical meaning)
  - Distance = 1: Vectors are perpendicular (unrelated meaning)
  - Distance = 2: Vectors point in opposite directions (opposite meaning)

Cosine distance ignores vector MAGNITUDE (length) and only cares about
DIRECTION. This is ideal for text embeddings because we care about what a
text means (direction), not how confident the model is (magnitude).

=== Why 768 dimensions? ===
The number of dimensions is determined by the embedding model. Nomic Embed
Text V1.5 outputs 768-dimensional vectors. Think of each dimension as
capturing one aspect of meaning -- tone, topic, specificity, formality, etc.
More dimensions = more nuanced representation, but also more storage and
computation. 768 is a good balance.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages all interactions with the Qdrant vector database.

    This class handles:
      - Creating the collection (if it doesn't exist)
      - Upserting (insert-or-update) chunk vectors
      - Querying collection info for monitoring
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_dimension: int = 768,
    ) -> None:
        """
        Parameters
        ----------
        host : str
            Qdrant server hostname (e.g., "qdrant" in Kubernetes).
        port : int
            Qdrant gRPC port (default 6333).
        collection_name : str
            Name of the collection to store vectors in. A collection in
            Qdrant is analogous to a table in PostgreSQL -- it holds all
            vectors of a given dimensionality with their metadata (payloads).
        vector_dimension : int
            Number of dimensions per vector. Must match the embedding model
            output. Nomic Embed Text V1.5 outputs 768 dimensions.
        """
        # We use the synchronous Qdrant client here because the ingestion
        # service is a batch job (not a web server), so async isn't needed
        # for Qdrant operations. The embedding calls are async because they
        # benefit from concurrent HTTP requests.
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

    def ensure_collection(self) -> None:
        """
        Create the Qdrant collection if it doesn't already exist.

        This is idempotent -- safe to call on every ingestion run. If the
        collection already exists, it does nothing.

        The collection is configured with:
          - Cosine distance metric (see module docstring for explanation)
          - Vector size matching our embedding model (768 for Nomic V1.5)
        """
        # Check if the collection already exists by listing all collections
        # and searching for our name.
        existing_collections = [
            c.name for c in self.client.get_collections().collections
        ]

        if self.collection_name in existing_collections:
            logger.info(
                "Collection '%s' already exists -- skipping creation",
                self.collection_name,
            )
            return

        # Create the collection with cosine distance.
        # VectorParams tells Qdrant:
        #   - size: Each vector has 768 numbers (must match Nomic V1.5 output)
        #   - distance: Use cosine similarity to compare vectors
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_dimension,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

        logger.info(
            "Created collection '%s' (dimension=%d, distance=Cosine)",
            self.collection_name,
            self.vector_dimension,
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update chunks and their embeddings into Qdrant.

        "Upsert" means "insert if new, update if exists." This is determined
        by the point ID -- if a point with the same ID already exists, its
        vector and payload are overwritten.

        Parameters
        ----------
        chunks : list[Chunk]
            The text chunks with metadata.
        embeddings : list[list[float]]
            The embedding vectors, one per chunk. Must be the same length as
            chunks.

        Why deterministic UUIDs?
        ------------------------
        We generate each point's UUID from the content_hash + chunk_index.
        This means:
          - The SAME chunk always gets the SAME UUID, regardless of when
            ingestion runs.
          - Re-ingesting the same document OVERWRITES existing points instead
            of creating duplicates.
          - This makes ingestion IDEMPOTENT -- running it multiple times
            produces the same result as running it once.

        Without deterministic IDs, every ingestion run would create new
        points, and the collection would grow without bound with duplicate
        content.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        # Build PointStruct objects for Qdrant.
        points: list[qdrant_models.PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings):
            # Generate a deterministic UUID from the content hash and chunk
            # index. uuid5 creates a UUID by hashing a namespace + name string.
            # Using NAMESPACE_URL is arbitrary -- what matters is that the
            # same input always produces the same UUID.
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk.metadata['content_hash']}_{chunk.metadata['chunk_index']}",
                )
            )

            # The "payload" in Qdrant is like a JSON column -- it stores
            # arbitrary metadata alongside the vector. At query time, Qdrant
            # returns both the vector similarity score AND the payload, so
            # we can display the chunk text and source information to the user.
            payload = {
                **chunk.metadata,
                "text": chunk.text,
            }

            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches of 100 to avoid overwhelming Qdrant with a
        # single massive request. This is especially important for large
        # document sets (thousands of chunks).
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(
                "Upserted batch %d-%d of %d points",
                i,
                min(i + batch_size, len(points)),
                len(points),
            )

        logger.info(
            "Upserted %d points into collection '%s'",
            len(points),
            self.collection_name,
        )

    def get_collection_info(self) -> dict:
        """
        Retrieve information about the collection (point count, etc.).

        Useful for monitoring and logging after ingestion to verify that
        the expected number of vectors were stored.

        Returns
        -------
        dict
            Collection info including vectors_count, points_count, status, etc.
        """
        info = self.client.get_collection(self.collection_name)

        result = {
            "collection_name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status,
        }

        logger.info("Collection info: %s", result)
        return result
