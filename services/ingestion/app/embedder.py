"""
Embedding Client -- Calls the Text Embeddings Inference (TEI) server to
convert text into vector representations.

=== What are embeddings? ===
An "embedding" is a list of numbers (a "vector") that represents the MEANING
of a piece of text. For example, the sentence "How do I reset my password?"
might become a vector like [0.023, -0.15, 0.87, ...] with 768 numbers.

The key property of embeddings is that texts with SIMILAR MEANING produce
vectors that are CLOSE TOGETHER in vector space. So "How do I reset my
password?" and "I forgot my password, how do I change it?" would produce
vectors that are very close to each other, even though the exact words differ.

This is what makes semantic search possible: instead of matching keywords,
we compare the meaning of the user's question to the meaning of every chunk
in our knowledge base by comparing their vectors.

=== What is the TEI server? ===
TEI (Text Embeddings Inference) is a server from Hugging Face that runs
embedding models efficiently. It:
  - Loads the Nomic Embed Text V1.5 model into memory (on CPU or GPU)
  - Exposes an HTTP API that accepts text and returns vectors
  - Handles batching, quantization, and other optimizations internally

We call it over HTTP so the embedding model runs in its own Kubernetes pod
and can be scaled independently of the ingestion service.

=== Why Nomic Embed Text V1.5? ===
This model was chosen because:
  - It produces 768-dimensional vectors (good balance of quality vs storage)
  - It supports long inputs (up to 8192 tokens per text)
  - It performs well on retrieval benchmarks
  - It's open-source and can run on CPU (no GPU required for embeddings)
  - It's well-suited for RAG workloads

=== About the prefix requirement ===
Nomic Embed Text V1.5 is a "bi-encoder" model that was trained with
task-specific prefixes. You MUST prepend a prefix to every text:

  - "search_document: " -- Use when INDEXING (storing) documents. This tells
    the model to create a vector optimized for being *found* by queries.
  - "search_query: " -- Use when QUERYING (searching). This tells the model
    to create a vector optimized for *finding* relevant documents.

The prefixes ensure that query vectors and document vectors end up in the
right "region" of the vector space for accurate matching. Forgetting the
prefix or using the wrong one will significantly degrade retrieval quality.

In this module (the ingestion service), we always use "search_document: "
because we are indexing documents. The Chat API service uses "search_query: "
when embedding the user's question.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Async HTTP client for the Text Embeddings Inference (TEI) server.

    This client sends text to the TEI server and receives back embedding
    vectors. It handles batching (sending texts in groups) and prefix
    prepending (required by Nomic V1.5).
    """

    def __init__(self, base_url: str, batch_size: int = 32) -> None:
        """
        Parameters
        ----------
        base_url : str
            URL of the TEI server, e.g., "http://embedding:8080".
        batch_size : int
            How many texts to embed in a single HTTP request. Larger batches
            are more efficient (fewer HTTP round-trips, better GPU utilization)
            but use more memory. 32 is a safe default.

            Why batch at all?
            -----------------
            If we have 10,000 chunks to embed, sending them one at a time
            would mean 10,000 HTTP requests -- very slow due to network
            overhead. Sending all 10,000 at once might exceed the server's
            memory. Batching (e.g., 32 at a time) gives us ~312 requests,
            which balances speed and memory usage.
        """
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        # We use a generous timeout because embedding large batches of text
        # can take a while, especially on CPU. 120 seconds prevents premature
        # timeouts for batch sizes of 32+ texts.
        self._timeout = httpx.Timeout(120.0)

    async def embed_texts(
        self,
        texts: list[str],
        prefix: str = "search_document: ",
    ) -> list[list[float]]:
        """
        Embed a list of texts into vectors using the TEI server.

        Parameters
        ----------
        texts : list[str]
            The raw text strings to embed.
        prefix : str
            The Nomic V1.5 task prefix. Defaults to "search_document: " for
            indexing. The Chat API will use "search_query: " when embedding
            user questions.

            IMPORTANT: This prefix is NOT optional. Nomic V1.5 was trained
            with these prefixes, and omitting them will significantly reduce
            retrieval accuracy. Think of it as telling the model "I'm storing
            a document" vs "I'm looking for a document."

        Returns
        -------
        list[list[float]]
            A list of embedding vectors. Each vector is a list of 768 floats.
            The i-th vector corresponds to the i-th input text.

        Raises
        ------
        httpx.HTTPStatusError
            If the TEI server returns an error response.
        """
        # Step 1: Prepend the task prefix to every text.
        # Example: "Reset your password by..." becomes
        #          "search_document: Reset your password by..."
        prefixed_texts = [f"{prefix}{text}" for text in texts]

        all_embeddings: list[list[float]] = []

        # Step 2: Split into batches and send each batch to the TEI server.
        total_batches = (len(prefixed_texts) + self.batch_size - 1) // self.batch_size

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for batch_index in range(0, len(prefixed_texts), self.batch_size):
                batch = prefixed_texts[batch_index : batch_index + self.batch_size]
                batch_number = (batch_index // self.batch_size) + 1

                logger.info(
                    "Embedding batch %d/%d (%d texts)",
                    batch_number,
                    total_batches,
                    len(batch),
                )

                # POST to the TEI /embed endpoint.
                # The TEI server expects a JSON body with an "inputs" field
                # containing a list of strings. It returns a JSON array of
                # arrays (one embedding vector per input text).
                response = await client.post(
                    f"{self.base_url}/embed",
                    json={"inputs": batch},
                )
                # Raise an exception if the server returned an error (4xx/5xx).
                response.raise_for_status()

                # The response is a list of embedding vectors, one per input.
                # Each vector is a list of 768 floats (for Nomic V1.5).
                batch_embeddings = response.json()
                all_embeddings.extend(batch_embeddings)

        logger.info(
            "Embedded %d texts into %d-dimensional vectors",
            len(texts),
            len(all_embeddings[0]) if all_embeddings else 0,
        )

        return all_embeddings
