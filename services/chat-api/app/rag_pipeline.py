"""
RAG Pipeline -- the heart of the Chat API.

RAG stands for "Retrieval-Augmented Generation." It is a technique that
improves LLM answers by first *retrieving* relevant documents and then
*augmenting* the LLM's input prompt with that context before *generating*
an answer. This grounds the LLM's response in actual data rather than
relying solely on its training knowledge (which may be outdated or wrong).

The RAG pipeline in this file follows these steps for every user question:

    1. EMBED -- Convert the user's question into a vector (a list of numbers
       that captures the question's meaning) using an embedding model.

    2. SEARCH -- Find the most similar document chunks in Qdrant by comparing
       the question vector against all stored document vectors. This is called
       "semantic search" because it matches meaning, not keywords.

    3. BUILD PROMPT -- Assemble a prompt that includes the retrieved chunks
       as context, any conversation history, and the user's question. This
       prompt tells the LLM: "Here is relevant information. Use it to answer
       this question."

    4. GENERATE -- Send the prompt to the LLM (via vLLM) and get the answer.

    5. RETURN -- Package up the answer and source citations for the API response.

This is a simple but effective pipeline. More advanced versions might add
re-ranking (scoring chunks a second time for relevance), query expansion
(reformulating the question), or hybrid search (combining semantic + keyword
search). We keep it simple for now and can add those later if needed.
"""

import logging
import time

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from .config import Config
from .models import Source

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    This class holds connections to all three backend services:
    - The embedding model (via TEI / Text Embeddings Inference server)
    - The vector database (Qdrant)
    - The LLM (via vLLM's OpenAI-compatible API)

    It coordinates them in sequence to answer each user question.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the RAG pipeline with connections to all backend services.

        Args:
            config: Application configuration containing service URLs, model
                    settings, and RAG tuning parameters.
        """
        self.config = config

        # -----------------------------------------------------------------
        # HTTP client for calling the embedding model and vLLM.
        #
        # We use httpx (an async HTTP library) instead of the standard
        # library's urllib because:
        # 1. It supports async/await, so our API can handle multiple
        #    concurrent requests without blocking.
        # 2. It reuses TCP connections (connection pooling), which is faster
        #    than opening a new connection for every request.
        #
        # The 120-second timeout is generous because LLM generation can be
        # slow, especially for long answers or when the GPU is under load.
        # -----------------------------------------------------------------
        self.http_client = httpx.AsyncClient(timeout=120.0)

        # -----------------------------------------------------------------
        # Qdrant client for vector similarity search.
        #
        # We use Qdrant's synchronous Python client here. Even though our
        # FastAPI app is async, the Qdrant search operation is fast enough
        # (typically <50ms) that running it synchronously won't cause
        # noticeable blocking. If we ever see performance issues with many
        # concurrent requests, we can switch to the async client or run
        # the sync client in a thread pool.
        # -----------------------------------------------------------------
        self.qdrant_client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
        )

        logger.info(
            "RAG pipeline created: embedding=%s, vllm=%s, qdrant=%s:%d, collection=%s",
            config.embedding_url,
            config.vllm_url,
            config.qdrant_host,
            config.qdrant_port,
            config.qdrant_collection,
        )

    # =====================================================================
    # Step 1: EMBED the question
    # =====================================================================

    async def embed_question(self, question: str) -> list[float]:
        """
        Convert a natural-language question into a vector (embedding).

        HOW EMBEDDINGS WORK:
        An embedding model reads text and outputs a fixed-length list of
        floating-point numbers (a "vector") that captures the *meaning* of
        that text in a high-dimensional space. Texts with similar meanings
        produce vectors that are close together in that space.

        For example, "What is the return policy?" and "How do I return an
        item?" would produce very similar vectors, even though they use
        different words. This is what makes semantic search possible.

        IMPORTANT -- QUERY vs DOCUMENT PREFIXES:
        Nomic Embed Text V1.5 uses different prefixes to distinguish between
        queries and documents:

        - "search_query: " -- Used here, when embedding a *question* that
          will be used to search for relevant documents.
        - "search_document: " -- Used by the ingestion service, when
          embedding a *document chunk* that will be stored in the database.

        These prefixes help the model produce better-aligned vectors for
        search. The query embedding is optimized to "point toward" relevant
        document embeddings. Using the wrong prefix would degrade search
        quality. This is a specific feature of Nomic's model -- not all
        embedding models use prefixes.

        Args:
            question: The user's natural-language question.

        Returns:
            A list of 768 floating-point numbers representing the question's
            meaning in vector space.
        """
        start = time.monotonic()

        # -----------------------------------------------------------------
        # Call the TEI (Text Embeddings Inference) server.
        #
        # TEI is HuggingFace's optimized server for running embedding models.
        # It provides a simple REST API: send text in, get vectors out.
        #
        # The /embed endpoint accepts a batch of texts and returns a batch
        # of vectors. We send a single-item list because we only need to
        # embed one question at a time.
        # -----------------------------------------------------------------
        response = await self.http_client.post(
            f"{self.config.embedding_url}/embed",
            json={
                # NOTE: "search_query: " prefix is critical here. It tells
                # the embedding model that this text is a search query, not
                # a document. See the docstring above for why this matters.
                "inputs": [f"search_query: {question}"],
            },
        )
        response.raise_for_status()

        # The response is a list of embeddings (one per input text).
        # Since we sent one question, we take the first (and only) embedding.
        embeddings = response.json()
        embedding_vector = embeddings[0]

        elapsed = time.monotonic() - start
        logger.info(
            "Embedded question in %.3fs (vector dimension: %d)",
            elapsed,
            len(embedding_vector),
        )

        return embedding_vector

    # =====================================================================
    # Step 2: SEARCH for similar document chunks
    # =====================================================================

    def search_qdrant(self, query_embedding: list[float], top_k: int) -> list[ScoredPoint]:
        """
        Search the vector database for document chunks similar to the question.

        WHAT IS SEMANTIC SEARCH?
        Traditional keyword search looks for exact word matches (e.g., searching
        for "refund" only finds documents containing the word "refund"). Semantic
        search instead compares the *meaning* of the query against the *meaning*
        of stored documents by comparing their vector representations.

        This means a question like "Can I get my money back?" will find chunks
        about refund policies, even if those chunks never use the phrase
        "money back."

        HOW COSINE SIMILARITY WORKS:
        Qdrant uses "cosine similarity" to measure how similar two vectors are.
        Imagine each vector as an arrow pointing in some direction in a
        768-dimensional space (hard to visualize, but mathematically straightforward).
        Cosine similarity measures the angle between two arrows:
        - If they point in the same direction: similarity = 1.0 (identical meaning)
        - If they're perpendicular: similarity = 0.0 (unrelated)
        - If they point in opposite directions: similarity = -1.0 (opposite meaning)

        In practice, most text embeddings fall in the 0.3-0.9 range.

        WHAT IS top_k?
        top_k controls how many results to return. We ask Qdrant: "Give me
        the K most similar chunks." A higher K gives the LLM more context to
        work with, but also uses more of the LLM's token budget and may
        include less relevant chunks. Our default is 5, which balances
        thoroughness with precision.

        Args:
            query_embedding: The vector representation of the user's question
                             (produced by embed_question).
            top_k: How many of the most similar chunks to return.

        Returns:
            A list of ScoredPoint objects from Qdrant, each containing:
            - .payload: A dict with the chunk text and metadata (page number,
              source filename, etc.)
            - .score: The cosine similarity score (0 to 1)
        """
        start = time.monotonic()

        # -----------------------------------------------------------------
        # Perform the vector similarity search.
        #
        # Under the hood, Qdrant uses an approximate nearest neighbor (ANN)
        # algorithm called HNSW (Hierarchical Navigable Small World). This
        # is much faster than comparing against every single vector in the
        # database (exact search), at the cost of occasionally missing the
        # absolute best match. For our use case with thousands of chunks,
        # the speed vs. accuracy trade-off is well worth it.
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # NOTE: qdrant-client >= 1.12 replaced .search() with .query_points().
        # The old .search() method was removed in 1.16. The new API returns
        # a QueryResponse object with a .points attribute instead of a plain
        # list. The individual point objects still have .payload and .score.
        # -----------------------------------------------------------------
        response = self.qdrant_client.query_points(
            collection_name=self.config.qdrant_collection,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        results = response.points

        elapsed = time.monotonic() - start
        logger.info(
            "Qdrant search returned %d results in %.3fs (top_k=%d)",
            len(results),
            elapsed,
            top_k,
        )

        # Log the similarity scores so we can monitor retrieval quality.
        # If scores are consistently low (< 0.5), it may indicate:
        # - The question is about a topic not covered in the documents
        # - The embedding model isn't performing well on this domain
        # - The chunking strategy needs adjustment
        for i, result in enumerate(results):
            logger.debug(
                "  Result %d: score=%.4f, page=%s, source=%s",
                i + 1,
                result.score,
                result.payload.get("page_number", "?"),
                result.payload.get("source", "?"),
            )

        return results

    # =====================================================================
    # Step 3: BUILD the prompt with retrieved context
    # =====================================================================

    def build_prompt(
        self,
        question: str,
        context_chunks: list[ScoredPoint],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Assemble the prompt that will be sent to the LLM.

        WHAT IS PROMPT ENGINEERING?
        Prompt engineering is the practice of carefully crafting the input to
        an LLM to get the best possible output. For RAG, this means:

        1. SYSTEM MESSAGE -- Sets the LLM's "personality" and instructions.
           We tell it to answer based on the provided context and cite sources.
           This is crucial because without explicit instructions, the LLM might
           ignore the context and answer from its training data (which could
           be outdated or wrong).

        2. CONTEXT -- The retrieved document chunks, formatted so the LLM can
           easily reference them. We number each chunk and include the page
           number so the LLM can cite specific sources in its answer.

        3. CONVERSATION HISTORY -- Previous messages in the conversation, so
           the LLM understands follow-up questions. Without this, the LLM
           would treat every question as independent.

        4. USER MESSAGE -- The actual question, clearly labeled and positioned
           after the context so the LLM knows what to answer.

        WHY WE INCLUDE CONTEXT IN THE PROMPT:
        LLMs have a fixed "knowledge cutoff" -- they only know what was in
        their training data. By including relevant document chunks in the prompt,
        we effectively "teach" the LLM new information at inference time.
        This is the "Augmented" part of Retrieval-Augmented Generation.

        WHY WE ASK FOR CITATIONS:
        Explicitly asking the LLM to cite page numbers serves two purposes:
        1. It helps users verify the answer against the source documents.
        2. It actually improves answer quality -- when the LLM is asked to
           cite sources, it tends to stay more faithful to the provided context
           rather than hallucinating (making things up).

        Args:
            question: The user's question.
            context_chunks: The search results from Qdrant (document chunks
                            with metadata and similarity scores).
            history: Optional conversation history in OpenAI message format.

        Returns:
            A list of message dicts in OpenAI chat format, ready to send
            to the vLLM server.
        """
        # -----------------------------------------------------------------
        # Build the messages list in OpenAI chat completion format.
        #
        # The format is a list of dicts, each with "role" and "content":
        # - "system": Instructions for the LLM (sets behavior and constraints)
        # - "user": Messages from the human user
        # - "assistant": Previous responses from the LLM (for multi-turn)
        #
        # This format was popularized by OpenAI's ChatGPT API and has become
        # an industry standard. vLLM implements this same format, which makes
        # it easy to switch between different LLM providers.
        # -----------------------------------------------------------------
        messages = []

        # --- System message: Set the LLM's behavior and constraints ---
        # The system message is the most important part of prompt engineering.
        # It tells the LLM what it should and shouldn't do. Key instructions:
        # - Answer based on the provided context (not training knowledge)
        # - Cite page numbers (for verifiability)
        # - Admit when it doesn't know (reduces hallucination)
        system_prompt = (
            "You are a helpful assistant that answers questions based on the "
            "provided context. Always cite which page numbers your answer comes "
            "from. If the context doesn't contain enough information to answer "
            "the question, say so honestly. Do not make up information that is "
            "not supported by the context."
        )
        messages.append({"role": "system", "content": system_prompt})

        # --- Conversation history (if provided) ---
        # Including previous messages allows the LLM to understand follow-up
        # questions. For example, if the user first asks "What is the refund
        # policy?" and then asks "Does that apply to international orders?",
        # the history tells the LLM what "that" refers to.
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # --- Format the retrieved context chunks ---
        # We format each chunk with a number and page reference so the LLM
        # can easily cite them in its answer. Example:
        #   [1] (Page 5): "The refund policy states that..."
        #   [2] (Page 12): "International orders are subject to..."
        #
        # Numbering the chunks helps the LLM organize its citations and
        # makes it clear which pieces of information come from where.
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            page_num = chunk.payload.get("page_number", "unknown")
            chunk_text = chunk.payload.get("text", "")
            context_parts.append(f"[{i}] (Page {page_num}): {chunk_text}")

        context_text = "\n\n".join(context_parts)

        # --- User message: Context + Question ---
        # We combine the context and question into a single user message.
        # The format clearly separates the context (what the LLM should base
        # its answer on) from the question (what the LLM should answer).
        user_message = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above. Cite page numbers."
        )
        messages.append({"role": "user", "content": user_message})

        logger.info(
            "Built prompt with %d context chunks and %d history messages",
            len(context_chunks),
            len(history) if history else 0,
        )

        return messages

    # =====================================================================
    # Step 4: GENERATE the answer using the LLM
    # =====================================================================

    async def generate_answer(self, messages: list[dict]) -> str:
        """
        Send the assembled prompt to the LLM and get the generated answer.

        WHAT IS vLLM?
        vLLM is a high-performance inference server for large language models.
        It's specifically designed to serve LLMs efficiently using techniques
        like PagedAttention (which manages GPU memory like an operating system
        manages RAM with virtual memory pages). This lets it serve more
        concurrent requests than loading the model directly in Python.

        WHY OpenAI-COMPATIBLE API?
        vLLM exposes an API that follows the same format as OpenAI's Chat
        Completions API. This is advantageous because:
        1. Tons of existing documentation and examples
        2. Easy to swap in a different LLM provider later (OpenAI, Anthropic,
           another vLLM instance) without changing our code
        3. Standard request/response format that developers can learn once

        WHAT DOES TEMPERATURE CONTROL?
        When an LLM generates text, it predicts the probability of each
        possible next word (token). Temperature controls how those
        probabilities are used:
        - Low temperature (0.0-0.3): The model almost always picks the
          highest-probability token. Outputs are consistent and factual.
          Best for RAG where we want accurate, grounded answers.
        - High temperature (0.7-1.0): The model is more willing to pick
          lower-probability tokens. Outputs are more creative and varied.
          Better for creative writing or brainstorming.

        Args:
            messages: The prompt in OpenAI chat format (system + history +
                      context + question).

        Returns:
            The LLM's generated answer as a string.
        """
        start = time.monotonic()

        # -----------------------------------------------------------------
        # Call vLLM's /v1/chat/completions endpoint.
        #
        # This endpoint follows the OpenAI Chat Completions API spec:
        # https://platform.openai.com/docs/api-reference/chat
        #
        # Key parameters:
        # - model: Must match the model that vLLM is actually serving.
        #   If this doesn't match, vLLM will return an error.
        # - messages: Our carefully constructed prompt with context.
        # - temperature: Controls randomness (low = more deterministic).
        # - max_tokens: Caps the response length to prevent runaway generation.
        # -----------------------------------------------------------------
        response = await self.http_client.post(
            f"{self.config.vllm_url}/v1/chat/completions",
            json={
                "model": self.config.vllm_model,
                "messages": messages,
                "temperature": self.config.rag_temperature,
                "max_tokens": self.config.rag_max_tokens,
            },
        )
        response.raise_for_status()

        # -----------------------------------------------------------------
        # Parse the response.
        #
        # The OpenAI-compatible response format looks like:
        # {
        #   "choices": [
        #     {
        #       "message": {
        #         "role": "assistant",
        #         "content": "The answer is..."
        #       },
        #       "finish_reason": "stop"
        #     }
        #   ],
        #   "usage": {
        #     "prompt_tokens": 512,
        #     "completion_tokens": 128,
        #     "total_tokens": 640
        #   }
        # }
        #
        # We extract the assistant's message content from choices[0].
        # -----------------------------------------------------------------
        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        # Log token usage for monitoring costs and performance.
        # "Prompt tokens" = how many tokens the input used (our context + question).
        # "Completion tokens" = how many tokens the LLM generated (the answer).
        # More tokens = more GPU time = slower and more expensive.
        usage = result.get("usage", {})
        elapsed = time.monotonic() - start
        logger.info(
            "LLM generated answer in %.3fs (prompt_tokens=%s, completion_tokens=%s, total_tokens=%s)",
            elapsed,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            usage.get("total_tokens", "?"),
        )

        return answer

    # =====================================================================
    # Step 5: Full RAG pipeline orchestration
    # =====================================================================

    async def query(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> tuple[str, list[Source]]:
        """
        Execute the full RAG pipeline: embed -> search -> prompt -> generate.

        This is the main entry point that the API endpoint calls. It
        orchestrates all four steps of the RAG pipeline in sequence and
        returns the answer along with source citations.

        THE FULL RAG FLOW:
        1. EMBED: "What does this question mean?" -> Convert to a vector
        2. SEARCH: "Which documents are about this?" -> Find similar chunks
        3. PROMPT: "Here's what I found, now answer this" -> Build the LLM input
        4. GENERATE: "Based on these documents, the answer is..." -> Get the answer
        5. PACKAGE: Bundle the answer with source citations for the API response

        Each step depends on the previous one, so they run sequentially.
        We log timing for each step to identify bottlenecks. Typical latencies:
        - Embedding: 10-50ms (fast, CPU-based model)
        - Qdrant search: 5-50ms (very fast, optimized ANN algorithm)
        - Prompt building: <1ms (just string formatting)
        - LLM generation: 1-10 seconds (the slowest step, GPU-bound)

        Args:
            question: The user's natural-language question.
            history: Optional conversation history for multi-turn context.

        Returns:
            A tuple of (answer_text, list_of_sources) where:
            - answer_text is the LLM's generated response
            - list_of_sources contains the retrieved chunks with metadata
        """
        total_start = time.monotonic()
        logger.info("Starting RAG pipeline for question: %s", question[:100])

        # --- Step 1: Embed the question ---
        # Convert the user's question into a vector so we can search for
        # similar document chunks in the vector database.
        query_embedding = await self.embed_question(question)

        # --- Step 2: Search for similar chunks ---
        # Find the top_k most similar document chunks in Qdrant.
        # These chunks will become the "context" that grounds the LLM's answer.
        search_results = self.search_qdrant(
            query_embedding=query_embedding,
            top_k=self.config.rag_top_k,
        )

        # --- Step 3: Build the prompt ---
        # Assemble the system instructions, conversation history, retrieved
        # context, and the question into a structured prompt for the LLM.
        messages = self.build_prompt(
            question=question,
            context_chunks=search_results,
            history=history,
        )

        # --- Step 4: Generate the answer ---
        # Send the prompt to the LLM and get the generated answer.
        # This is typically the slowest step (1-10 seconds depending on
        # answer length and GPU load).
        answer = await self.generate_answer(messages)

        # --- Step 5: Package source citations ---
        # Convert the Qdrant search results into Source objects that the
        # API can return to the client. This gives users transparency into
        # which documents the answer is based on.
        sources = []
        for result in search_results:
            sources.append(
                Source(
                    text=result.payload.get("text", ""),
                    page_number=result.payload.get("page_number", 0),
                    source=result.payload.get("source", "unknown"),
                    score=round(result.score, 4),
                )
            )

        total_elapsed = time.monotonic() - total_start
        logger.info(
            "RAG pipeline completed in %.3fs: %d sources, answer length=%d chars",
            total_elapsed,
            len(sources),
            len(answer),
        )

        return answer, sources
