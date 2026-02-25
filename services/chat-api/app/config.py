"""
Configuration for the Chat API service.

We use pydantic-settings to manage configuration. This library lets us define
all our settings as a Python class with type hints and defaults, then
automatically loads overrides from environment variables or a .env file.

Why this matters for our RAG system:
- We need to connect to three separate services (Qdrant, embedding model, vLLM),
  each with their own host/port.
- We have several "tuning knobs" for the RAG pipeline (top_k, temperature, etc.)
  that we want to adjust without changing code.
- pydantic-settings validates types at startup, so a misconfigured port like
  "abc" will fail immediately instead of causing a confusing runtime error.
"""

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Central configuration for the Chat API.

    Every field here can be overridden by setting an environment variable with
    the same name (case-insensitive). For example, setting QDRANT_HOST=my-host
    in the environment will override the default "qdrant" value below.
    """

    # -------------------------------------------------------------------------
    # Qdrant (Vector Database) settings
    # -------------------------------------------------------------------------
    # Qdrant is the vector database where our document chunks are stored as
    # high-dimensional vectors (embeddings). When a user asks a question, we
    # convert it to a vector and search Qdrant for the most similar document
    # chunks. Think of it like a search engine, but instead of matching keywords,
    # it matches *meaning*.
    # -------------------------------------------------------------------------

    qdrant_host: str = "qdrant"
    """Hostname of the Qdrant server. In Kubernetes, this is the service name."""

    qdrant_port: int = 6333
    """Qdrant's gRPC port. 6333 is the default for the REST/gRPC API."""

    qdrant_collection: str = "pdf_chunks"
    """
    Name of the Qdrant "collection" (similar to a database table).
    The ingestion service creates this collection and fills it with document
    chunk vectors. The chat API reads from it at query time.
    """

    # -------------------------------------------------------------------------
    # Embedding Model settings
    # -------------------------------------------------------------------------
    # The embedding model converts text into vectors (lists of numbers) that
    # capture the *meaning* of the text. We use Nomic Embed Text V1.5, served
    # via HuggingFace's Text Embeddings Inference (TEI) server.
    #
    # Key concept: An "embedding" is a fixed-length list of floating-point
    # numbers (a vector) that represents the semantic meaning of a piece of
    # text. Similar texts produce similar vectors. This is what makes semantic
    # search possible -- we can find documents that *mean* similar things to
    # the user's question, even if they use completely different words.
    # -------------------------------------------------------------------------

    embedding_url: str = "http://embedding:8080"
    """URL of the TEI (Text Embeddings Inference) server hosting the embedding model."""

    embedding_dimension: int = 768
    """
    The number of dimensions in each embedding vector.

    Nomic Embed Text V1.5 produces 768-dimensional vectors. This means each
    piece of text is converted into a list of 768 floating-point numbers.
    More dimensions = more nuance in meaning representation, but also more
    memory and slower searches. 768 is a good balance for our use case.
    """

    # -------------------------------------------------------------------------
    # vLLM (Large Language Model) settings
    # -------------------------------------------------------------------------
    # vLLM is the inference server that runs our LLM (Llama 3.1 8B Instruct).
    # It exposes an OpenAI-compatible API, so we can talk to it using the same
    # format that OpenAI's ChatGPT API uses. This is convenient because there
    # are tons of examples and libraries built for that format.
    #
    # vLLM is specifically designed for fast LLM inference. It uses a technique
    # called "PagedAttention" to efficiently manage GPU memory, which lets it
    # handle more concurrent requests than naive approaches.
    # -------------------------------------------------------------------------

    vllm_url: str = "http://vllm:8000"
    """URL of the vLLM inference server."""

    vllm_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    """
    The model identifier that vLLM is serving.

    We use the AWQ INT4 quantized version to save cost. Quantization compresses
    the model by reducing the precision of its weights (from 16-bit floats to
    4-bit integers), which dramatically reduces GPU memory usage with only a
    small quality trade-off. This lets us run on a cheaper g4dn.xlarge GPU
    instance instead of needing a more expensive one.

    For production, consider using the full FP16 (16-bit) version on a
    g6.xlarge for better answer quality.
    """

    # -------------------------------------------------------------------------
    # RAG Pipeline tuning parameters
    # -------------------------------------------------------------------------
    # These are the main "knobs" that control how the RAG pipeline behaves.
    # Adjusting these lets us trade off between answer quality, speed, and cost.
    # -------------------------------------------------------------------------

    rag_top_k: int = 5
    """
    How many document chunks to retrieve from Qdrant for each question.

    "top_k" means "return the K most similar results." More chunks give the LLM
    more context to work with, but also:
    - Use more of the LLM's limited context window (token budget)
    - Can introduce noise if less-relevant chunks are included
    - Increase latency (more text for the LLM to process)

    5 is a reasonable starting point. We can tune this based on answer quality.
    If answers are missing information, try increasing to 8-10.
    If answers include irrelevant information, try decreasing to 3.
    """

    rag_temperature: float = 0.1
    """
    Controls the "creativity" or randomness of the LLM's output.

    Temperature ranges from 0.0 to 2.0:
    - 0.0 = Deterministic (always picks the most likely next word)
    - 0.1 = Nearly deterministic, with tiny variation (our default)
    - 0.7 = Moderately creative (good for general chat)
    - 1.0+ = Highly creative / unpredictable

    For RAG, we want LOW temperature because we want the LLM to stick closely
    to the retrieved context, not make things up. A temperature of 0.1 gives
    us consistent, factual answers while allowing minimal variation so
    responses don't feel robotic.
    """

    rag_max_tokens: int = 1024
    """
    Maximum number of tokens the LLM can generate in its response.

    A "token" is roughly 3/4 of a word in English (e.g., "unhappiness" is
    3 tokens: "un", "happiness" might be split differently by the tokenizer).
    1024 tokens is roughly 750 words, which is plenty for a detailed answer
    with citations.

    Setting this too high wastes GPU time on long-winded answers.
    Setting this too low may cut off answers mid-sentence.
    """

    # -------------------------------------------------------------------------
    # pydantic-settings configuration
    # -------------------------------------------------------------------------

    model_config = {
        "env_file": ".env",
        # This tells pydantic-settings to look for a .env file in the current
        # directory and load any KEY=VALUE pairs from it. Environment variables
        # set directly (e.g., in a Kubernetes deployment) take priority over
        # .env file values.
    }
