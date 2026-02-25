"""
FastAPI application for the PDF RAG Chatbot.

WHAT IS FastAPI?
FastAPI is a modern Python web framework for building APIs. It's built on top
of Starlette (for the web server parts) and Pydantic (for data validation).
Key features that make it great for our RAG chatbot:

1. ASYNC SUPPORT -- FastAPI natively supports Python's async/await pattern.
   This means our API can handle multiple requests concurrently. While one
   request is waiting for the LLM to generate an answer (which takes seconds),
   the server can start processing another request. Without async, each request
   would block the entire server.

2. AUTOMATIC VALIDATION -- FastAPI uses our Pydantic models (ChatRequest,
   ChatResponse) to automatically validate incoming requests and generate
   OpenAPI documentation. If a client sends invalid data, they get a clear
   error message without us writing any validation code.

3. AUTOMATIC DOCS -- FastAPI generates interactive API documentation at
   /docs (Swagger UI) and /redoc. This makes it easy for frontend developers
   to understand and test the API.

WHY A GLOBAL PIPELINE INSTANCE?
We create one RAGPipeline instance when the server starts and reuse it for
all requests. This is important because:
- The RAG pipeline holds HTTP connection pools and a Qdrant client.
  Creating these for every request would be wasteful and slow.
- The pipeline is stateless (it doesn't store per-request data), so sharing
  it across concurrent requests is safe.
- The lifespan context manager ensures proper cleanup when the server shuts down.

WHAT ARE LIFESPAN EVENTS?
FastAPI's lifespan context manager lets us run setup code when the server
starts and cleanup code when it stops. We use this to:
- On startup: Create the RAG pipeline (establish connections to Qdrant, set up
  HTTP clients for embedding and vLLM servers).
- On shutdown: Close HTTP connections gracefully (avoids "connection reset"
  errors in logs and ensures in-flight requests complete).

This pattern is similar to opening/closing a database connection pool in a
traditional web application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import Config
from .models import ChatRequest, ChatResponse
from .rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# We configure logging at the module level so that all log messages across
# the application have a consistent format. The format includes:
# - Timestamp (asctime) -- when the event happened
# - Logger name (name) -- which module produced the message
# - Level (levelname) -- severity (DEBUG, INFO, WARNING, ERROR)
# - Message (message) -- the actual log content
#
# In production on Kubernetes, these logs are collected by the cluster's
# logging stack and can be searched/filtered for debugging.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline instance
# ---------------------------------------------------------------------------
# This variable holds the single RAGPipeline instance that all request handlers
# share. It starts as None and gets initialized during the lifespan startup.
#
# Why global? FastAPI doesn't have a built-in way to pass application-level
# state to route handlers (unlike frameworks with dependency injection containers).
# The global variable pattern is the recommended approach in FastAPI's docs for
# resources that are expensive to create and should be shared across requests.
# ---------------------------------------------------------------------------
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle: startup and shutdown.

    This is an async context manager that FastAPI calls when the server
    starts (everything before 'yield') and when it stops (everything after
    'yield'). Think of it as:

        # Startup (runs once when server starts)
        setup_code()
        yield  # Server is running and handling requests
        # Shutdown (runs once when server stops)
        cleanup_code()

    We use this to:
    1. Create the RAG pipeline with all its connections (startup)
    2. Close HTTP connections cleanly (shutdown)
    """
    global pipeline

    # --- Startup: Initialize the RAG pipeline ---
    config = Config()
    pipeline = RAGPipeline(config)

    logger.info("RAG pipeline initialized")
    logger.info("  Embedding server: %s", config.embedding_url)
    logger.info("  vLLM server: %s", config.vllm_url)
    logger.info("  Qdrant: %s:%d", config.qdrant_host, config.qdrant_port)
    logger.info("  Collection: %s", config.qdrant_collection)
    logger.info("  Top-K: %d, Temperature: %.2f, Max tokens: %d",
                config.rag_top_k, config.rag_temperature, config.rag_max_tokens)

    # 'yield' pauses this function. The server runs and handles requests
    # until it receives a shutdown signal (e.g., SIGTERM from Kubernetes).
    yield

    # --- Shutdown: Clean up resources ---
    # Close the HTTP client to release pooled connections. This is good
    # practice to avoid "ResourceWarning: unclosed" messages and ensure
    # any in-progress requests complete gracefully.
    if pipeline and pipeline.http_client:
        await pipeline.http_client.aclose()
    logger.info("RAG pipeline shut down")


# ---------------------------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------------------------
# The FastAPI instance is the core of our web application. All route handlers
# (endpoints) are registered on this instance.
#
# Parameters:
# - title/description/version: Populate the auto-generated API docs at /docs
# - lifespan: Our startup/shutdown lifecycle manager defined above
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PDF RAG Chatbot",
    description="Ask questions about PDF documents using Retrieval-Augmented Generation (RAG)",
    version="0.1.0",
    lifespan=lifespan,
)


# ===========================================================================
# Web UI -- serve the single-page chat interface at the root URL
# ===========================================================================

@app.get("/")
async def ui():
    """
    Serve the web chat UI.

    This returns a single HTML file that contains all the CSS and JavaScript
    needed for the chat interface. It communicates with the /chat endpoint
    on the same origin, so there are no CORS issues to handle.

    The file lives at app/static/index.html and is included in the Docker
    image automatically (the Dockerfile already copies the entire app/ dir).
    """
    return FileResponse("app/static/index.html")


# ===========================================================================
# Health check endpoint
# ===========================================================================

@app.get("/health")
async def health():
    """
    Simple health check endpoint.

    Kubernetes uses this to determine if the pod is alive and ready to
    receive traffic. The liveness and readiness probes in our K8s manifest
    will hit this endpoint periodically.

    Returns a simple JSON response: {"status": "healthy"}

    In a more sophisticated version, we could check connectivity to Qdrant,
    the embedding server, and vLLM, and return "degraded" if any are down.
    For now, a simple 200 OK is sufficient.
    """
    return {"status": "healthy"}


# ===========================================================================
# Chat endpoint -- the main API
# ===========================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user's question through the RAG pipeline and return an answer.

    This is the main API endpoint. The flow is:
    1. Client sends a POST request with a question (and optional history)
    2. We pass it through the RAG pipeline (embed -> search -> prompt -> generate)
    3. We return the answer with source citations

    The response_model=ChatResponse parameter tells FastAPI to:
    - Validate the response matches the ChatResponse schema
    - Generate accurate API documentation
    - Serialize the response to JSON automatically

    Error handling:
    - 503 if the RAG pipeline isn't initialized (server still starting up)
    - 500 if anything goes wrong during processing (embedding server down,
      Qdrant unreachable, vLLM error, etc.)
    """
    # --- Guard: Ensure the pipeline is ready ---
    # The pipeline might be None if the server hasn't finished starting up,
    # or if initialization failed. Return 503 (Service Unavailable) to tell
    # the client to retry later.
    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. The server may still be starting up.",
        )

    try:
        # --- Execute the full RAG pipeline ---
        # This is where the magic happens. The pipeline:
        # 1. Embeds the question (converts text to a vector)
        # 2. Searches Qdrant for similar document chunks
        # 3. Builds a prompt with the retrieved context
        # 4. Sends the prompt to the LLM for answer generation
        # 5. Returns the answer and source citations
        answer, sources = await pipeline.query(
            question=request.question,
            history=request.history,
        )

        # --- Build and return the response ---
        return ChatResponse(
            answer=answer,
            sources=sources,
            model=pipeline.config.vllm_model,
            chunks_retrieved=len(sources),
        )

    except Exception as e:
        # --- Error handling ---
        # We catch all exceptions here to return a clean 500 error to the
        # client instead of an ugly stack trace. The exc_info=True parameter
        # includes the full traceback in our server logs for debugging.
        #
        # Common errors that might end up here:
        # - httpx.ConnectError: Embedding server or vLLM is unreachable
        # - httpx.HTTPStatusError: Embedding server or vLLM returned an error
        # - qdrant_client exceptions: Qdrant is down or collection doesn't exist
        logger.error("Error processing chat request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
