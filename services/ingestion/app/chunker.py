"""
Text Chunking Module -- Splits documents into fixed-size token windows.

=== Why do we chunk documents? ===
Large Language Models (LLMs) have a limited "context window" -- the maximum
amount of text they can process at once. For Llama 3.1 8B, this is 128K
tokens, but in practice we want to send only the MOST RELEVANT snippets,
not entire documents.

By splitting documents into small chunks (~512 tokens each), we can:
  1. Store each chunk as its own vector in the vector database.
  2. At query time, retrieve ONLY the 5-10 most relevant chunks.
  3. Send just those chunks to the LLM, keeping the prompt focused and the
     answer grounded in specific, relevant content.

This is the core idea behind RAG (Retrieval-Augmented Generation): instead
of stuffing the entire knowledge base into the prompt, we retrieve just
the pieces we need.

=== Why 512 tokens? ===
This is a balance between two competing goals:
  - TOO SMALL (e.g., 128 tokens): Each chunk lacks enough context to be
    meaningful on its own. The LLM might get snippets that are too fragmented
    to generate a good answer.
  - TOO LARGE (e.g., 2048 tokens): Each chunk covers too many topics, so
    the retriever might pull in chunks that are only partially relevant,
    diluting the useful content with noise.
  - 512 tokens is a widely-used starting point that works well for most
    knowledge-base content. It can be tuned later based on retrieval quality.

=== Why overlapping chunks? ===
Imagine a sentence that explains a critical concept, but it happens to fall
right at the boundary between two chunks. Without overlap, that sentence
would be split in half -- chunk A gets the first half, chunk B gets the
second half, and NEITHER chunk contains the full concept.

Overlapping chunks solve this by repeating some tokens at the boundary.
With 50-token overlap, the last 50 tokens of chunk A are also the first
50 tokens of chunk B. This ensures that concepts near boundaries appear
fully in at least one chunk.

=== What are tokens? ===
Tokens are NOT the same as words or characters. A "token" is the smallest
unit that the AI model processes. Common words like "the" are one token,
but longer or rarer words get split into multiple tokens. For example:
  - "hello" = 1 token
  - "unbelievable" = 3 tokens ("un", "believ", "able")
  - "2024-01-15" = 4 tokens

We count TOKENS (not characters or words) because:
  1. The embedding model processes tokens, so chunk sizes in tokens give us
     precise control over what the model sees.
  2. LLM context windows are measured in tokens.
  3. Two texts with the same character count can have very different token
     counts depending on the words used.

=== What is cl100k_base? ===
cl100k_base is the tokenizer (also called "encoding") used by OpenAI's
newer models. We use it here because:
  - tiktoken (the library) is fast and widely available.
  - cl100k_base handles English text, code, and special characters well.
  - The exact tokenizer choice matters less than being CONSISTENT -- we
    must use the same tokenizer for chunking and for counting tokens in
    prompts later. cl100k_base is a solid general-purpose default.

This module is data-source agnostic -- it works the same regardless of whether
the input comes from PDFs, APIs, or other sources.
"""

import logging
from dataclasses import dataclass

import tiktoken

from app.pdf_parser import Document

logger = logging.getLogger(__name__)

# Load the tokenizer once at module level. This is an expensive operation
# (it loads a large vocabulary file), so we do it once and reuse it.
# cl100k_base is the encoding used by GPT-4 and many modern models.
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """
    Represents a single chunk of text ready to be embedded and stored.

    The metadata dict carries provenance information so that when the chatbot
    retrieves this chunk at query time, it can tell the user exactly where
    the information came from (which PDF, which page).

    Fields in metadata:
      - page_number: Which page of the source document this chunk came from.
      - source: The filename of the source document.
      - content_hash: Hash of the original page content (for deduplication).
      - chunk_index: Position of this chunk within its parent document's
        chunks (0-indexed). Used with content_hash to generate deterministic
        IDs in the vector store.
    """

    text: str
    metadata: dict


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split a list of Documents into overlapping token-window chunks.

    Parameters
    ----------
    documents : list[Document]
        Documents to chunk (typically from pdf_parser.parse_pdf).
    chunk_size : int
        Number of tokens per chunk. Default 512.
    chunk_overlap : int
        Number of tokens that overlap between consecutive chunks. Default 50.

    Returns
    -------
    list[Chunk]
        All chunks from all documents, ready for embedding.

    Algorithm (sliding window over tokens)
    ---------------------------------------
    For each document:
      1. Encode the full text into a list of token IDs using tiktoken.
         Example: "Hello world" -> [9906, 1917]
      2. Slide a window of `chunk_size` tokens across the token list,
         advancing by `step = chunk_size - chunk_overlap` tokens each time.
         This is what creates the overlap between consecutive chunks.
      3. Decode each window of token IDs back into a text string.
      4. Wrap the text in a Chunk dataclass with metadata from the parent
         Document.

    Visual example (chunk_size=10, overlap=3, step=7):

      Tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      Chunk 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      Chunk 2:                   [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                                  ^^^^^^^^^
                                  overlap!

      Tokens 7, 8, 9 appear in BOTH chunks, so any concept spanning that
      boundary is fully captured in at least one chunk.
    """
    # The step size determines how far the window moves between chunks.
    # step = chunk_size - chunk_overlap ensures the desired overlap.
    step = chunk_size - chunk_overlap

    all_chunks: list[Chunk] = []

    for document in documents:
        # --- Step 1: Encode text into token IDs ---
        # This converts the human-readable text into a list of integer IDs
        # that represent the model's vocabulary. We operate on tokens (not
        # characters) because that's what the embedding model actually
        # processes.
        token_ids = _ENCODING.encode(document.content)

        # If the document is shorter than one chunk, it becomes a single
        # chunk (no splitting needed).
        if len(token_ids) == 0:
            continue

        chunk_index = 0

        # --- Step 2: Slide the window across the token list ---
        for start in range(0, len(token_ids), step):
            # Extract a window of chunk_size tokens.
            window = token_ids[start : start + chunk_size]

            # --- Step 3: Decode token IDs back to text ---
            # This reverses the encoding, turning [9906, 1917] back into
            # "Hello world". The decoded text is what we'll embed and store.
            chunk_text = _ENCODING.decode(window)

            # Skip chunks that are too small to be useful. A chunk with
            # fewer than 20 tokens is likely just a heading or whitespace
            # and won't provide meaningful content for retrieval.
            if len(window) < 20:
                logger.debug(
                    "Skipping tiny chunk (%d tokens) from %s page %d",
                    len(window),
                    document.source,
                    document.page_number,
                )
                continue

            # --- Step 4: Create the Chunk with metadata ---
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    "page_number": document.page_number,
                    "source": document.source,
                    "content_hash": document.content_hash,
                    # chunk_index is used together with content_hash to
                    # create deterministic UUIDs in the vector store.
                    # This makes ingestion idempotent -- re-running the
                    # pipeline overwrites existing vectors instead of
                    # creating duplicates.
                    "chunk_index": chunk_index,
                },
            )
            all_chunks.append(chunk)
            chunk_index += 1

            # If this window reached the end of the token list, no need
            # to continue sliding.
            if start + chunk_size >= len(token_ids):
                break

        logger.debug(
            "Document '%s' page %d: %d tokens -> %d chunks",
            document.source,
            document.page_number,
            len(token_ids),
            chunk_index,
        )

    logger.info(
        "Chunked %d documents into %d total chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )

    return all_chunks
