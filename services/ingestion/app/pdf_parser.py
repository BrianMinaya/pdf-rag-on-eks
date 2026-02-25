"""
PDF Parsing Module -- Extracts text from PDFs as structured documents.

=== Swappable data source ===
This module can be replaced with any data source fetcher (REST API, wiki, CMS,
web scraper) -- the rest of the pipeline (chunking, embedding, vector storage)
is identical regardless of where the raw text comes from.

=== Why Markdown? ===
We convert PDFs to Markdown (not plain text) because Markdown preserves
structural information -- headings, lists, tables, bold/italic emphasis.
When this structured text is later chunked and fed to the LLM as context,
the model can better understand the document's organization, leading to
higher-quality answers.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a single page extracted from a PDF.

    In a RAG pipeline, we need to track where each piece of text came from
    so we can cite sources in the chatbot's answers. That's why we store
    page_number and source alongside the content.

    content_hash is a SHA-256 hash of the page text. This enables an
    important optimization:

        INCREMENTAL UPDATES (skip-unchanged logic)
        -------------------------------------------
        When re-ingesting documents, we can compare content hashes to detect
        which pages have changed since the last run. Unchanged pages can be
        skipped entirely, saving embedding compute time and API calls.

        For large document collections (thousands of pages), only a few
        typically change between ingestion runs. Hashing lets us re-ingest
        in minutes instead of hours.
    """

    content: str
    page_number: int
    source: str
    content_hash: str = field(default="")

    def __post_init__(self):
        """Compute content_hash automatically if not provided."""
        if not self.content_hash:
            # SHA-256 produces a unique fingerprint of the text content.
            # Even a single character change produces a completely different
            # hash, so we can reliably detect modifications.
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()


def parse_pdf(pdf_path: str) -> list[Document]:
    """
    Extract text from a PDF file and return a list of Document objects,
    one per page.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to the PDF file.

    Returns
    -------
    list[Document]
        One Document per non-empty page, with content in Markdown format.

    How it works
    ------------
    1. pymupdf4llm.to_markdown() reads the PDF and converts each page to
       Markdown. This library is specifically designed for LLM use cases --
       it handles multi-column layouts, tables, and embedded images better
       than generic PDF-to-text tools.

    2. The result is a list of dicts, one per page, each with a "text" key.

    3. We wrap each page in a Document dataclass, computing the content hash
       for deduplication / incremental update support.

    4. Empty pages (e.g., blank separator pages) are skipped since they
       would produce meaningless chunks and waste embedding compute.
    """
    pdf_filename = Path(pdf_path).name
    logger.info("Parsing PDF: %s", pdf_filename)

    # pymupdf4llm.to_markdown() returns a list of dicts when page_chunks=True.
    # Each dict has: {"text": "...", "metadata": {...}}
    # We use page_chunks=True to get per-page output so we can track page
    # numbers for citation purposes.
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    documents: list[Document] = []

    for page_index, page_data in enumerate(pages):
        # Extract the text content from the page dict.
        # pymupdf4llm returns each page as a dict with a "text" key.
        page_text = page_data.get("text", "").strip()

        # Skip empty pages -- they would produce empty chunks that waste
        # embedding compute and add noise to search results.
        if not page_text:
            logger.debug(
                "Skipping empty page %d in %s", page_index + 1, pdf_filename
            )
            continue

        document = Document(
            content=page_text,
            # page_number is 1-indexed (human-friendly) for citation display.
            page_number=page_index + 1,
            # source is just the filename, not the full path, so citations
            # in chat responses are clean and readable.
            source=pdf_filename,
            # content_hash is computed automatically in __post_init__
        )
        documents.append(document)

    logger.info(
        "Extracted %d non-empty pages from %s (total pages in PDF: %d)",
        len(documents),
        pdf_filename,
        len(pages),
    )

    return documents
