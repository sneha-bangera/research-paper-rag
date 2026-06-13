"""
Exposes the three main entry points for the ingestion pipeline:

    from ingestion import ingest_arxiv_keyword, ingest_arxiv_id, ingest_uploaded_pdf

Each function runs the full pipeline:
  fetch / receive  →  parse  →  chunk  →  embed + index

Returns a summary dict with paper metadata and chunk stats.
"""

import logging
from typing import Optional

from .arxiv_fetcher import fetch_by_keyword, fetch_by_id
from .pdf_parser import parse_pdf, parse_uploaded_pdf
from .chunker import chunk_document, get_chunk_summary
from .indexer import get_indexer

logger = logging.getLogger(__name__)


def _run_pipeline(parsed_doc: dict) -> dict:
    """Shared chunk → index pipeline used by all three ingest functions."""
    chunks = chunk_document(parsed_doc)
    indexer = get_indexer()
    added = indexer.add_chunks(chunks)
    indexer.save_bm25()
    return {
        "metadata": parsed_doc["metadata"],
        "chunk_summary": get_chunk_summary(chunks),
        "new_chunks_added": added,
    }


def ingest_arxiv_keyword(query: str, max_results: int = 5) -> list[dict]:
    """
    Fetch papers by keyword from ArXiv, parse, chunk, and index them.

    Returns:
        List of result dicts (one per paper), each with metadata + chunk stats.
    """
    papers = fetch_by_keyword(query, max_results=max_results)
    results = []
    for paper in papers:
        try:
            parsed = parse_pdf(paper["pdf_bytes"], metadata=paper)
            result = _run_pipeline(parsed)
            results.append(result)
            logger.info("Ingested: %s", paper["title"])
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", paper.get("title", "?"), exc)
    return results


def ingest_arxiv_id(identifier: str) -> dict:
    """
    Fetch a single paper by ArXiv ID or URL, parse, chunk, and index it.

    Returns:
        Result dict with metadata + chunk stats.
    """
    paper = fetch_by_id(identifier)
    parsed = parse_pdf(paper["pdf_bytes"], metadata=paper)
    return _run_pipeline(parsed)


def ingest_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str = "upload.pdf",
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    year: Optional[int] = None,
) -> dict:
    """
    Parse, chunk, and index a user-uploaded PDF.

    Returns:
        Result dict with metadata + chunk stats.
    """
    parsed = parse_uploaded_pdf(
        pdf_bytes,
        filename=filename,
        title=title,
        authors=authors,
        year=year,
    )
    return _run_pipeline(parsed)