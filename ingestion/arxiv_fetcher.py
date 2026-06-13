"""
Fetches papers from ArXiv by keyword search or direct paper ID/URL.
Returns metadata dict + raw PDF bytes ready for the parser.
"""

import re
import time
import logging
import requests
import arxiv

logger = logging.getLogger(__name__)


def _extract_arxiv_id(identifier: str) -> str:
    """
    Normalise an ArXiv identifier.
    Accepts:
      - bare ID:         2005.11401  or  2005.11401v3
      - abs URL:         https://arxiv.org/abs/2005.11401
      - pdf URL:         https://arxiv.org/pdf/2005.11401
    """
    # Strip trailing whitespace / version suffix for lookup
    identifier = identifier.strip()
    # URL patterns
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+(?:v\d+)?)", identifier)
    if match:
        return match.group(1)
    # Bare ID pattern
    match = re.match(r"^([0-9]{4}\.[0-9]+(?:v\d+)?)$", identifier)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot parse ArXiv ID from: {identifier!r}")


def _download_pdf(pdf_url: str, retries: int = 3) -> bytes:
    """Download PDF bytes from a URL with retry logic."""
    headers = {"User-Agent": "ResearchPaperRAG/1.0 (educational project)"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(pdf_url, headers=headers, timeout=30)
            resp.raise_for_status()
            if "application/pdf" not in resp.headers.get("Content-Type", ""):
                logger.warning("Response may not be a PDF (Content-Type: %s)", resp.headers.get("Content-Type"))
            return resp.content
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, pdf_url, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download PDF from {pdf_url} after {retries} attempts")


def _result_to_metadata(result: arxiv.Result) -> dict:
    """Convert an arxiv.Result object to our standard metadata dict."""
    return {
        "arxiv_id": result.entry_id.split("/")[-1],  # e.g. 2005.11401v3
        "title": result.title.strip(),
        "authors": [str(a) for a in result.authors],
        "year": result.published.year,
        "abstract": result.summary.strip(),
        "categories": result.categories,
        "pdf_url": result.pdf_url,
        "source": "arxiv",
    }


# Public API

def fetch_by_keyword(query: str, max_results: int = 5) -> list[dict]:
    """
    Search ArXiv by keyword and return a list of paper dicts.
    Each dict has keys: arxiv_id, title, authors, year, abstract,
                        categories, pdf_url, source, pdf_bytes.

    Args:
        query:       Free-text search query (e.g. "retrieval augmented generation")
        max_results: Maximum number of papers to return (default 5)

    Returns:
        List of paper metadata dicts with pdf_bytes populated.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        meta = _result_to_metadata(result)
        logger.info("Downloading PDF for: %s", meta["title"])
        try:
            meta["pdf_bytes"] = _download_pdf(meta["pdf_url"])
            papers.append(meta)
        except RuntimeError as exc:
            logger.error("Skipping %s — %s", meta["arxiv_id"], exc)

    logger.info("Fetched %d papers for query: %r", len(papers), query)
    return papers


def fetch_by_id(identifier: str) -> dict:
    """
    Fetch a single paper by its ArXiv ID or URL.

    Args:
        identifier: ArXiv ID (e.g. "2005.11401") or URL

    Returns:
        Paper metadata dict with pdf_bytes populated.
    """
    arxiv_id = _extract_arxiv_id(identifier)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        raise ValueError(f"No paper found for ArXiv ID: {arxiv_id}")

    meta = _result_to_metadata(results[0])
    logger.info("Downloading PDF for: %s", meta["title"])
    meta["pdf_bytes"] = _download_pdf(meta["pdf_url"])
    return meta


# Quick test (run file directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = fetch_by_keyword("retrieval augmented generation", max_results=2)
    for p in papers:
        print(f"  [{p['year']}] {p['title']} — {len(p['pdf_bytes'])} bytes")