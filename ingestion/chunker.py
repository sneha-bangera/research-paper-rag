"""
Hierarchical chunking strategy with three levels:
  Level 1 — Abstract chunk (one per paper)
  Level 2 — Section-level chunks
  Level 3 — Paragraph-level chunks (~300–500 tokens, with overlap)

Each chunk is a dict ready to be embedded and stored in ChromaDB.
"""

import re
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate token count: 1 token ≈ 4 characters (conservative for academic English)
CHARS_PER_TOKEN = 4
TARGET_CHUNK_TOKENS = 400       # target size for level-3 chunks
MAX_CHUNK_TOKENS = 500          # hard ceiling
OVERLAP_TOKENS = 50             # overlap between adjacent level-3 chunks

TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN   # 1600
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN         # 2000
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN             # 200

# Sections that should NOT produce level-3 chunks (noisy/unhelpful)
SKIP_SECTIONS = {"references", "appendix", "_unclassified"}

# Human-readable section labels
SECTION_LABELS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related_work": "Related Work",
    "methodology": "Methodology",
    "experiments": "Experiments",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "limitations": "Limitations",
    "future_work": "Future Work",
    "_unclassified": "Body",
}


def _make_chunk_id() -> str:
    return str(uuid.uuid4())


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _paragraph_chunks(
    paragraphs: list[str],
    paper_id: str,
    paper_metadata: dict,
    section_key: str,
    section_label: str,
    start_chunk_index: int,
) -> list[dict]:
    """
    Convert a list of paragraphs into overlapping level-3 chunks.
    Accumulates paragraphs until the target size is reached, then yields a chunk.
    Overlap is achieved by carrying the last OVERLAP_CHARS of the previous chunk
    as a prefix of the next.
    """
    chunks: list[dict] = []
    chunk_index = start_chunk_index
    buffer: list[str] = []
    buffer_chars = 0
    overlap_tail = ""

    def _flush(buf: list[str], overlap: str) -> Optional[dict]:
        text = (overlap + " " + " ".join(buf)).strip() if overlap else " ".join(buf).strip()
        text = re.sub(r"\s{2,}", " ", text)
        if len(text) < 50:
            return None
        return {
            "chunk_id": _make_chunk_id(),
            "chunk_level": 3,
            "chunk_index": chunk_index,
            "section_key": section_key,
            "section_label": section_label,
            "text": text,
            "approx_tokens": _approx_tokens(text),
            **paper_metadata,
        }

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds the max chunk size, split it by sentences
        if len(para) > MAX_CHUNK_CHARS:
            sentences = _split_into_sentences(para)
            for sent in sentences:
                buffer.append(sent)
                buffer_chars += len(sent)
                if buffer_chars >= TARGET_CHUNK_CHARS:
                    chunk = _flush(buffer, overlap_tail)
                    if chunk:
                        chunk["chunk_index"] = chunk_index
                        chunks.append(chunk)
                        chunk_index += 1
                        # Compute overlap tail
                        full_text = " ".join(buffer)
                        overlap_tail = full_text[-OVERLAP_CHARS:] if len(full_text) > OVERLAP_CHARS else full_text
                    buffer = []
                    buffer_chars = 0
        else:
            buffer.append(para)
            buffer_chars += len(para)
            if buffer_chars >= TARGET_CHUNK_CHARS:
                chunk = _flush(buffer, overlap_tail)
                if chunk:
                    chunk["chunk_index"] = chunk_index
                    chunks.append(chunk)
                    chunk_index += 1
                    full_text = " ".join(buffer)
                    overlap_tail = full_text[-OVERLAP_CHARS:] if len(full_text) > OVERLAP_CHARS else full_text
                buffer = []
                buffer_chars = 0

    # Flush remaining buffer
    if buffer:
        chunk = _flush(buffer, overlap_tail)
        if chunk:
            chunk["chunk_index"] = chunk_index
            chunks.append(chunk)

    return chunks


# Public API

def chunk_document(parsed_doc: dict) -> list[dict]:
    """
    Apply hierarchical chunking to a parsed document.

    Args:
        parsed_doc: Output from pdf_parser.parse_pdf() — must have:
                    { "metadata": {...}, "sections": {...} }

    Returns:
        List of chunk dicts, each containing:
        {
            chunk_id, chunk_level, chunk_index,
            section_key, section_label, text, approx_tokens,
            paper_id, paper_title, authors, year, source, arxiv_id,
        }
    """
    meta = parsed_doc["metadata"]
    sections = parsed_doc.get("sections", {})

    # Build a stable paper_id from arxiv_id or a uuid derived from title
    paper_id = meta.get("arxiv_id") or re.sub(r"\W+", "_", meta.get("title", "unknown")).lower()

    # Common metadata attached to every chunk
    paper_metadata = {
        "paper_id": paper_id,
        "paper_title": meta.get("title", "Unknown"),
        "authors": ", ".join(meta.get("authors", [])) if isinstance(meta.get("authors"), list) else meta.get("authors", ""),
        "year": meta.get("year", 0),
        "source": meta.get("source", "unknown"),
        "arxiv_id": meta.get("arxiv_id", ""),
    }

    all_chunks: list[dict] = []
    global_chunk_index = 0

    # ── Level 1: Abstract ──
    abstract_paras = sections.get("abstract", [])
    if not abstract_paras and meta.get("abstract"):
        abstract_paras = [meta["abstract"]]

    if abstract_paras:
        abstract_text = " ".join(abstract_paras).strip()
        abstract_text = re.sub(r"\s{2,}", " ", abstract_text)
        all_chunks.append({
            "chunk_id": _make_chunk_id(),
            "chunk_level": 1,
            "chunk_index": global_chunk_index,
            "section_key": "abstract",
            "section_label": "Abstract",
            "text": abstract_text,
            "approx_tokens": _approx_tokens(abstract_text),
            **paper_metadata,
        })
        global_chunk_index += 1
        logger.debug("Level-1 abstract chunk: %d tokens", _approx_tokens(abstract_text))

    # ── Level 2 + 3: Section and paragraph chunks ──
    ordered_sections = [
        "introduction", "background", "related_work",
        "methodology", "experiments", "results",
        "discussion", "conclusion", "limitations", "future_work",
    ]
    # Also include any sections found in the doc that aren't in our ordered list
    extra_sections = [k for k in sections if k not in ordered_sections and k not in {"abstract", "_unclassified"} | SKIP_SECTIONS]
    all_section_keys = ordered_sections + extra_sections

    for section_key in all_section_keys:
        if section_key not in sections:
            continue
        paragraphs = sections[section_key]
        if not paragraphs:
            continue

        section_label = SECTION_LABELS.get(section_key, section_key.replace("_", " ").title())

        # Level 2: Full section as one chunk
        section_text = " ".join(paragraphs).strip()
        section_text = re.sub(r"\s{2,}", " ", section_text)
        if section_text and len(section_text) > 50:
            all_chunks.append({
                "chunk_id": _make_chunk_id(),
                "chunk_level": 2,
                "chunk_index": global_chunk_index,
                "section_key": section_key,
                "section_label": section_label,
                "text": section_text,
                "approx_tokens": _approx_tokens(section_text),
                **paper_metadata,
            })
            global_chunk_index += 1

        # Level 3: Paragraph-level chunks (skip noisy sections)
        if section_key not in SKIP_SECTIONS:
            l3_chunks = _paragraph_chunks(
                paragraphs,
                paper_id=paper_id,
                paper_metadata=paper_metadata,
                section_key=section_key,
                section_label=section_label,
                start_chunk_index=global_chunk_index,
            )
            all_chunks.extend(l3_chunks)
            global_chunk_index += len(l3_chunks)

    logger.info(
        "Chunked '%s': %d total chunks (L1=%d, L2=%d, L3=%d)",
        paper_metadata["paper_title"],
        len(all_chunks),
        sum(1 for c in all_chunks if c["chunk_level"] == 1),
        sum(1 for c in all_chunks if c["chunk_level"] == 2),
        sum(1 for c in all_chunks if c["chunk_level"] == 3),
    )

    return all_chunks


def get_chunk_summary(chunks: list[dict]) -> dict:
    """Return a summary of chunk distribution for display/debugging."""
    by_level = {1: 0, 2: 0, 3: 0}
    by_section: dict[str, int] = {}
    total_tokens = 0
    for c in chunks:
        by_level[c["chunk_level"]] = by_level.get(c["chunk_level"], 0) + 1
        by_section[c["section_label"]] = by_section.get(c["section_label"], 0) + 1
        total_tokens += c.get("approx_tokens", 0)
    return {
        "total_chunks": len(chunks),
        "by_level": by_level,
        "by_section": by_section,
        "approx_total_tokens": total_tokens,
    }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_doc = {
        "metadata": {
            "title": "Test Paper",
            "authors": ["Alice", "Bob"],
            "year": 2024,
            "arxiv_id": "0000.00000",
            "source": "arxiv",
            "abstract": "This is the abstract of the test paper. It describes the paper briefly.",
        },
        "sections": {
            "abstract": ["This is the abstract of the test paper. It describes the paper briefly."],
            "introduction": [
                "Retrieval-Augmented Generation (RAG) is a method that combines retrieval with generation.",
                "It was first proposed to address knowledge limitations in LLMs.",
                "This paper extends RAG with hierarchical retrieval strategies.",
            ],
            "methodology": [
                "We propose a three-level chunking strategy.",
                "The first level stores the full abstract.",
                "The second level stores complete sections.",
                "The third level stores overlapping paragraph-level chunks.",
            ],
        },
        "page_count": 10,
        "full_text": "",
    }
    chunks = chunk_document(sample_doc)
    summary = get_chunk_summary(chunks)
    print(f"\nChunk summary: {summary}")
    for c in chunks:
        print(f"  [L{c['chunk_level']}] {c['section_label']:20s} | {c['approx_tokens']:4d} tok | {c['text'][:80]}...")