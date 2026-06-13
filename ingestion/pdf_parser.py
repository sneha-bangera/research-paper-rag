"""
Extracts and cleans text from PDF bytes using PyMuPDF (fitz).
Detects section headings, strips noise, and handles multi-column layouts.
Returns a structured dict ready for chunking.
"""

import re
import logging
from typing import Optional
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Section heading patterns (order matters — more specific first)
SECTION_PATTERNS = [
    (re.compile(r"^\s*abstract\s*$", re.I), "abstract"),
    (re.compile(r"^\s*\d*\.?\s*introduction\s*$", re.I), "introduction"),
    (re.compile(r"^\s*\d*\.?\s*related\s+work\s*$", re.I), "related_work"),
    (re.compile(r"^\s*\d*\.?\s*background\s*$", re.I), "background"),
    (re.compile(r"^\s*\d*\.?\s*method(?:ology|s)?\s*$", re.I), "methodology"),
    (re.compile(r"^\s*\d*\.?\s*(?:proposed\s+)?(?:approach|model|framework|architecture)\s*$", re.I), "methodology"),
    (re.compile(r"^\s*\d*\.?\s*experiment(?:s|al\s+(?:setup|results))?\s*$", re.I), "experiments"),
    (re.compile(r"^\s*\d*\.?\s*results?\s*(?:and\s+discussion)?\s*$", re.I), "results"),
    (re.compile(r"^\s*\d*\.?\s*discussion\s*$", re.I), "discussion"),
    (re.compile(r"^\s*\d*\.?\s*(?:conclusion|conclusions|concluding\s+remarks)\s*$", re.I), "conclusion"),
    (re.compile(r"^\s*\d*\.?\s*(?:limitation|limitations)\s*$", re.I), "limitations"),
    (re.compile(r"^\s*\d*\.?\s*(?:future\s+work|future\s+directions)\s*$", re.I), "future_work"),
    (re.compile(r"^\s*\d*\.?\s*(?:references|bibliography)\s*$", re.I), "references"),
    (re.compile(r"^\s*\d*\.?\s*appendix\s*", re.I), "appendix"),
]

# Lines to strip: page numbers, URLs, email addresses, short noise lines
NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                          # bare page numbers
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),                  # — N —
    re.compile(r"https?://\S+"),                          # URLs (inline strip)
    re.compile(r"\S+@\S+\.\S+"),                          # emails
    re.compile(r"^[\W_]{5,}$"),                           # lines of symbols only
    re.compile(r"^\s*(?:preprint|arxiv|submitted to)\b", re.I),  # journal footers
]


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False  # blank lines handled separately
    if len(stripped) < 3:
        return True
    for pat in NOISE_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def _detect_section(line: str) -> Optional[str]:
    stripped = line.strip()
    for pattern, section_name in SECTION_PATTERNS:
        if pattern.match(stripped):
            return section_name
    return None


def _extract_blocks_from_page(page: fitz.Page) -> list[str]:
    """
    Extract text blocks from a page, handling multi-column layouts.
    Uses PyMuPDF's block extraction sorted by vertical then horizontal position.
    """
    blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
    # Filter image blocks (block_type == 1)
    text_blocks = [b for b in blocks if b[6] == 0]
    # Sort: top-to-bottom, then left-to-right within a horizontal band
    # Band height ~20pt handles minor baseline misalignments
    text_blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
    return [b[4] for b in text_blocks]


# Public API
def parse_pdf(
    pdf_bytes: bytes,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Parse PDF bytes and return a structured document dict.

    Args:
        pdf_bytes: Raw PDF file bytes
        metadata:  Optional dict with keys: title, authors, year, arxiv_id, source, abstract

    Returns:
        {
            "metadata": {...},
            "full_text": str,           # cleaned full text
            "sections": {               # section_name -> list of paragraph strings
                "abstract": [...],
                "introduction": [...],
                ...
                "_unclassified": [...]  # paragraphs before first detected heading
            },
            "page_count": int,
        }
    """
    if metadata is None:
        metadata = {}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count

    # ── Collect all text blocks across pages ──
    all_lines: list[str] = []
    for page in doc:
        blocks = _extract_blocks_from_page(page)
        for block_text in blocks:
            # Split block into lines, strip each
            for line in block_text.split("\n"):
                all_lines.append(line)

    doc.close()

    # ── Pass 1: Clean noise lines ──
    cleaned_lines: list[str] = []
    for line in all_lines:
        if _is_noise(line):
            continue
        # Strip inline URLs / emails
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"\S+@\S+\.\S+", "", line)
        cleaned_lines.append(line)

    # ── Pass 2: Segment into sections ──
    sections: dict[str, list[str]] = {"_unclassified": []}
    current_section = "_unclassified"
    in_references = False

    for line in cleaned_lines:
        detected = _detect_section(line)
        if detected:
            current_section = detected
            if detected not in sections:
                sections[detected] = []
            if detected == "references":
                in_references = True
            continue  # skip the heading line itself

        # Skip reference content (noisy, not useful for Q&A)
        if in_references and current_section == "references":
            continue

        sections.setdefault(current_section, []).append(line)

    # ── Pass 3: Merge lines into paragraphs per section ──
    def _lines_to_paragraphs(lines: list[str]) -> list[str]:
        """Join consecutive non-empty lines into paragraphs, split on blank lines."""
        paragraphs: list[str] = []
        current: list[str] = []
        for line in lines:
            if line.strip() == "":
                if current:
                    para = " ".join(current).strip()
                    # Remove excessive whitespace
                    para = re.sub(r"\s{2,}", " ", para)
                    if len(para) > 20:  # filter very short fragments
                        paragraphs.append(para)
                    current = []
            else:
                current.append(line.strip())
        if current:
            para = " ".join(current).strip()
            para = re.sub(r"\s{2,}", " ", para)
            if len(para) > 20:
                paragraphs.append(para)
        return paragraphs

    structured_sections: dict[str, list[str]] = {}
    for section_name, lines in sections.items():
        paragraphs = _lines_to_paragraphs(lines)
        if paragraphs:
            structured_sections[section_name] = paragraphs

    # ── Build full text (excluding _unclassified noise near references) ──
    ordered_section_keys = [
        "abstract", "introduction", "background", "related_work",
        "methodology", "experiments", "results", "discussion",
        "conclusion", "limitations", "future_work", "_unclassified",
    ]
    full_text_parts: list[str] = []
    for key in ordered_section_keys:
        if key in structured_sections:
            full_text_parts.extend(structured_sections[key])
    # Add any remaining sections not in the ordered list
    for key, paras in structured_sections.items():
        if key not in ordered_section_keys:
            full_text_parts.extend(paras)

    full_text = "\n\n".join(full_text_parts)

    # ── If abstract not detected but metadata has it, use that ──
    if "abstract" not in structured_sections and metadata.get("abstract"):
        structured_sections["abstract"] = [metadata["abstract"]]

    logger.info(
        "Parsed PDF: %d pages, %d sections, %d chars",
        page_count,
        len(structured_sections),
        len(full_text),
    )

    return {
        "metadata": metadata,
        "full_text": full_text,
        "sections": structured_sections,
        "page_count": page_count,
    }


def parse_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str = "uploaded.pdf",
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    year: Optional[int] = None,
) -> dict:
    """
    Convenience wrapper for user-uploaded PDFs.
    Falls back to filename-derived title and unknown metadata if not provided.
    """
    inferred_title = title or filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()
    metadata = {
        "title": inferred_title,
        "authors": authors or ["Unknown"],
        "year": year or 0,
        "arxiv_id": None,
        "source": "upload",
        "abstract": "",
        "pdf_url": None,
        "categories": [],
    }
    return parse_pdf(pdf_bytes, metadata=metadata)


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys, pathlib
    if len(sys.argv) > 1:
        path = pathlib.Path(sys.argv[1])
        doc = parse_pdf(path.read_bytes(), metadata={"title": path.stem})
        for sec, paras in doc["sections"].items():
            print(f"\n=== {sec.upper()} ({len(paras)} paragraphs) ===")
            print(paras[0][:300] if paras else "(empty)")
    else:
        print("Usage: python pdf_parser.py <path_to_pdf>")