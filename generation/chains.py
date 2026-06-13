"""
LLM generation functions using Groq (LLaMA 3).
Supports streaming and non-streaming responses.
Maintains conversational memory via a simple in-memory history list.
"""

import os
import logging
from typing import Generator, Optional

from groq import Groq
from dotenv import load_dotenv

from .prompts import qa_prompt, summarize_prompt, compare_prompt
from retrieval.hybrid_retriever import retrieve, retrieve_for_paper

load_dotenv()
logger = logging.getLogger(__name__)

# llama-3.1-8b-instant: 6x higher TPM (131k limit vs 12k for 70b on free tier)
# Switch to llama3-70b-8192 if you upgrade to Groq Dev tier
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024

# Hard cap on chars sent per chunk to stay under TPM limits
# ~500 chars ≈ 125 tokens; 5 chunks = ~625 input tokens for Q&A (well within limit)
QA_CHUNK_CHARS = 600
SUMMARIZE_CHUNK_CHARS = 800   # fewer chunks, can afford a bit more per chunk
COMPARE_CHUNK_CHARS = 600     # 3 chunks × 2 papers = 6 chunks, keep tight


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
    return Groq(api_key=api_key)


def _truncate_chunks(chunks: list[dict], max_chars: int) -> list[dict]:
    """Return copies of chunks with text truncated to max_chars."""
    result = []
    for c in chunks:
        copy = dict(c)
        text = copy.get("text", "")
        if len(text) > max_chars:
            copy["text"] = text[:max_chars] + "…"
        result.append(copy)
    return result


def _format_history(messages: list[dict]) -> str:
    """Convert message list to a compact history string for the prompt."""
    lines = []
    for m in messages[-4:]:  # last 2 turns only, keep it tight
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:200]}")
    return "\n".join(lines)


# Chat / Q&A  (streaming)

def stream_qa(
    query: str,
    chat_history: list[dict],
    paper_ids: Optional[list[str]] = None,
) -> tuple[Generator[str, None, None], list[dict]]:
    """
    Run hybrid retrieval then stream a cited answer from the LLM.

    Args:
        query:        User's question
        chat_history: List of {"role": "user"/"assistant", "content": str}
        paper_ids:    Optional list to restrict retrieval to specific papers

    Returns:
        (token_generator, retrieved_chunks)
        Caller iterates the generator to get streamed tokens.
        retrieved_chunks is available immediately for the citation panel.
    """
    chunks = retrieve(query, top_k=5, paper_ids=paper_ids)
    if not chunks:
        def _empty():
            yield "I couldn't find relevant content in the indexed papers for this question. Try indexing more papers or rephrasing your query."
        return _empty(), []

    history_str = _format_history(chat_history)
    prompt = qa_prompt(query, _truncate_chunks(chunks, QA_CHUNK_CHARS), chat_history=history_str)

    client = _get_client()
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        stream=True,
    )

    def _token_gen():
        for chunk_event in stream:
            delta = chunk_event.choices[0].delta.content
            if delta:
                yield delta

    return _token_gen(), chunks


# Summarization  (non-streaming, returns full text)

def summarize_paper(paper_id: str, paper_title: str) -> tuple[str, list[dict]]:
    """
    Generate a structured summary for a single paper.

    Returns:
        (summary_text, source_chunks)
    """
    chunks = retrieve_for_paper(paper_id, top_k=4)   # 4 chunks max (down from 8)
    if not chunks:
        return f"No content found for paper '{paper_title}'. Make sure it has been indexed.", []

    prompt = summarize_prompt(paper_title, _truncate_chunks(chunks, SUMMARIZE_CHUNK_CHARS))
    client = _get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content, chunks


# Paper comparison  (non-streaming)

def compare_papers(paper_a: dict, paper_b: dict) -> tuple[str, list[dict]]:
    """
    Generate a structured comparison between two papers.

    Args:
        paper_a, paper_b: dicts with at least {paper_id, paper_title}

    Returns:
        (comparison_text, combined_source_chunks)
    """
    chunks_a = retrieve_for_paper(paper_a["paper_id"], top_k=3)   # 3 per paper (down from 6)
    chunks_b = retrieve_for_paper(paper_b["paper_id"], top_k=3)

    if not chunks_a:
        return f"No content found for '{paper_a['paper_title']}'.", []
    if not chunks_b:
        return f"No content found for '{paper_b['paper_title']}'.", []

    prompt = compare_prompt(
        paper_a, paper_b,
        _truncate_chunks(chunks_a, COMPARE_CHUNK_CHARS),
        _truncate_chunks(chunks_b, COMPARE_CHUNK_CHARS),
    )
    client = _get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,   # was MAX_TOKENS * 2; 1024 is enough for comparison
    )
    return response.choices[0].message.content, chunks_a + chunks_b