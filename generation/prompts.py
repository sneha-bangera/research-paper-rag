"""
All prompt templates for the RAG system.
Each function returns a formatted string ready to send to the LLM.
"""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("paper_title", "Unknown")
        section = chunk.get("section_label", "")
        year = chunk.get("year", "")
        header = f"[{i}] {title} ({year}) — {section}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def qa_prompt(query: str, chunks: list[dict], chat_history: str = "") -> str:
    context = _format_context(chunks)
    history_block = f"\n\nConversation so far:\n{chat_history}" if chat_history else ""
    return f"""You are a research assistant that answers questions about academic papers.
Answer the question using ONLY the provided context. Be precise and specific.
After each claim, cite the source using the format: [Paper Title, Section].
If the context does not contain enough information to answer, say so clearly.
{history_block}

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with inline citations):"""


def summarize_prompt(paper_title: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return f"""You are a research assistant. Summarize the following paper in a structured format.

PAPER: {paper_title}

CONTENT:
{context}

Provide a structured summary with these exact sections:
**Objective**: What problem does this paper solve?
**Methodology**: What approach or method do the authors propose?
**Key Findings**: What are the main results or contributions?
**Limitations**: What are the stated or apparent limitations?
**One-line takeaway**: A single sentence summary for a busy researcher."""


def compare_prompt(paper_a: dict, paper_b: dict, chunks_a: list[dict], chunks_b: list[dict]) -> str:
    ctx_a = _format_context(chunks_a)
    ctx_b = _format_context(chunks_b)
    return f"""You are a research assistant comparing two academic papers.

PAPER A: {paper_a['paper_title']}
{ctx_a}

---

PAPER B: {paper_b['paper_title']}
{ctx_b}

---

Compare these two papers across the following dimensions. Be specific and cite which paper supports each point.

**Problem / Objective**: What problem does each paper address?
**Approach / Method**: How does each paper tackle the problem?
**Dataset / Evaluation**: What datasets or benchmarks do they use?
**Key Results**: What are the reported outcomes?
**Limitations**: What are the weaknesses of each approach?
**Verdict**: In what scenarios would you prefer one over the other?"""


def confidence_prompt(query: str, answer: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return f"""You are evaluating a RAG system answer.

QUESTION: {query}

ANSWER GIVEN: {answer[:400]}

CONTEXT THAT WAS AVAILABLE:
{context}

Rate how well the CONTEXT supported the ANSWER on a scale of 1 to 5:
5 = answer is fully and directly supported by the context
4 = answer is mostly supported, minor gaps
3 = answer is partially supported
2 = answer is only loosely related to context
1 = context had little to no relevant information

Reply with a SINGLE digit (1, 2, 3, 4, or 5) and nothing else."""
