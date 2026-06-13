"""
Export session data (chat history + summaries + comparisons) as a Markdown file.
"""

from datetime import datetime


def export_session_markdown(
    papers: list[dict],
    chat_history: list[dict],
    summaries: dict[str, str],   # paper_id -> summary text
    comparisons: list[dict],     # list of {paper_a, paper_b, text}
) -> str:
    """
    Build a Markdown research brief from the current session.

    Returns:
        Markdown string ready to be written to a .md file or offered for download.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Research Brief",
        f"*Generated: {now}*",
        "",
    ]

    # ── Papers section ──
    lines += ["## Papers Indexed", ""]
    for p in papers:
        title = p.get("paper_title", "Untitled")
        authors = p.get("authors", "Unknown")
        year = p.get("year", "")
        source = p.get("source", "")
        arxiv_id = p.get("arxiv_id", "")
        line = f"- **{title}** — {authors} ({year})"
        if arxiv_id:
            line += f" · [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"
        elif source == "upload":
            line += " · Uploaded PDF"
        lines.append(line)
    lines.append("")

    # ── Summaries ──
    if summaries:
        lines += ["## Paper Summaries", ""]
        for paper_id, summary_text in summaries.items():
            # Find paper title
            title = next((p["paper_title"] for p in papers if p["paper_id"] == paper_id), paper_id)
            lines += [f"### {title}", "", summary_text, ""]

    # ── Comparisons ──
    if comparisons:
        lines += ["## Paper Comparisons", ""]
        for comp in comparisons:
            title_a = comp.get("paper_a", {}).get("paper_title", "Paper A")
            title_b = comp.get("paper_b", {}).get("paper_title", "Paper B")
            lines += [f"### {title_a} vs {title_b}", "", comp.get("text", ""), ""]

    # ── Chat history ──
    if chat_history:
        lines += ["## Q&A Session", ""]
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lines += [f"**Q:** {content}", ""]
            else:
                lines += [f"**A:** {content}", ""]

    return "\n".join(lines)
