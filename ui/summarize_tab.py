"""
Combined Summarize + Compare tab.
- Summarize: select one paper, generate structured summary
- Compare: select two papers, generate side-by-side comparison
"""

import streamlit as st
from generation.chains import summarize_paper, compare_papers
from utils.export import export_session_markdown


def _citation_panel(chunks: list[dict], label: str = "Sources"):
    with st.expander(f"📎 {label} ({len(chunks)} chunks used)", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("paper_title", "Unknown")
            section = chunk.get("section_label", "")
            year = chunk.get("year", "")
            st.markdown(f"**[{i}] {title}** ({year}) · *{section}*")
            st.caption(chunk.get("text", "")[:400] + ("…" if len(chunk.get("text", "")) > 400 else ""))
            if i < len(chunks):
                st.divider()


def render(papers: list[dict]):
    """
    Render the Summarize & Compare tab.

    Args:
        papers: List of indexed paper dicts from indexer.list_papers()
    """

    if not papers:
        st.info("No papers indexed yet. Add papers from the sidebar first.", icon="📭")
        return

    paper_map = {p["paper_title"]: p for p in papers}
    paper_titles = list(paper_map.keys())

    # ── Session state for results ──
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}   # paper_id -> summary text
    if "comparisons" not in st.session_state:
        st.session_state.comparisons = []  # list of {paper_a, paper_b, text}

    # Summarize
    st.markdown("### Summarize a Paper")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_title = st.selectbox("Select paper to summarize:", paper_titles, key="sum_select")
    with col2:
        st.write("")
        st.write("")
        summarize_btn = st.button("✨ Summarize", use_container_width=True, type="primary")

    selected_paper = paper_map[selected_title]

    if summarize_btn:
        with st.spinner(f"Summarizing '{selected_title}'…"):
            try:
                summary_text, chunks = summarize_paper(
                    paper_id=selected_paper["paper_id"],
                    paper_title=selected_paper["paper_title"],
                )
                st.session_state.summaries[selected_paper["paper_id"]] = summary_text
                st.session_state[f"sum_chunks_{selected_paper['paper_id']}"] = chunks
            except Exception as e:
                st.error(f"Error: {e}")

    # Display all cached summaries
    if st.session_state.summaries:
        # Show the one just generated or the selected paper's if it exists
        pid = selected_paper["paper_id"]
        if pid in st.session_state.summaries:
            with st.container(border=True):
                st.markdown(f"#### {selected_title}")
                st.markdown(st.session_state.summaries[pid])
                cached_chunks = st.session_state.get(f"sum_chunks_{pid}", [])
                if cached_chunks:
                    _citation_panel(cached_chunks, label="Content used")

    # ── Show other cached summaries ──
    other_summaries = {k: v for k, v in st.session_state.summaries.items() if k != selected_paper["paper_id"]}
    if other_summaries:
        with st.expander(f"📚 Previously generated summaries ({len(other_summaries)})", expanded=False):
            for pid, text in other_summaries.items():
                title = next((p["paper_title"] for p in papers if p["paper_id"] == pid), pid)
                st.markdown(f"**{title}**")
                st.markdown(text)
                st.divider()

    st.divider()

    
    
    # Compare
    st.markdown("### Compare Two Papers")

    if len(papers) < 2:
        st.info("Index at least 2 papers to use the comparison feature.", icon="ℹ️")
    else:
        col_a, col_b, col_btn = st.columns([3, 3, 1])
        with col_a:
            title_a = st.selectbox("Paper A:", paper_titles, key="cmp_a")
        with col_b:
            other_titles = [t for t in paper_titles if t != title_a]
            title_b = st.selectbox("Paper B:", other_titles, key="cmp_b")
        with col_btn:
            st.write("")
            st.write("")
            compare_btn = st.button("⚖️ Compare", use_container_width=True, type="primary")

        paper_a = paper_map[title_a]
        paper_b = paper_map[title_b]

        if compare_btn:
            with st.spinner(f"Comparing papers…"):
                try:
                    comparison_text, chunks = compare_papers(paper_a, paper_b)
                    st.session_state.comparisons.append({
                        "paper_a": paper_a,
                        "paper_b": paper_b,
                        "text": comparison_text,
                        "chunks": chunks,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

        # Show comparisons (most recent first)
        for comp in reversed(st.session_state.comparisons):
            ta = comp["paper_a"]["paper_title"]
            tb = comp["paper_b"]["paper_title"]
            with st.container(border=True):
                st.markdown(f"#### {ta} vs {tb}")
                st.markdown(comp["text"])
                if comp.get("chunks"):
                    _citation_panel(comp["chunks"], label="Content used")

    st.divider()

    
    
    # Export
    st.markdown("### Export Research Brief")

    chat_history = st.session_state.get("chat_history", [])
    has_content = st.session_state.summaries or st.session_state.comparisons or chat_history

    if not has_content:
        st.caption("Generate summaries, comparisons, or chat Q&A first — then export everything as Markdown.")
    else:
        if st.button("Download as Markdown", use_container_width=True):
            md = export_session_markdown(
                papers=papers,
                chat_history=chat_history,
                summaries=st.session_state.summaries,
                comparisons=st.session_state.comparisons,
            )
            st.download_button(
                label="Save research_brief.md",
                data=md,
                file_name="research_brief.md",
                mime="text/markdown",
                use_container_width=True,
            )

