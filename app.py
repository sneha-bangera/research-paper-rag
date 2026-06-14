"""
app.py — Phase 2 entry point
Streamlit app with two tabs: Chat and Summarize/Compare.
Sidebar handles all ingestion (ArXiv + PDF upload) and paper management.

Run with:  streamlit run app.py
"""

import logging
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from ingestion import ingest_arxiv_keyword, ingest_arxiv_id, ingest_uploaded_pdf
from ingestion.indexer import get_indexer
from ui.chat_tab import render as render_chat
from ui.summarize_tab import render as render_summarize

# Page config
st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

# Sidebar — ingestion + paper list
with st.sidebar:
    st.markdown("""
     <style>
     .custom-title {
         font-size: 2.5rem;
         font-weight: 700;
         background: linear-gradient(135deg, #00a2ff, #0084d1);
         -webkit-background-clip: text;
         -webkit-text-fill-color: transparent;
         background-clip: text;
     }
     </style>
     <h1 class="custom-title">Research Paper RAG</h1>
 """, unsafe_allow_html=True)
    # st.title("Research Paper RAG")
    st.caption("Add papers, then chat or summarize.")

    st.header("+ Add Papers")

    ingestion_mode = st.radio(
        "Source",
        ["ArXiv Keyword", "ArXiv ID / URL", "Upload PDF"],
        label_visibility="collapsed",
    )

    if ingestion_mode == "ArXiv Keyword":
        query = st.text_input("Search query", placeholder="e.g. retrieval augmented generation")
        max_results = st.slider("Max papers", 1, 10, 3)
        if st.button("Fetch & Index", use_container_width=True, type="primary"):
            if query:
                with st.spinner(f"Fetching papers for '{query}'…"):
                    try:
                        results = ingest_arxiv_keyword(query, max_results=max_results)
                        st.success(f"Ingested {len(results)} paper(s)")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Enter a search query.")

    elif ingestion_mode == "ArXiv ID / URL":
        arxiv_input = st.text_input("ArXiv ID or URL", placeholder="2005.11401")
        if st.button("Fetch & Index", use_container_width=True, type="primary"):
            if arxiv_input:
                with st.spinner("Fetching paper…"):
                    try:
                        result = ingest_arxiv_id(arxiv_input)
                        st.success(f" {result['metadata']['title']}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Enter an ArXiv ID or URL.")

    else:  # Upload PDF
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        with st.expander("Optional metadata"):
            manual_title = st.text_input("Title (optional)")
            manual_authors = st.text_input("Authors (comma-separated, optional)")
            manual_year = st.number_input("Year", min_value=1900, max_value=2100, value=2024, step=1)
        if st.button("Upload & Index", use_container_width=True, type="primary"):
            if uploaded_files:
                authors_list = [a.strip() for a in manual_authors.split(",") if a.strip()] or None
                for f in uploaded_files:
                    with st.spinner(f"Processing {f.name}…"):
                        try:
                            ingest_uploaded_pdf(
                                pdf_bytes=f.read(),
                                filename=f.name,
                                title=manual_title or None,
                                authors=authors_list,
                                year=int(manual_year),
                            )
                        except Exception as e:
                            st.error(f"{f.name}: {e}")
                st.success(" Done")
                st.rerun()
            else:
                st.warning("Upload at least one PDF.")

    st.divider()

    # ── Indexed papers list ──
    indexer = get_indexer()
    papers = indexer.list_papers()
    stats = indexer.get_collection_stats()

    col1, col2 = st.columns(2)
    col1.metric("Papers", stats["papers"])
    col2.metric("Chunks", stats["total_chunks"])

    if papers:
        st.markdown("**Indexed Papers**")
        for paper in papers:
            title = paper.get("paper_title", "Untitled")
            year = paper.get("year", "")
            source_icon = "🌐" if paper.get("source") == "arxiv" else "📎"
            abstract = paper.get("abstract", "")
            arxiv_id = paper.get("arxiv_id", "")
            authors = paper.get("authors", "")

            # Title row with remove button
            left, right = st.columns([5, 1])
            short_title = title[:42] + "…" if len(title) > 42 else title
            left.caption(f"{source_icon} **{short_title}** ({year})")
            if right.button("✕", key=f"rm_{paper['paper_id']}", help="Remove"):
                indexer.delete_paper(paper["paper_id"])
                indexer.save_bm25()
                st.rerun()

            # Abstract expander — always show, populate with whatever we have
            with st.expander("Abstract", expanded=False):
                if authors:
                    st.caption(f"*{authors}*")
                if arxiv_id:
                    st.caption(f"[arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})")
                if abstract:
                    st.write(abstract)
                elif paper.get("source") == "upload":
                    st.caption("Abstract not available for uploaded PDFs without metadata.")
                else:
                    st.caption("Abstract not found in indexed chunks.")

    st.divider()
    with st.expander("⚠️ Reset index"):
        if st.button("Clear everything", use_container_width=True):
            indexer.reset()
            for key in ["chat_history", "chat_sources", "summaries", "comparisons"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()




tab_chat, tab_sum = st.tabs(["💬 Chat", "📝 Summarize & Compare"])

with tab_chat:
    render_chat(papers)

with tab_sum:
    render_summarize(papers)
