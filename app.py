# import os
# import sys
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()

# # Page config — must be first Streamlit call
# st.set_page_config(
#     page_title="Research Paper Q&A",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Add project root to path
# sys.path.insert(0, os.path.dirname(__file__))

# from ingestion import ingest_arxiv_keyword, ingest_arxiv_id, ingest_uploaded_pdf
# from ingestion.indexer import get_indexer
# from retrieval.hybrid_retriever import HybridRetriever
# from ui.chat_tab import render_chat_tab
# from ui.summarize_compare_tab import render_summarize_compare_tab
# from utils.export import export_session_as_markdown

# # from ingestion.arxiv_fetcher import fetch_arxiv_papers, fetch_by_arxiv_id
# # from ingestion.pdf_parser import parse_pdf_bytes, parse_pdf_file
# # from ingestion.chunker import chunk_paper
# # from ingestion.indexer import PaperIndexer
# # from retrieval.hybrid_retriever import HybridRetriever
# # from ui.chat_tab import render_chat_tab
# # from ui.summarize_compare_tab import render_summarize_compare_tab
# # from utils.export import export_session_as_markdown

# # -------------------------------------------------------------------
# # Config
# # -------------------------------------------------------------------
# CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
# BM25_PATH = os.path.join(CHROMA_DIR, "bm25_index.pkl")

# # -------------------------------------------------------------------
# # Session state init
# # -------------------------------------------------------------------
# if "indexed_papers" not in st.session_state:
#     st.session_state.indexed_papers = []

# # -------------------------------------------------------------------
# # Load indexer + retriever
# # -------------------------------------------------------------------
# @st.cache_resource
# def load_indexer():
#     return get_indexer()

# @st.cache_resource
# def load_retriever():
#     return HybridRetriever(
#         chroma_persist_dir=CHROMA_DIR,
#         bm25_index_path=BM25_PATH
#     )

# indexer = load_indexer()
# retriever = load_retriever()

# # Load already-indexed papers into session if fresh start
# if not st.session_state.indexed_papers:
#     existing = indexer.get_all_papers()
#     if existing:
#         st.session_state.indexed_papers = existing

# # -------------------------------------------------------------------
# # Sidebar
# # -------------------------------------------------------------------
# with st.sidebar:
#     st.title("📚 Research Q&A")
#     st.markdown("---")

#     # --- ArXiv Fetch ---
#     st.subheader("🔍 Fetch from ArXiv")

#     fetch_mode = st.radio(
#         "Search by",
#         ["Topic/Keyword", "ArXiv ID or URL"],
#         horizontal=True,
#         key="fetch_mode"
#     )

#     if fetch_mode == "Topic/Keyword":
#         arxiv_query = st.text_input(
#             "Search topic",
#             placeholder="e.g. retrieval augmented generation",
#             key="arxiv_query"
#         )
#         n_papers = st.slider("Number of papers", 1, 10, 3, key="n_papers")
#         fetch_btn = st.button("Fetch Papers", type="primary", key="fetch_btn")

#         # Topic/Keyword fetch block — replace the if fetch_btn block:
#         if fetch_btn and arxiv_query:
#             with st.spinner(f"Fetching {n_papers} papers on '{arxiv_query}'..."):
#                 results = ingest_arxiv_keyword(arxiv_query, max_results=n_papers)
#                 newly_added = 0
#                 for r in results:
#                     meta = r["metadata"]
#                     paper_id = meta.get("arxiv_id") or meta.get("paper_id")
#                     already = any(p["paper_id"] == paper_id for p in st.session_state.indexed_papers)
#                     if not already:
#                         st.session_state.indexed_papers.append({
#                             "paper_id": paper_id,
#                             "paper_title": meta.get("title", "Unknown"),
#                             "authors": ", ".join(meta.get("authors", [])),
#                             "year": str(meta.get("year", "")),
#                             "arxiv_id": meta.get("arxiv_id", ""),
#                             "source": "arxiv"
#                         })
#                         newly_added += 1
#                 retriever.reload_bm25()
#                 if newly_added:
#                     st.success(f"✅ Indexed {newly_added} new paper(s)")
#                 else:
#                     st.info("These papers are already indexed.")
#         # if fetch_btn and arxiv_query:
#         #     with st.spinner(f"Fetching {n_papers} papers on '{arxiv_query}'..."):
#         #         papers_data = ingest_arxiv_keyword(arxiv_query, max_results=n_papers)
#         #         newly_added = 0
#         #         for pdata in papers_data:
#         #             already = any(
#         #                 p["paper_id"] == pdata["paper_id"]
#         #                 for p in st.session_state.indexed_papers
#         #             )
#         #             if not already:
#         #                 chunks = chunk_paper(pdata)
#         #                 indexer.index_paper(pdata, chunks)
#         #                 st.session_state.indexed_papers.append(pdata)
#         #                 newly_added += 1

#         #         retriever.reload_bm25()
#         #         if newly_added:
#         #             st.success(f"✅ Indexed {newly_added} new paper(s)")
#         #         else:
#         #             st.info("These papers are already indexed.")

#     else:
#         arxiv_id_input = st.text_input(
#             "ArXiv ID or URL",
#             placeholder="e.g. 2005.11401 or arxiv.org/abs/...",
#             key="arxiv_id_input"
#         )
#         fetch_id_btn = st.button("Fetch Paper", type="primary", key="fetch_id_btn")

#         # ArXiv ID fetch block — replace the if fetch_id_btn block:
#         if fetch_id_btn and arxiv_id_input:
#             with st.spinner("Fetching paper..."):
#                 try:
#                     r = ingest_arxiv_id(arxiv_id_input)
#                     meta = r["metadata"]
#                     paper_id = meta.get("arxiv_id") or meta.get("paper_id")
#                     already = any(p["paper_id"] == paper_id for p in st.session_state.indexed_papers)
#                     if not already:
#                         st.session_state.indexed_papers.append({
#                             "paper_id": paper_id,
#                             "paper_title": meta.get("title", "Unknown"),
#                             "authors": ", ".join(meta.get("authors", [])),
#                             "year": str(meta.get("year", "")),
#                             "arxiv_id": meta.get("arxiv_id", ""),
#                             "source": "arxiv"
#                         })
#                         retriever.reload_bm25()
#                         st.success(f"✅ Indexed: {meta.get('title', paper_id)}")
#                     else:
#                         st.info("Already indexed.")
#                 except Exception as e:
#                     st.error(f"Could not fetch paper: {e}")
#         # if fetch_id_btn and arxiv_id_input:
#         #     with st.spinner("Fetching paper..."):
#         #         pdata = fetch_by_arxiv_id(arxiv_id_input)
#         #         if pdata:
#         #             already = any(
#         #                 p["paper_id"] == pdata["paper_id"]
#         #                 for p in st.session_state.indexed_papers
#         #             )
#         #             if not already:
#         #                 chunks = chunk_paper(pdata)
#         #                 indexer.index_paper(pdata, chunks)
#         #                 st.session_state.indexed_papers.append(pdata)
#         #                 retriever.reload_bm25()
#         #                 st.success(f"✅ Indexed: {pdata['paper_title']}")
#         #             else:
#         #                 st.info("Already indexed.")
#         #         else:
#         #             st.error("Could not fetch paper. Check the ID.")

#     st.markdown("---")

#     # --- PDF Upload ---
#     st.subheader("📤 Upload PDFs")

#     uploaded_files = st.file_uploader(
#         "Upload research papers",
#         type=["pdf"],
#         accept_multiple_files=True,
#         key="pdf_uploader"
#     )

#     if uploaded_files:
#         index_btn = st.button("Index Uploaded PDFs", type="primary", key="index_btn")

#         # PDF upload block — replace the if index_btn block:
#         if index_btn:
#             newly_added = 0
#             for uploaded_file in uploaded_files:
#                 with st.spinner(f"Processing {uploaded_file.name}..."):
#                     pdf_bytes = uploaded_file.read()
#                     try:
#                         r = ingest_uploaded_pdf(
#                             pdf_bytes,
#                             filename=uploaded_file.name
#                         )
#                         meta = r["metadata"]
#                         paper_id = meta.get("arxiv_id") or meta.get("paper_id")
#                         already = any(p["paper_id"] == paper_id for p in st.session_state.indexed_papers)
#                         if not already:
#                             st.session_state.indexed_papers.append({
#                                 "paper_id": paper_id,
#                                 "paper_title": meta.get("title", uploaded_file.name),
#                                 "authors": ", ".join(meta.get("authors", [])),
#                                 "year": str(meta.get("year", "")),
#                                 "arxiv_id": "",
#                                 "source": "uploaded"
#                             })
#                             newly_added += 1
#                     except Exception as e:
#                         st.error(f"Failed to process {uploaded_file.name}: {e}")
#             retriever.reload_bm25()
#             if newly_added:
#                 st.success(f"✅ Indexed {newly_added} PDF(s)")
#             else:
#                 st.info("All uploaded PDFs are already indexed.")
                
#         # if index_btn:
#         #     newly_added = 0
#         #     for uploaded_file in uploaded_files:
#         #         with st.spinner(f"Processing {uploaded_file.name}..."):
#         #             pdf_bytes = uploaded_file.read()
#         #             pdata = parse_pdf_bytes(pdf_bytes, filename=uploaded_file.name)

#         #             already = any(
#         #                 p["paper_id"] == pdata["paper_id"]
#         #                 for p in st.session_state.indexed_papers
#         #             )
#         #             if not already:
#         #                 chunks = chunk_paper(pdata)
#         #                 indexer.index_paper(pdata, chunks)
#         #                 st.session_state.indexed_papers.append(pdata)
#         #                 newly_added += 1

#         #     retriever.reload_bm25()
#         #     if newly_added:
#         #         st.success(f"✅ Indexed {newly_added} PDF(s)")
#         #     else:
#         #         st.info("All uploaded PDFs are already indexed.")

#     st.markdown("---")

#     # --- Indexed Papers List ---
#     st.subheader(f"📋 Indexed Papers ({len(st.session_state.indexed_papers)})")

#     if not st.session_state.indexed_papers:
#         st.caption("No papers indexed yet.")
#     else:
#         for paper in st.session_state.indexed_papers:
#             col1, col2 = st.columns([4, 1])
#             with col1:
#                 source_icon = "🌐" if paper.get("source") == "arxiv" else "📄"
#                 st.caption(
#                     f"{source_icon} {paper['paper_title'][:45]}..."
#                     if len(paper['paper_title']) > 45
#                     else f"{source_icon} {paper['paper_title']}"
#                 )
#             with col2:
#                 if st.button("✕", key=f"remove_{paper['paper_id']}", help="Remove paper"):
#                     indexer.remove_paper(paper["paper_id"])
#                     st.session_state.indexed_papers = [
#                         p for p in st.session_state.indexed_papers
#                         if p["paper_id"] != paper["paper_id"]
#                     ]
#                     retriever.reload_bm25()
#                     st.rerun()

#     st.markdown("---")

#     # --- Export ---
#     st.subheader("⬇️ Export Session")
#     if st.button("Export as Markdown", key="export_btn"):
#         md_content = export_session_as_markdown(
#             papers=st.session_state.indexed_papers,
#             qa_history=st.session_state.get("qa_history", []),
#             summaries=st.session_state.get("summaries", {}),
#             comparisons=st.session_state.get("comparisons", [])
#         )
#         st.download_button(
#             label="⬇️ Download research_brief.md",
#             data=md_content,
#             file_name="research_brief.md",
#             mime="text/markdown",
#             key="download_md"
#         )

# # -------------------------------------------------------------------
# # Main tabs
# # -------------------------------------------------------------------
# st.title("📚 Research Paper Q&A")

# if not st.session_state.indexed_papers:
#     st.info(
#         "👈 Get started by fetching papers from ArXiv or uploading your own PDFs in the sidebar."
#     )
# else:
#     tab1, tab2 = st.tabs(["💬 Chat", "📄 Summarize & Compare"])

#     with tab1:
#         render_chat_tab(retriever, st.session_state.indexed_papers)

#     with tab2:
#         render_summarize_compare_tab(retriever, st.session_state.indexed_papers)





# """
# app.py — Phase 1 entry point
# Streamlit UI for ingesting papers (ArXiv keyword/ID or PDF upload)
# and inspecting the indexed collection.

# Run with:  streamlit run app.py
# """

# import logging
# import streamlit as st
# from dotenv import load_dotenv

# load_dotenv()
# logging.basicConfig(level=logging.INFO)

# from ingestion import ingest_arxiv_keyword, ingest_arxiv_id, ingest_uploaded_pdf
# from ingestion.indexer import get_indexer

# # ──────────────────────────────────────────────
# # Page config
# # ──────────────────────────────────────────────
# st.set_page_config(
#     page_title="Research Paper RAG",
#     page_icon="📚",
#     layout="wide",
# )

# st.title("📚 Research Paper RAG — Phase 1: Ingestion")
# st.caption("Ingest papers from ArXiv or upload PDFs to build your vector index.")

# # ──────────────────────────────────────────────
# # Sidebar — ingestion controls
# # ──────────────────────────────────────────────
# with st.sidebar:
#     st.header("➕ Add Papers")

#     ingestion_mode = st.radio(
#         "Source",
#         ["ArXiv Keyword Search", "ArXiv Paper ID / URL", "Upload PDF"],
#         index=0,
#     )

#     st.divider()

#     if ingestion_mode == "ArXiv Keyword Search":
#         query = st.text_input("Search query", placeholder="e.g. retrieval augmented generation")
#         max_results = st.slider("Max papers to fetch", 1, 10, 5)
#         fetch_btn = st.button("🔍 Fetch & Index", use_container_width=True, type="primary")

#         if fetch_btn and query:
#             with st.spinner(f"Fetching up to {max_results} papers for: '{query}'…"):
#                 try:
#                     results = ingest_arxiv_keyword(query, max_results=max_results)
#                     st.success(f"✅ Ingested {len(results)} papers!")
#                     for r in results:
#                         st.write(f"• **{r['metadata']['title']}** — {r['new_chunks_added']} new chunks")
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Error: {e}")
#         elif fetch_btn:
#             st.warning("Please enter a search query.")

#     elif ingestion_mode == "ArXiv Paper ID / URL":
#         arxiv_input = st.text_input(
#             "ArXiv ID or URL",
#             placeholder="e.g. 2005.11401 or https://arxiv.org/abs/2005.11401",
#         )
#         fetch_btn = st.button("📥 Fetch & Index", use_container_width=True, type="primary")

#         if fetch_btn and arxiv_input:
#             with st.spinner("Fetching paper…"):
#                 try:
#                     result = ingest_arxiv_id(arxiv_input)
#                     st.success(f"✅ Indexed: {result['metadata']['title']}")
#                     st.write(f"New chunks added: **{result['new_chunks_added']}**")
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Error: {e}")
#         elif fetch_btn:
#             st.warning("Please enter an ArXiv ID or URL.")

#     else:  # Upload PDF
#         uploaded_files = st.file_uploader(
#             "Upload PDF(s)",
#             type=["pdf"],
#             accept_multiple_files=True,
#         )

#         with st.expander("Optional metadata (applied to all uploads)"):
#             manual_title = st.text_input("Title override", placeholder="Leave blank to auto-detect")
#             manual_authors = st.text_input("Authors (comma-separated)", placeholder="Leave blank to use 'Unknown'")
#             manual_year = st.number_input("Year", min_value=1900, max_value=2100, value=2024, step=1)

#         upload_btn = st.button("📤 Upload & Index", use_container_width=True, type="primary")

#         if upload_btn and uploaded_files:
#             authors_list = [a.strip() for a in manual_authors.split(",") if a.strip()] if manual_authors else None
#             year_val = int(manual_year) if manual_year else None

#             progress = st.progress(0)
#             for i, uploaded_file in enumerate(uploaded_files):
#                 with st.spinner(f"Processing {uploaded_file.name}…"):
#                     try:
#                         result = ingest_uploaded_pdf(
#                             pdf_bytes=uploaded_file.read(),
#                             filename=uploaded_file.name,
#                             title=manual_title or None,
#                             authors=authors_list,
#                             year=year_val,
#                         )
#                         st.success(f"✅ {uploaded_file.name} — {result['new_chunks_added']} new chunks")
#                     except Exception as e:
#                         st.error(f"Error processing {uploaded_file.name}: {e}")
#                 progress.progress((i + 1) / len(uploaded_files))
#             st.rerun()
#         elif upload_btn:
#             st.warning("Please upload at least one PDF.")

#     st.divider()

#     # ── Danger zone ──
#     with st.expander("⚠️ Danger Zone"):
#         if st.button("🗑️ Reset Entire Index", use_container_width=True):
#             get_indexer().reset()
#             st.success("Index cleared.")
#             st.rerun()

# # ──────────────────────────────────────────────
# # Main panel — indexed papers
# # ──────────────────────────────────────────────
# indexer = get_indexer()
# stats = indexer.get_collection_stats()
# papers = indexer.list_papers()

# col1, col2, col3 = st.columns(3)
# col1.metric("Papers Indexed", stats["papers"])
# col2.metric("Total Chunks", stats["total_chunks"])
# col3.metric("BM25 Entries", stats["bm25_entries"])

# st.divider()

# if not papers:
#     st.info(
#         "No papers indexed yet. Use the sidebar to add papers from ArXiv or upload PDFs.",
#         icon="📭",
#     )
# else:
#     st.subheader(f"📄 Indexed Papers ({len(papers)})")

#     for paper in papers:
#         with st.container(border=True):
#             left, right = st.columns([5, 1])
#             with left:
#                 title = paper.get("paper_title") or "Untitled"
#                 source_badge = "🌐 ArXiv" if paper.get("source") == "arxiv" else "📎 Upload"
#                 st.markdown(f"**{title}** &nbsp; {source_badge}")

#                 authors = paper.get("authors", "Unknown")
#                 year = paper.get("year", "")
#                 arxiv_id = paper.get("arxiv_id", "")
#                 meta_line = f"*{authors}*"
#                 if year:
#                     meta_line += f" · {year}"
#                 if arxiv_id:
#                     meta_line += f" · [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"
#                 st.caption(meta_line)
#                 st.caption(f"🔖 {paper.get('chunk_count', 0)} chunks indexed")

#             with right:
#                 if st.button("🗑️", key=f"del_{paper['paper_id']}", help="Remove this paper"):
#                     deleted = indexer.delete_paper(paper["paper_id"])
#                     indexer.save_bm25()
#                     st.success(f"Removed {deleted} chunks.")
#                     st.rerun()

# st.divider()
# st.caption(
#     "Phase 1 complete ✅ — Once you have papers indexed, Phase 2 will add Q&A, "
#     "summarization, and hybrid retrieval on top of this index."
# )


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

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

# ──────────────────────────────────────────────
# Sidebar — ingestion + paper list
# ──────────────────────────────────────────────
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
        if st.button(" Fetch & Index", use_container_width=True, type="primary"):
            if query:
                with st.spinner(f"Fetching papers for '{query}'…"):
                    try:
                        results = ingest_arxiv_keyword(query, max_results=max_results)
                        st.success(f" Ingested {len(results)} paper(s)")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                st.warning("Enter a search query.")

    elif ingestion_mode == "ArXiv ID / URL":
        arxiv_input = st.text_input("ArXiv ID or URL", placeholder="2005.11401")
        if st.button(" Fetch & Index", use_container_width=True, type="primary"):
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
            with st.container():
                title = paper.get("paper_title", "Untitled")
                year = paper.get("year", "")
                source_icon = "🌐" if paper.get("source") == "arxiv" else "📎"
                left, right = st.columns([5, 1])
                left.caption(f"{source_icon} **{title[:45]}{'…' if len(title) > 45 else ''}** ({year})")
                if right.button("✕", key=f"rm_{paper['paper_id']}", help="Remove"):
                    indexer.delete_paper(paper["paper_id"])
                    indexer.save_bm25()
                    st.rerun()

    st.divider()
    with st.expander("⚠️ Reset index"):
        if st.button(" Clear everything", use_container_width=True):
            indexer.reset()
            for key in ["chat_history", "chat_sources", "summaries", "comparisons"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ──────────────────────────────────────────────
# Main panel — two tabs
# ──────────────────────────────────────────────
tab_chat, tab_sum = st.tabs(["💬 Chat", "📝 Summarize & Compare"])

with tab_chat:
    render_chat(papers)

with tab_sum:
    render_summarize(papers)

