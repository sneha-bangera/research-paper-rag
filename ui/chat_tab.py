"""
Conversational Q&A tab with streaming responses, source citation panel,
and in-session memory.
"""

import streamlit as st
from generation.chains import stream_qa, score_confidence

# Score → (label, colour, tooltip)
SCORE_META = {
    5: ("★★★★★ Excellent", "green",  "Answer is fully supported by the retrieved context."),
    4: ("★★★★☆ Good",      "blue",   "Answer is mostly supported; minor gaps."),
    3: ("★★★☆☆ Partial",   "orange", "Answer is partially supported by the context."),
    2: ("★★☆☆☆ Weak",      "red",    "Answer is only loosely related to the context."),
    1: ("★☆☆☆☆ Low",       "red",    "Context had little relevant information."),
    0: ("Score N/A",        "gray",   "Could not compute a confidence score."),
}

def _confidence_badge(score: int):
    """Render a small coloured badge showing the LLM self-evaluation score."""
    label, colour, tooltip = SCORE_META.get(score, SCORE_META[0])
    st.markdown(
        f'<span title="{tooltip}" style="'
        f'background:{colour};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600;">'
        f'Context confidence: {label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"*{tooltip}*")


def _citation_panel(chunks: list[dict]):
    """Render an expandable panel showing the retrieved source chunks."""
    with st.expander(f"📎 Sources ({len(chunks)} chunks retrieved)", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("paper_title", "Unknown")
            section = chunk.get("section_label", "")
            year = chunk.get("year", "")
            rrf = chunk.get("rrf_score", 0)
            st.markdown(f"**[{i}] {title}** ({year}) · *{section}* · RRF score: `{rrf}`")
            st.caption(chunk.get("text", "")[:400] + ("…" if len(chunk.get("text", "")) > 400 else ""))
            if i < len(chunks):
                st.divider()


def _onboarding_screen():
    st.markdown("---")
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            "<div style='text-align:center; padding: 2rem 0;'>"
            "<div style='font-size:3rem'>📭</div>"
            "<h3 style='margin-top:0.5rem'>No papers indexed yet</h3>"
            "<p style='color:gray'>Add papers from the sidebar to start chatting.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Ways to add papers:**")
        st.markdown("- **ArXiv Keyword** — search by topic, e.g. *retrieval augmented generation*")
        st.markdown("- **ArXiv ID / URL** — paste `2005.11401` or a full arxiv.org link")
        st.markdown("- **Upload PDF** — use papers you already have on disk")
        st.info(
            "💡 Run `python demo_ingest.py` to auto-index 5 foundational RAG papers in one command.",
            icon="⚡",
        )


def render(papers: list[dict]):
    """
    Render the Chat tab.

    Args:
        papers: List of indexed paper dicts (for optional paper-scoping filter)
    """
    st.subheader("Chat with Your Papers")

    if not papers:
        _onboarding_screen()
        return

    # ── Paper scope filter ──
    with st.expander("Scope to specific papers (optional)", expanded=False):
        paper_options = {p["paper_title"]: p["paper_id"] for p in papers}
        selected_titles = st.multiselect(
            "Restrict answers to these papers (leave empty to search all):",
            options=list(paper_options.keys()),
        )
        scoped_paper_ids = [paper_options[t] for t in selected_titles] if selected_titles else None

    # ── Session state ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = {}  # message_index -> chunks
    if "chat_scores" not in st.session_state:
        st.session_state.chat_scores = {} 


    # ── Render message history ──
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if i in st.session_state.chat_scores:
                    _confidence_badge(st.session_state.chat_scores[i])
                if i in st.session_state.chat_sources:
                    _citation_panel(st.session_state.chat_sources[i])

    # for i, msg in enumerate(st.session_state.chat_history):
    #     with st.chat_message(msg["role"]):
    #         st.markdown(msg["content"])
    #         # Show sources below assistant messages
    #         if msg["role"] == "assistant" and i in st.session_state.chat_sources:
    #             _citation_panel(st.session_state.chat_sources[i])

    #  Chat input 
    query = st.chat_input("Ask a question about your papers…")

    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            token_gen, chunks = stream_qa(
                query=query,
                chat_history=st.session_state.chat_history[:-1],  # exclude current query
                paper_ids=scoped_paper_ids,
            )

            for token in token_gen:
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            # ── Confidence score (fast follow-up call) ──
            score = 0
            if chunks:
                with st.spinner("Evaluating answer confidence…"):
                    score = score_confidence(query, full_response, chunks)
            _confidence_badge(score)

            # Citation panel immediately below the answer
            if chunks:
                _citation_panel(chunks)

        # Save to history
        assistant_msg_index = len(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.session_state.chat_scores[assistant_msg_index] = score
        if chunks:
            st.session_state.chat_sources[assistant_msg_index] = chunks


    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.chat_sources = {}
            st.session_state.chat_scores = {}
            st.rerun()
