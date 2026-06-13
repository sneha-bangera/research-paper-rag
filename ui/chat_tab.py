"""
Conversational Q&A tab with streaming responses, source citation panel,
and in-session memory.
"""

import streamlit as st
from generation.chains import stream_qa


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


def render(papers: list[dict]):
    """
    Render the Chat tab.

    Args:
        papers: List of indexed paper dicts (for optional paper-scoping filter)
    """
    st.subheader("Chat with Your Papers")

    if not papers:
        st.info("No papers indexed yet. Add papers from the sidebar first.", icon="📭")
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

    # ── Render message history ──
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources below assistant messages
            if msg["role"] == "assistant" and i in st.session_state.chat_sources:
                _citation_panel(st.session_state.chat_sources[i])

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

            # Citation panel immediately below the answer
            if chunks:
                _citation_panel(chunks)

        # Save to history
        assistant_msg_index = len(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        if chunks:
            st.session_state.chat_sources[assistant_msg_index] = chunks

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.chat_sources = {}
            st.rerun()
