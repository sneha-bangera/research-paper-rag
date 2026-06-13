"""
Combines BM25 keyword search and ChromaDB semantic search,
then fuses results using Reciprocal Rank Fusion (RRF).

Returns top-K chunks with their metadata and relevance scores,
ready to be passed as context to the LLM.
"""

import logging
from typing import Optional

from ingestion.indexer import get_indexer

logger = logging.getLogger(__name__)

# Number of candidates to pull from each retriever before fusion
SEMANTIC_TOP_N = 10
BM25_TOP_N = 10
# Final number of chunks to pass to the LLM
FINAL_TOP_K = 5
# RRF constant — 60 is standard; higher = less weight to top ranks
RRF_K = 60


def _reciprocal_rank_fusion(
    semantic_ids: list[str],
    bm25_ids: list[str],
) -> list[tuple[str, float]]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across all lists the doc appears in.
    Returns list of (chunk_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(semantic_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(
    query: str,
    top_k: int = FINAL_TOP_K,
    paper_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Run hybrid retrieval for a query.

    Args:
        query:     The user question or search string
        top_k:     Number of final chunks to return (default 5)
        paper_ids: Optional list of paper_ids to restrict search to

    Returns:
        List of chunk dicts sorted by RRF score descending.
    """
    indexer = get_indexer()

    # ── 1. Semantic search via ChromaDB ──
    query_embedding = indexer.model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    chroma_where = {"paper_id": {"$in": paper_ids}} if paper_ids else None

    total_chunks = indexer.collection.count()
    if total_chunks == 0:
        return []

    chroma_results = indexer.collection.query(
        query_embeddings=query_embedding,
        n_results=min(SEMANTIC_TOP_N, total_chunks),
        where=chroma_where,
        include=["metadatas", "documents", "distances"],
    )

    chunk_lookup: dict[str, dict] = {}
    semantic_ids: list[str] = []

    if chroma_results["ids"] and chroma_results["ids"][0]:
        for cid, doc, meta, dist in zip(
            chroma_results["ids"][0],
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
            chroma_results["distances"][0],
        ):
            semantic_ids.append(cid)
            chunk_lookup[cid] = {"chunk_id": cid, "text": doc, "semantic_distance": dist, **meta}

    # ── 2. BM25 keyword search ──
    bm25_ids: list[str] = []
    if indexer._bm25_index is not None and indexer._bm25_chunk_ids:
        tokenized_query = query.lower().split()
        bm25_scores = indexer._bm25_index.get_scores(tokenized_query)

        scored = sorted(
            ((i, bm25_scores[i]) for i in range(len(indexer._bm25_chunk_ids)) if bm25_scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:BM25_TOP_N]

        for idx, score in scored:
            cid = indexer._bm25_chunk_ids[idx]
            bm25_ids.append(cid)
            if cid not in chunk_lookup:
                try:
                    res = indexer.collection.get(ids=[cid], include=["metadatas", "documents"])
                    if res["ids"]:
                        chunk_lookup[cid] = {"chunk_id": cid, "text": res["documents"][0], "semantic_distance": 1.0, **res["metadatas"][0]}
                except Exception as exc:
                    logger.warning("Could not fetch BM25 chunk %s: %s", cid, exc)

        if paper_ids:
            paper_id_set = set(paper_ids)
            bm25_ids = [cid for cid in bm25_ids if chunk_lookup.get(cid, {}).get("paper_id", "") in paper_id_set]

    logger.info("Retrieval: %d semantic, %d BM25 candidates", len(semantic_ids), len(bm25_ids))

    # ── 3. RRF fusion ──
    fused = _reciprocal_rank_fusion(semantic_ids, bm25_ids)

    # ── 4. Build result list ──
    results: list[dict] = []
    seen: set[str] = set()

    for chunk_id, rrf_score in fused[:top_k]:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            continue
        results.append({
            "chunk_id": chunk_id,
            "text": chunk.get("text", ""),
            "paper_id": chunk.get("paper_id", ""),
            "paper_title": chunk.get("paper_title", "Unknown"),
            "authors": chunk.get("authors", ""),
            "year": chunk.get("year", ""),
            "section_label": chunk.get("section_label", ""),
            "chunk_level": chunk.get("chunk_level", 3),
            "rrf_score": round(rrf_score, 4),
            "semantic_distance": chunk.get("semantic_distance", 1.0),
        })

    logger.info("Returning %d fused chunks for query: %r", len(results), query[:60])
    return results


def retrieve_for_paper(paper_id: str, top_k: int = 8) -> list[dict]:
    """
    Retrieve the most content-rich chunks for a single paper.
    Used for summarization — prefers L1 (abstract) then L2 (sections).
    """
    indexer = get_indexer()
    try:
        result = indexer.collection.get(
            where={"paper_id": paper_id},
            include=["metadatas", "documents"],
        )
    except Exception as exc:
        logger.error("Could not retrieve chunks for paper %s: %s", paper_id, exc)
        return []

    chunks = []
    for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        chunks.append({"chunk_id": cid, "text": doc, **meta})

    # Sort: L1 first, then L2, then L3
    chunks.sort(key=lambda c: (c.get("chunk_level", 3), c.get("chunk_index", 0)))

    priority = [c for c in chunks if c.get("chunk_level", 3) <= 2]
    if len(priority) < top_k:
        extra = [c for c in chunks if c.get("chunk_level", 3) == 3]
        priority += extra[: top_k - len(priority)]

    return priority[:top_k]