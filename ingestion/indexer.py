"""
Embeds chunks with sentence-transformers (all-MiniLM-L6-v2),
stores them in a persistent ChromaDB collection,
and maintains a parallel BM25 keyword index serialised to disk.

Both indexes survive app restarts and are queried together in Phase 2.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# Defaults
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = "research_papers"
HF_MODEL = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BM25_INDEX_PATH = os.path.join(CHROMA_PERSIST_DIR, "bm25_index.pkl")
BM25_META_PATH = os.path.join(CHROMA_PERSIST_DIR, "bm25_meta.json")

# ChromaDB metadata fields that must be strings (lists get JSON-encoded)
LIST_METADATA_FIELDS = {"authors", "categories"}

# Embedding batch size (MiniLM is small; 128 is fine on CPU)
EMBED_BATCH_SIZE = 128


def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str/int/float/bool in metadata.
    Convert lists to JSON strings and None to empty string.
    """
    sanitized = {}
    for k, v in meta.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, list):
            sanitized[k] = json.dumps(v)
        elif isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()



# Indexer class
class PaperIndexer:
    """
    Manages both ChromaDB (semantic) and BM25 (keyword) indexes.

    Usage:
        indexer = PaperIndexer()
        indexer.add_chunks(chunks)          # list of dicts from chunker.py
        indexer.save_bm25()                 # persist BM25 to disk
    """

    def __init__(
        self,
        chroma_persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION,
        model_name: str = HF_MODEL,
    ):
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.model_name = model_name

        Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)

        # ── Embedding model ──
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        # ── ChromaDB ──
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' has %d existing chunks",
            collection_name,
            self.collection.count(),
        )

        # ── BM25: load from disk if exists, else start fresh ──
        self._bm25_corpus: list[list[str]] = []   # tokenized texts
        self._bm25_chunk_ids: list[str] = []       # parallel list of chunk IDs
        self._bm25_index: Optional[BM25Okapi] = None
        self._load_bm25()


    # BM25 persistence
    def _load_bm25(self):
        if Path(BM25_INDEX_PATH).exists() and Path(BM25_META_PATH).exists():
            try:
                with open(BM25_INDEX_PATH, "rb") as f:
                    self._bm25_index = pickle.load(f)
                with open(BM25_META_PATH, "r") as f:
                    meta = json.load(f)
                    self._bm25_corpus = meta["corpus"]
                    self._bm25_chunk_ids = meta["chunk_ids"]
                logger.info("Loaded BM25 index with %d entries", len(self._bm25_chunk_ids))
            except Exception as exc:
                logger.warning("Could not load BM25 index: %s — starting fresh", exc)
                self._bm25_corpus = []
                self._bm25_chunk_ids = []
                self._bm25_index = None
        else:
            logger.info("No existing BM25 index found — will build from scratch")

    def save_bm25(self):
        """Rebuild and persist the BM25 index to disk."""
        if not self._bm25_corpus:
            logger.warning("BM25 corpus is empty — nothing to save")
            return

        logger.info("Building BM25 index over %d entries…", len(self._bm25_corpus))
        self._bm25_index = BM25Okapi(self._bm25_corpus)

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self._bm25_index, f)
        with open(BM25_META_PATH, "w") as f:
            json.dump(
                {"corpus": self._bm25_corpus, "chunk_ids": self._bm25_chunk_ids},
                f,
            )
        logger.info("BM25 index saved to %s", BM25_INDEX_PATH)


    # Indexing
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        embeddings = self.model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(texts) > 20,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def _get_existing_chunk_ids(self) -> set[str]:
        """Return the set of chunk IDs already stored in ChromaDB."""
        try:
            result = self.collection.get(include=[])
            return set(result["ids"])
        except Exception:
            return set()

    def add_chunks(self, chunks: list[dict], skip_duplicates: bool = True) -> int:
        """
        Embed chunks and add them to ChromaDB + BM25 corpus.

        Args:
            chunks:           List of chunk dicts from chunker.chunk_document()
            skip_duplicates:  If True, chunks whose chunk_id already exists are skipped

        Returns:
            Number of new chunks actually added
        """
        if not chunks:
            logger.warning("No chunks provided to add_chunks()")
            return 0

        existing_ids = self._get_existing_chunk_ids() if skip_duplicates else set()

        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            logger.info("All %d chunks already indexed — nothing to add", len(chunks))
            return 0

        logger.info("Embedding %d new chunks (skipping %d duplicates)…", len(new_chunks), len(chunks) - len(new_chunks))

        texts = [c["text"] for c in new_chunks]
        embeddings = self._embed(texts)

        #  Add to ChromaDB in batches of 500 
        batch_size = 500
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [c["chunk_id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = []
            for c in batch:
                meta = {k: v for k, v in c.items() if k not in {"chunk_id", "text"}}
                metadatas.append(_sanitize_metadata(meta))

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        # ── Add to BM25 corpus ──
        for c in new_chunks:
            self._bm25_corpus.append(_tokenize_for_bm25(c["text"]))
            self._bm25_chunk_ids.append(c["chunk_id"])

        logger.info(
            "Added %d chunks to ChromaDB. Collection total: %d",
            len(new_chunks),
            self.collection.count(),
        )
        return len(new_chunks)


    # Inspection helpers
    def list_papers(self) -> list[dict]:
        """
        Return a deduplicated list of indexed papers with basic stats.
        Includes abstract text sourced from the L1 abstract chunk.
        """
        try:
            result = self.collection.get(include=["metadatas", "documents"])
        except Exception:
            return []

        seen: dict[str, dict] = {}
        for meta, doc in zip(result["metadatas"], result["documents"]):
            pid = meta.get("paper_id", "")
            if pid not in seen:
                seen[pid] = {
                    "paper_id": pid,
                    "paper_title": meta.get("paper_title", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "source": meta.get("source", ""),
                    "arxiv_id": meta.get("arxiv_id", ""),
                    "abstract": "",
                    "chunk_count": 0,
                }
            seen[pid]["chunk_count"] += 1
            # Grab text from the L1 abstract chunk (chunk_level == 1)
            if meta.get("chunk_level") == 1 and not seen[pid]["abstract"]:
                seen[pid]["abstract"] = doc

        return list(seen.values())
    # def list_papers(self) -> list[dict]:
    #     """
    #     Return a deduplicated list of indexed papers with basic stats.
    #     """
    #     try:
    #         result = self.collection.get(include=["metadatas"])
    #     except Exception:
    #         return []

    #     seen: dict[str, dict] = {}
    #     for meta in result["metadatas"]:
    #         pid = meta.get("paper_id", "")
    #         if pid not in seen:
    #             seen[pid] = {
    #                 "paper_id": pid,
    #                 "paper_title": meta.get("paper_title", ""),
    #                 "authors": meta.get("authors", ""),
    #                 "year": meta.get("year", ""),
    #                 "source": meta.get("source", ""),
    #                 "arxiv_id": meta.get("arxiv_id", ""),
    #                 "chunk_count": 0,
    #             }
    #         seen[pid]["chunk_count"] += 1

    #     return list(seen.values())

    def get_collection_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "bm25_entries": len(self._bm25_chunk_ids),
            "papers": len(self.list_papers()),
        }

    def delete_paper(self, paper_id: str) -> int:
        """Remove all chunks for a given paper_id from ChromaDB and BM25 corpus."""
        result = self.collection.get(where={"paper_id": paper_id}, include=["metadatas"])
        ids_to_delete = result["ids"]
        if not ids_to_delete:
            logger.warning("No chunks found for paper_id: %s", paper_id)
            return 0

        self.collection.delete(ids=ids_to_delete)
        delete_set = set(ids_to_delete)

        # Rebuild BM25 corpus without deleted entries
        new_corpus = []
        new_ids = []
        for tok, cid in zip(self._bm25_corpus, self._bm25_chunk_ids):
            if cid not in delete_set:
                new_corpus.append(tok)
                new_ids.append(cid)
        self._bm25_corpus = new_corpus
        self._bm25_chunk_ids = new_ids
        self._bm25_index = None  # invalidate; will be rebuilt on next save_bm25()

        logger.info("Deleted %d chunks for paper_id: %s", len(ids_to_delete), paper_id)
        return len(ids_to_delete)

    def reset(self):
        """Wipe the entire index (ChromaDB + BM25). Useful for testing."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_corpus = []
        self._bm25_chunk_ids = []
        self._bm25_index = None
        for path in [BM25_INDEX_PATH, BM25_META_PATH]:
            if Path(path).exists():
                Path(path).unlink()
        logger.info("Index reset complete")


# Module-level singleton (used by Streamlit app)
_indexer_instance: Optional[PaperIndexer] = None


def get_indexer() -> PaperIndexer:
    """Return a cached PaperIndexer singleton (safe for Streamlit's multi-run model)."""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = PaperIndexer()
    return _indexer_instance


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    indexer = PaperIndexer()
    print("Stats before:", indexer.get_collection_stats())

    sample_chunks = [
        {
            "chunk_id": "test-chunk-001",
            "chunk_level": 1,
            "chunk_index": 0,
            "section_key": "abstract",
            "section_label": "Abstract",
            "text": "This is a test abstract for a paper about retrieval augmented generation.",
            "approx_tokens": 15,
            "paper_id": "test_paper_001",
            "paper_title": "Test Paper on RAG",
            "authors": "Alice, Bob",
            "year": 2024,
            "source": "arxiv",
            "arxiv_id": "0000.00001",
        }
    ]

    added = indexer.add_chunks(sample_chunks)
    indexer.save_bm25()
    print(f"Added {added} chunks")
    print("Stats after:", indexer.get_collection_stats())
    print("Papers:", indexer.list_papers())




