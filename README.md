
# Research Paper RAG

A RAG-powered research assistant that lets you query across multiple academic papers without reading them all. Supports ArXiv ingestion, PDF uploads, cross-paper Q&A and summarization.


https://github.com/user-attachments/assets/b3aead0f-918c-46f8-96e8-58a63ded8794

---

## Architecture

```
User Query
    │
    ▼
Hybrid Retrieval (BM25 + Semantic)
    │
    ├── ChromaDB (sentence-transformers embeddings)
    └── BM25 keyword index (rank_bm25)
    │
    ▼
LangChain Chain (query-type-specific prompt)
    │
    ▼
Groq LLaMA 3 → Cited Answer
```

Ingestion pipeline:

```
ArXiv API / PDF Upload
    │
    ▼
PyMuPDF text extraction + section detection
    │
    ▼
Hierarchical Chunker (L1: abstract, L2: section, L3: paragraph)
    │
    ▼
SentenceTransformer embeddings → ChromaDB (persistent)
    +
BM25 index → disk (bm25_index.pkl)
```

---

## Features

- **Dual ingestion**: ArXiv keyword search, direct paper ID/URL, or PDF upload
- **Hierarchical chunking**: Abstract → section → paragraph, with 50-token overlap
- **Hybrid retrieval**: BM25 + semantic search fused with Reciprocal Rank Fusion
- **Query routing**: Auto-detect factual Q&A, summarization, comparison
- **Cross-paper Q&A**: Synthesize information across multiple papers
- **Source citation**: Every answer cites `[Paper Title, Section]`
- **Conversational memory**: Follow-up questions work naturally
- **Export**: Research briefs as Markdown or PDF

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourname/research-paper-rag.git
cd research-paper-rag

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 5. Run the app
streamlit run app.py
```

---

## Tech Stack

| Category | Tool |
|---|---|
| LLM | LLaMA 3 via Groq API |
| LLM framework | LangChain |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector DB | ChromaDB (persistent) |
| Keyword search | rank_bm25 |
| PDF parsing | PyMuPDF |
| ArXiv data | arxiv (pip) |
| UI | Streamlit |
| Export | fpdf2 |
| Language | Python 3.10+ |

---

## Project Structure

```
research-paper-rag/
├── app.py                  # Streamlit entry point
├── ingestion/
│   ├── __init__.py         # Public pipeline functions
│   ├── arxiv_fetcher.py    # ArXiv API fetch + download
│   ├── pdf_parser.py       # PyMuPDF extraction + section detection
│   ├── chunker.py          # Hierarchical chunking (L1/L2/L3)
│   └── indexer.py          # ChromaDB + BM25 indexing
├── retrieval/
│   ├── hybrid_retriever.py # BM25 + semantic + RRF fusion 
├── generation/
│   ├── prompts.py          # Prompt templates 
│   └── chains.py           # LangChain chains 
├── ui/
│   ├── chat_tab.py         # Conversational Q&A 
│   ├── summarize_tab.py    # Per-paper summarization 
├── utils/
│   └── export.py           # Markdown/PDF export 
├── chroma_db/              # Persisted vector store (gitignored)
├── .env
├── requirements.txt
└── README.md
```

---
