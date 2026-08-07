# Iran History RAG

A retrieval-augmented generation (RAG) pipeline over Wikipedia articles on Iranian history, built as a portfolio project to explore the core components of a modern RAG system: chunking, hybrid retrieval, cross-encoder reranking, and grounded generation.

## Architecture

```
Wikipedia articles (History of Iran, Achaemenid Empire, Sasanian Empire,
Safavid dynasty, Qajar dynasty, Pahlavi dynasty, Iranian Revolution, Iran–Iraq War)
    ↓
Chunking (400 words per chunk, 80-word overlap)
    ↓
    ├── Bi-encoder embeddings (sentence-transformers: all-MiniLM-L6-v2) → ChromaDB
    └── BM25 lexical index (rank_bm25)
    ↓
Query → dense retrieval (top-20) + BM25 retrieval (top-20)
    ↓
Reciprocal Rank Fusion (RRF) — merge both candidate lists
    ↓
Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2) → top-4
    ↓
Gemini generation (grounded in retrieved context)
    ↓
Answer
```

## Why hybrid retrieval?

Dense (bi-encoder) retrieval is strong on semantic/paraphrased matches but can miss exact terms — specific names, dates, and rare entities the embedding space doesn't weight heavily. BM25, a classic lexical/keyword algorithm, is the opposite: excellent at exact term matches, weak on paraphrasing. Combining both gets the strengths of each, which is a standard pattern in production RAG systems.

The two ranked lists (dense and BM25) are merged with **Reciprocal Rank Fusion** rather than combining raw scores directly, since cosine distances and BM25 scores live on incomparable scales. RRF scores each chunk purely by its rank position in each list (`1 / (k + rank)`, summed across lists), so a chunk that ranks highly in *both* dense and lexical search rises to the top of the merged pool — without needing to normalize or tune weights between two different scoring systems.

## Why two-stage retrieval (retrieval → rerank)?

A bi-encoder embeds the query and each document independently, which makes it fast enough to search across a full corpus, but the lack of direct interaction between query and document limits precision. A cross-encoder jointly encodes the query and a candidate document together, producing a much more accurate relevance score — but it's too slow to run over an entire corpus.

This project combines retrieval and reranking in two stages: hybrid retrieval (dense + BM25, merged via RRF) casts a wide, diverse net, and the cross-encoder reranks that pool down to the top-4 most relevant chunks before they're passed to the LLM. This balances recall (don't miss the right chunk) against precision (don't pass noise to the LLM).

## Project structure

```
.
├── 1_ingest.py          # Fetches Wikipedia articles, chunks, embeds into ChromaDB, builds BM25 index
├── 2_query.py           # Interactive query loop: hybrid retrieve → RRF merge → rerank → generate
├── requirements.txt     # Python dependencies
├── .env.example          # Template for required environment variables
└── .gitignore
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/arsalanrzp/iran-history-rag.git
   cd iran-history-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your-key-here
   ```

5. Build the vector database and BM25 index (only needs to be run once — subsequent runs will skip re-ingestion):
   ```bash
   python 1_ingest.py
   ```

6. Start asking questions:
   ```bash
   python 2_query.py
   ```

## Example usage

```
❓ Your question: What caused the fall of the Sasanian Empire?

📚 Hybrid retrieval (RRF merged, top 10):
  [1] Sasanian__chunk_3 (rrf: 0.0325)
  [2] History__chunk_7 (rrf: 0.0325)
  [3] Achaemenid__chunk_1 (rrf: 0.0159)
  ...

📚 Retrieved chunks: (using cross-encoder reranker)
  [1] Sasanian Empire (relevance: 8.421)
  [2] History of Iran (relevance: 6.203)
  [3] Achaemenid Empire (relevance: 2.114)
  [4] Iranian Revolution (relevance: 1.897)

🤖 Answer:
[Gemini's answer, grounded strictly in the retrieved context above]
```

## Tech stack

| Component            | Choice                                      |
|-----------------------|----------------------------------------------|
| Data source            | Wikipedia (via `wikipedia-api`)               |
| Embedding model        | `all-MiniLM-L6-v2` (sentence-transformers)   |
| Vector store            | ChromaDB (persistent, local)                  |
| Lexical index            | BM25 (`rank_bm25`)                           |
| Retrieval fusion        | Reciprocal Rank Fusion (RRF)                  |
| Reranker                | `cross-encoder/ms-marco-MiniLM-L-6-v2`        |
| Generation model        | Google Gemini                                 |

## Design notes

- **Chunking strategy:** 400-word chunks with 80-word overlap balance context sufficiency against noise. Overlap prevents sentences at chunk boundaries from being split awkwardly.
- **Hybrid retrieval:** dense embeddings and BM25 are built once at ingestion time and queried independently; their ranked outputs are merged via RRF before reranking, so exact-term queries (dates, proper nouns) aren't at the mercy of embedding similarity alone.
- **Idempotent ingestion:** `1_ingest.py` checks whether the collection is already populated before embedding, so it's safe to re-run without duplicating data.
- **Grounded generation:** the prompt explicitly instructs the model to answer only from retrieved context and to say so if the context is insufficient, reducing hallucination.

## Future work

- Query rewriting / expansion for better retrieval on ambiguous questions
- Context compression before generation to reduce token usage
- Evaluation harness (retrieval recall@k / MRR, answer faithfulness, latency benchmarks)
- Swap in a local open-source LLM (via Ollama or HuggingFace) as an alternative to Gemini

## License

MIT