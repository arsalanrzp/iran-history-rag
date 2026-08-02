# 1_ingest.py
import wikipediaapi
import chromadb
from sentence_transformers import SentenceTransformer

# ── 1. Articles to fetch ──────────────────────────────────────────────────────
ARTICLES = [
    "History of Iran",
    "Achaemenid Empire",
    "Sasanian Empire",
    "Safavid dynasty",
    "Qajar dynasty",
    "Pahlavi dynasty",
    "Iranian Revolution",
    "Iran–Iraq War",
]


# ── 2. Fetch from Wikipedia ───────────────────────────────────────────────────
def fetch_articles(titles):
    wiki = wikipediaapi.Wikipedia(
        language="en", user_agent="IranRAG/1.0 (learning project)"
    )
    corpus = {}
    for title in titles:
        page = wiki.page(title)
        if page.exists():
            corpus[title] = page.text
            print(f"✓ Fetched: {title} ({len(page.text):,} chars)")
        else:
            print(f"✗ Not found: {title}")
    return corpus


# ── 3. Chunking ───────────────────────────────────────────────────────────────
# WHY chunking matters:
# - Too large → retrieved chunk has too much noise, LLM gets confused
# - Too small → chunk lacks enough context to be useful
# - Overlap → ensures a sentence at a chunk boundary isn't lost


def chunk_text(text, chunk_size=400, overlap=80):
    """
    Split text into word-based chunks with overlap.
    chunk_size=400 words ≈ ~500-550 tokens (safe for most embedding models)
    overlap=80 words → repeated at start of next chunk for continuity
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, keeping overlap
    return chunks


# ── 4. ChromaDB setup ─────────────────────────────────────────────────────────
# PersistentClient saves everything to ./chroma_db/ on disk
# Next time you run 2_query.py, it reads from there — no re-embedding needed


def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    # get_or_create → safe to re-run this script without duplicating data
    collection = client.get_or_create_collection(
        name="iran_history",
        metadata={"hnsw:space": "cosine"},  # use cosine similarity for search
    )
    return collection


# ── 5. Embed + Store ──────────────────────────────────────────────────────────
def ingest(corpus, collection, model):
    all_chunks, all_ids, all_metadata = [], [], []

    for article_title, text in corpus.items():
        chunks = chunk_text(text)
        print(f"  → {article_title}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{article_title}__chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append(
                {
                    "source": article_title,  # we can filter by this later
                    "chunk_index": i,
                }
            )

    print(f"\nEmbedding {len(all_chunks)} chunks... (this takes ~1-2 min)")
    embeddings = model.encode(
        all_chunks, normalize_embeddings=True, show_progress_bar=True
    )

    # ChromaDB expects lists, not numpy arrays
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        ids=all_ids,
        metadatas=all_metadata,
    )
    print(f"\n✅ Stored {len(all_chunks)} chunks in ChromaDB.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nFetching Wikipedia articles...")
    corpus = fetch_articles(ARTICLES)

    collection = get_collection()

    # Check if already ingested (avoid duplicates)
    if collection.count() > 0:
        print(
            f"\nCollection already has {collection.count()} chunks. Skipping ingestion."
        )
        print("Delete ./chroma_db/ folder to re-ingest.")
    else:
        ingest(corpus, collection, model)
