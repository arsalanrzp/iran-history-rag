# 1_ingest.py
import wikipediaapi
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle


# Wikipedia article titles to ingest
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


# Split text into overlapping chunks for retrieval
def chunk_text(text, chunk_size=400, overlap=80):
    """Split text into word-based chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, keeping overlap
    return chunks


# Create or open the ChromaDB collection used for retrieval


def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    # get_or_create → safe to re-run this script without duplicating data
    collection = client.get_or_create_collection(
        name="iran_history",
        metadata={"hnsw:space": "cosine"},  # use cosine similarity for search
    )
    return collection


# Build BM25 index for keyword search


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


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

    # Build BM25 index for keyword search
    print("\nBuilding BM25 index...")
    bm25 = build_bm25_index(all_chunks)

    # Persist it — BM25Okapi doesn't save to ChromaDB, so pickle it separately
    with open("./chroma_db/bm25_index.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": all_chunks,
                "ids": all_ids,
                "sources": [m["source"] for m in all_metadata],
            },
            f,
        )

    print("✅ BM25 index saved.")


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
