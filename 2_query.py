# 2_query.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


load_dotenv()

# ── 1. Load the persisted DB (instant — no re-embedding) ─────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("iran_history")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load once, alongside your bi-encoder model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ── Gemini client ──────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.5-flash"

print(f"✅ Loaded collection with {collection.count()} chunks\n")


# ── 2. Retrieval function ─────────────────────────────────────────────────────
def retrieve(query, top_k=15, rerank_top_n=4):
    """
    Stage 1: bi-encoder retrieves a wider candidate pool (top_k).
    Stage 2: cross-encoder reranks and returns the best rerank_top_n.
    """
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    scores = results["distances"][0]  # cosine distance (lower = more similar)

    print("\n📚 Retrieved chunks: (bi_encoder)")
    for i, (src, score) in enumerate(zip(sources, scores)):
        print(f"  [{i + 1}] {src} (distance: {score:.3f})")

    # ── Cross-encoder reranking ──────────────────────────────────
    pairs = [[query, chunk] for chunk in chunks]
    rerank_scores = reranker.predict(pairs)  # higher = more relevant

    # Sort candidates by rerank score, descending
    ranked = sorted(
        zip(chunks, sources, rerank_scores),
        key=lambda x: x[2],
        reverse=True,
    )[:rerank_top_n]

    reranked_chunks = [r[0] for r in ranked]
    reranked_sources = [r[1] for r in ranked]
    reranked_scores = [r[2] for r in ranked]

    return reranked_chunks, reranked_sources, reranked_scores


# ── 3. Prompt builder ─────────────────────────────────────────────────────────
def build_prompt(query, chunks, sources):
    context_parts = []
    for chunk, source in zip(chunks, sources):
        context_parts.append(f"[Source: {source}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a knowledgeable assistant about Iranian history.
Answer the question using ONLY the context provided below.
If the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


# ── 4. LLM call ────────────────────────────────────────────────────────────
def generate_answer(prompt):
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error calling Gemini API: {e}"


# ── 5. Interactive loop ───────────────────────────────────────────────────────
def main():
    print("Iran History RAG — ask anything! (type 'quit' to exit)\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        chunks, sources, scores = retrieve(query)

        # Show what was retrieved (great for debugging)
        print("\n📚 Retrieved chunks: (using cross-encoder reranker)")
        for i, (src, score) in enumerate(zip(sources, scores)):
            print(f"  [{i + 1}] {src} (relevance: {score:.3f})")

        prompt = build_prompt(query, chunks, sources)

        print("\n🤖 Answer:")
        answer = generate_answer(prompt)
        print(answer)
        print()


if __name__ == "__main__":
    main()
