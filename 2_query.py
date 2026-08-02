# 2_query.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()  # add this near the top of 2_query.py, before gemini_client is created
# ── 1. Load the persisted DB (instant — no re-embedding) ─────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("iran_history")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Gemini client ──────────────────────────────────────────────────────────
# Set your key: export GEMINI_API_KEY="your-key-here"
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.5-flash"  # or "gemini-2.5-pro" for higher quality

print(f"✅ Loaded collection with {collection.count()} chunks\n")


# ── 2. Retrieval function ─────────────────────────────────────────────────────
def retrieve(query, top_k=4):
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    scores = results["distances"][0]  # cosine distance (lower = more similar)

    return chunks, sources, scores


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

        chunks, sources, scores = retrieve(query, top_k=4)

        # Show what was retrieved (great for debugging)
        print("\n📚 Retrieved chunks:")
        for i, (src, score) in enumerate(zip(sources, scores)):
            print(f"  [{i + 1}] {src} (distance: {score:.3f})")

        prompt = build_prompt(query, chunks, sources)

        print("\n🤖 Answer:")
        answer = generate_answer(prompt)
        print(answer)
        print()


if __name__ == "__main__":
    main()
