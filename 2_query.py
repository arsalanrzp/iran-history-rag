# 2_query.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import pickle


load_dotenv()

# Load persisted DB and BM25 index
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("iran_history")
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("./chroma_db/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
bm25_chunks = bm25_data["chunks"]
bm25_ids = bm25_data["ids"]
bm25_sources = bm25_data["sources"]


def tokenize(text):
    return text.lower().split()


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Gemini client
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.5-flash"

print(f"✅ Loaded collection with {collection.count()} chunks\n")


def retrieve(query, dense_top_k=20, bm25_top_k=20, rerank_top_n=4, k=60):
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=dense_top_k,
        include=["documents", "metadatas"],
    )
    dense_ids = dense_results["ids"][0]
    dense_chunks = dense_results["documents"][0]
    dense_sources = [m["source"] for m in dense_results["metadatas"][0]]

    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:bm25_top_k]
    bm25_result_ids = [bm25_ids[i] for i in top_bm25_idx]

    lookup = {
        cid: (chunk, src)
        for cid, chunk, src in zip(dense_ids, dense_chunks, dense_sources)
    }
    for i in top_bm25_idx:
        cid = bm25_ids[i]
        lookup.setdefault(cid, (bm25_chunks[i], bm25_sources[i]))

    rrf_scores = {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
    for rank, cid in enumerate(bm25_result_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)

    merged_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    merged_chunks = [lookup[cid][0] for cid in merged_ids]
    merged_sources = [lookup[cid][1] for cid in merged_ids]

    print("\n📚 Hybrid retrieval (RRF merged, top 10):")
    for i, cid in enumerate(merged_ids[:10]):
        print(f"  [{i + 1}] {cid} (rrf: {rrf_scores[cid]:.4f})")

    pairs = [[query, chunk] for chunk in merged_chunks]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(
        zip(merged_chunks, merged_sources, rerank_scores),
        key=lambda x: x[2],
        reverse=True,
    )[:rerank_top_n]

    return [r[0] for r in ranked], [r[1] for r in ranked], [r[2] for r in ranked]


# Prompt builder
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


# LLM call
def generate_answer(prompt):
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error calling Gemini API: {e}"


# Interactive loop
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
