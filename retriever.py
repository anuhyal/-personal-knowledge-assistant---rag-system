import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # must match ingest.py
INDEX_DIR        = "faiss_index"
INDEX_FILE       = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE      = os.path.join(INDEX_DIR, "chunks.pkl")
TOP_K            = 5                     # number of chunks to retrieve


# ── Load index ───────────────────────────────────────────────────

def load_index():
    """Load FAISS index and chunks from disk."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(
            "FAISS index not found. Please run ingest.py first."
        )
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ── Retriever ────────────────────────────────────────────────────

class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.model  = SentenceTransformer(EMBED_MODEL_NAME)
        print("Loading FAISS index...")
        self.index, self.chunks = load_index()
        print(f"Index loaded — {self.index.ntotal} chunks available")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Find the most relevant chunks for a query.

        Args:
            query:  the user's question
            top_k:  how many chunks to return

        Returns:
            list of dicts with keys: 'text', 'score', 'rank'
        """
        # Embed the query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = query_vector.astype("float32")

        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue  # FAISS returns -1 if not enough results
            results.append({
                "rank":  rank + 1,
                "score": float(dist),
                "text":  self.chunks[idx]
            })

        return results


# ── CLI test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = Retriever()

    print("\n" + "─" * 50)
    print("Retriever test — type a question to search your knowledge base")
    print("Type 'quit' to exit")
    print("─" * 50 + "\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        results = retriever.retrieve(query)

        print(f"\nTop {len(results)} relevant chunks:\n")
        for r in results:
            print(f"  Rank {r['rank']} (score: {r['score']:.4f})")
            print(f"  {r['text'][:300]}...")
            print()