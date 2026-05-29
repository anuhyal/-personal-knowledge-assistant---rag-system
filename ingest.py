import os
import sys
import pickle
import requests
import fitz  # pymupdf
import faiss
import numpy as np

from docx import Document
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Configuration ────────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, lightweight, 384-dim
CHUNK_SIZE       = 500                   # characters per chunk
CHUNK_OVERLAP    = 50                    # overlap between chunks
INDEX_DIR        = "faiss_index"         # folder where index is saved
INDEX_FILE       = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE      = os.path.join(INDEX_DIR, "chunks.pkl")


# ── Loaders ──────────────────────────────────────────────────────

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def load_docx(file_path: str) -> str:
    """Extract text from a Word document."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def load_txt(file_path: str) -> str:
    """Read a plain text or markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_url(url: str) -> str:
    """Scrape visible text from a web page."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Remove script and style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_document(source: str) -> str:
    """Auto-detect source type and load it."""
    if source.startswith("http://") or source.startswith("https://"):
        print(f"  [URL]   {source}")
        return load_url(source)
    elif source.lower().endswith(".pdf"):
        print(f"  [PDF]   {source}")
        return load_pdf(source)
    elif source.lower().endswith(".docx"):
        print(f"  [DOCX]  {source}")
        return load_docx(source)
    elif source.lower().endswith((".txt", ".md")):
        print(f"  [TXT]   {source}")
        return load_txt(source)
    else:
        raise ValueError(f"Unsupported file type: {source}")


# ── Chunking ─────────────────────────────────────────────────────

def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# ── Embedding ────────────────────────────────────────────────────

def embed_chunks(chunks: list[str], model: SentenceTransformer) -> np.ndarray:
    """Convert list of text chunks to numpy embedding matrix."""
    print(f"  Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


# ── FAISS index ──────────────────────────────────────────────────

def build_or_update_index(
    all_chunks: list[str],
    all_embeddings: np.ndarray
):
    """Build a new FAISS index or add to an existing one, then save to disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = all_embeddings.shape[1]

    # Load existing index if it exists
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("  Existing index found — appending new documents...")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            existing_chunks = pickle.load(f)
        all_chunks = existing_chunks + all_chunks
    else:
        print("  Creating new FAISS index...")
        index = faiss.IndexFlatL2(dim)

    index.add(all_embeddings)
    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"  Index saved — total chunks in store: {index.ntotal}")


# ── Main entry point ─────────────────────────────────────────────

def ingest(sources: list[str]):
    """
    Main ingestion function.

    Args:
        sources: list of file paths or URLs to ingest
    """
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    all_embeddings_list = []

    for source in sources:
        try:
            print(f"\nProcessing: {source}")
            text = load_document(source)

            if not text.strip():
                print(f"  WARNING: No text extracted from {source}, skipping.")
                continue

            chunks = split_into_chunks(text)
            print(f"  Split into {len(chunks)} chunks")

            embeddings = embed_chunks(chunks, model)
            all_chunks.extend(chunks)
            all_embeddings_list.append(embeddings)

        except Exception as e:
            print(f"  ERROR processing {source}: {e}")
            continue

    if not all_chunks:
        print("\nNo documents were successfully ingested. Exiting.")
        return

    all_embeddings = np.vstack(all_embeddings_list)
    print(f"\nBuilding FAISS index with {len(all_chunks)} total chunks...")
    build_or_update_index(all_chunks, all_embeddings)
    print("\nIngestion complete!")


# ── CLI usage ────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_or_url> [<file_or_url> ...]")
        print("\nExamples:")
        print("  python ingest.py my_notes.pdf")
        print("  python ingest.py report.docx notes.txt")
        print("  python ingest.py https://en.wikipedia.org/wiki/Python_(programming_language)")
        sys.exit(1)

    sources = sys.argv[1:]
    ingest(sources)