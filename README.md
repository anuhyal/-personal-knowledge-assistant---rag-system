# Personal Knowledge Assistant - RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built from scratch in Python. Feed it your PDFs, Word documents, text files,
and web URLs - then ask it anything. Every answer is grounded in your documents with zero hallucination.


## What it does

- Ingests PDFs, Word documents, TXT, Markdown, and live web URLs
- Chunks and embeds content into a local FAISS vector index
- Answers natural language questions using top-k semantic retrieval
- Returns answers with cited sources — no hallucination
- Exposes a FastAPI REST API with Swagger docs
- Includes a responsive frontend UI with dark/light theme toggle
---

## Architecture

Documents (PDF, DOCX, TXT, URL)
↓

Text extraction
↓

Chunking (500 chars, 50 overlap)
↓

Embeddings (all-MiniLM-L6-v2)
↓

FAISS index (saved to disk)
↓

Query → embed → search → top-5 chunks
↓

Groq LLaMA 3.1 → grounded answer
↓

FastAPI → JSON response + sources

---

## Project structure

rag-assistant/
├── ingest.py          # Document loader, chunker, embedder, FAISS builder

├── retriever.py       # FAISS search and query embedding

├── llm.py             # Groq LLaMA integration and prompt builder

├── main.py            # FastAPI server with /query and /ingest endpoints

├── index.html         # Frontend UI (zero dependencies)

├── requirements.txt   # All Python dependencies

├── .env.example       # Environment variable template

└── faiss_index/       # Auto-created after first ingest

├── index.faiss

└── chunks.pkl

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| API server | FastAPI + Uvicorn |
| Vector store | FAISS (faiss-cpu 1.8.0) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq API — LLaMA 3.1 8B Instant |
| PDF loader | PyMuPDF |
| DOCX loader | python-docx |
| Web scraper | requests + BeautifulSoup4 |
| Text splitter | LangChain RecursiveCharacterTextSplitter |
| Frontend | Vanilla HTML5 / CSS3 / JavaScript |
| Config | python-dotenv + Pydantic v2 |

---

## Getting started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/personal-knowledge-assistant-rag.git
cd personal-knowledge-assistant-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install numpy==1.26.4
pip install faiss-cpu==1.8.0
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux
```
Then open `.env` and add your Groq API key:
GROQ_API_KEY=gsk_your_key_here

Get a free key at **console.groq.com** — no credit card needed.

### 5. Ingest your first document
```bash
# From a URL
python ingest.py https://en.wikipedia.org/wiki/Artificial_intelligence

# From a PDF
python ingest.py my_notes.pdf

# From a Word document
python ingest.py report.docx

# Multiple sources at once
python ingest.py file1.pdf file2.docx https://example.com
```

### 6. Start the API server
```bash
python main.py
```
Server runs at **http://localhost:8000**
API docs at **http://localhost:8000/docs**

### 7. Open the frontend
Double-click `index.html` in your project folder — it opens in your browser and connects to the backend automatically.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status and chunk count |
| POST | `/query` | Ask a question, get an answer |
| POST | `/ingest` | Add new documents to the index |

### Example query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?", "top_k": 5}'
```

### Example response
```json
{
  "question": "What is artificial intelligence?",
  "answer": "Artificial intelligence is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, and problem-solving.",
  "sources": [
    {"rank": 1, "score": 0.54, "preview": "Artificial intelligence (AI) is..."},
    {"rank": 2, "score": 0.61, "preview": "AI founder John McCarthy..."}
  ]
}
```

---

## Key numbers

- 496 chunks indexed from a single Wikipedia article in under 60 seconds
- 384-dimensional dense vector embeddings
- Top-5 semantic retrieval per query
- 6 modular Python files — fully extensible
- 4 document types supported + live web scraping
- Sub-second FAISS retrieval
- Zero frontend dependencies — no npm, no build step

---

## Future improvements

- [ ] Add conversation memory / multi-turn chat history
- [ ] Support for hybrid search (BM25 + vector)
- [ ] Docker containerization
- [ ] User authentication
- [ ] Support for more file types (Excel, PowerPoint, images with OCR)
- [ ] Switch to a reranking model for better retrieval quality
- [ ] Deploy to cloud (AWS / GCP / Azure)

---

## Author

Built by **ANUHYA L **

---
