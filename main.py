import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from retriever import Retriever
from llm import ask_groq

load_dotenv()

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Personal Knowledge Assistant",
    description="A RAG-powered assistant that answers questions from your documents.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load retriever once at startup ───────────────────────────────
retriever = Retriever()


# ── Request / Response models ────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k:    int = 5       # how many chunks to retrieve

class IngestRequest(BaseModel):
    sources: list[str]      # list of file paths or URLs

class SourceResult(BaseModel):
    rank:    int
    score:   float
    preview: str

class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[SourceResult]

class IngestResponse(BaseModel):
    message:      str
    sources_done: list[str]
    sources_failed: list[str]


# ── Routes ───────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if the server is running."""
    return {
        "status": "ok",
        "chunks_in_index": retriever.index.ntotal
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a question and get an answer from your knowledge base.

    Example request body:
    {
        "question": "What is artificial intelligence?",
        "top_k": 5
    }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Retrieve relevant chunks
        chunks = retriever.retrieve(request.question, top_k=request.top_k)

        # Ask Groq with retrieved context
        result = ask_groq(request.question, chunks)

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=[SourceResult(**s) for s in result["sources"]]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Add new documents to the knowledge base.

    Example request body:
    {
        "sources": [
            "my_notes.pdf",
            "https://en.wikipedia.org/wiki/Machine_learning"
        ]
    }
    """
    if not request.sources:
        raise HTTPException(status_code=400, detail="No sources provided.")

    from ingest import ingest as run_ingest

    sources_done   = []
    sources_failed = []

    for source in request.sources:
        try:
            run_ingest([source])
            sources_done.append(source)
        except Exception as e:
            sources_failed.append(f"{source} — {str(e)}")

    # Reload retriever index after ingestion
    from retriever import load_index
    retriever.index, retriever.chunks = load_index()

    return IngestResponse(
        message=f"Ingested {len(sources_done)} source(s) successfully.",
        sources_done=sources_done,
        sources_failed=sources_failed
    )


# ── Run server ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
