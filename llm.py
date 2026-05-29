import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configuration
MODEL      = "llama-3.1-8b-instant"   
MAX_TOKENS = 1024               

# Groq client 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Prompt builder 

def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the prompt sent to Groq.
    Injects retrieved chunks as context above the question.
    """
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Source {chunk['rank']}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful personal knowledge assistant.
You answer questions strictly based on the context provided below.
If the answer is not found in the context, say "I don't have enough information in my knowledge base to answer that."
Do not make up information. Be clear and concise.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt


# LLM caller 

def ask_groq(query: str, chunks: list[dict]) -> dict:
    """
    Send retrieved chunks + question to Groq and return the answer.

    Args:
        query:  the user's question
        chunks: list of retrieved chunks from retriever.py

    Returns:
        dict with keys: 'answer', 'sources'
    """
    if not chunks:
        return {
            "answer":  "No relevant documents found in the knowledge base.",
            "sources": []
        }

    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        messages=[
            {
                "role":    "system",
                "content": "You are a helpful personal knowledge assistant that answers questions based only on the provided context."
            },
            {
                "role":    "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    # Return answer + source texts used
    sources = [
        {
            "rank":    chunk["rank"],
            "score":   chunk["score"],
            "preview": chunk["text"][:200] + "..."
        }
        for chunk in chunks
    ]

    return {
        "answer":  answer,
        "sources": sources
    }


# CLI test

if __name__ == "__main__":
    from retriever import Retriever

    retriever = Retriever()

    print("\n" + "─" * 50)
    print("RAG Assistant — powered by Groq (free tier)")
    print("Type your question. Type 'quit' to exit.")
    print("─" * 50 + "\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        print("\nSearching knowledge base...")
        chunks = retriever.retrieve(query)

        print("Asking Groq...\n")
        result = ask_groq(query, chunks)

        print(f"Assistant: {result['answer']}")
        print("\n── Sources used ──")
        for s in result["sources"]:
            print(f"  [{s['rank']}] score: {s['score']:.4f} | {s['preview']}")
        print()
