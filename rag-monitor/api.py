import time
import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import groq
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from monitor import log_query, get_metrics
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store
rag_state = {
    "chunks": [],
    "bm25": None,
    "pdf_name": None,
}

groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    content = await file.read()
    reader = PdfReader(io.BytesIO(content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    chunks = chunk_text(text)
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    rag_state["chunks"] = chunks
    rag_state["bm25"] = bm25
    rag_state["pdf_name"] = file.filename

    return {"message": "PDF loaded", "chunks": len(chunks), "filename": file.filename}


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query(req: QueryRequest):
    if not rag_state["chunks"]:
        raise HTTPException(400, "No PDF uploaded yet")

    start = time.time()
    success = True
    answer = ""
    sources = []

    try:
        query_tokens = req.question.lower().split()
        scores = rag_state["bm25"].get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
        sources = [rag_state["chunks"][i][:200] for i in top_indices]
        context = "\n\n---\n\n".join([rag_state["chunks"][i] for i in top_indices])

        prompt = f"""You are a precise document assistant. Answer ONLY from the context below.
Always end your answer with: "Source: {rag_state['pdf_name']}"

Context:
{context}

Question: {req.question}
Answer:"""

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        success = False
        answer = f"Error: {str(e)}"

    latency_ms = (time.time() - start) * 1000
    log_entry = log_query(req.question, answer, latency_ms, sources, success)

    return {
        "answer": answer,
        "latency_ms": round(latency_ms, 2),
        "cost_usd": log_entry["cost_usd"],
        "sources": sources,
        "success": success,
    }


@app.get("/metrics")
async def metrics():
    return get_metrics()


@app.delete("/metrics/reset")
async def reset_metrics():
    import json
    with open("query_logs.json", "w") as f:
        json.dump([], f)
    return {"message": "Logs cleared"}
