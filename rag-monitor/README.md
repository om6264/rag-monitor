# RAG Monitor — AI Observability Layer

A production monitoring dashboard built on top of a RAG (Retrieval-Augmented Generation) pipeline. Tracks latency, cost, and query quality in real time.

## Features

- **Latency tracking** — per-query, avg, P50, P95
- **Cost estimation** — per query and cumulative (USD)
- **Query logs** — full history with status, latency, cost
- **Bar charts** — visual latency and cost trends
- **BM25 retrieval** — keyword-based document search
- **Citation enforcement** — answers always cite the source PDF

## Tech Stack

- **Backend**: FastAPI + Python
- **LLM**: Groq (LLaMA 3.1 8B Instant)
- **Retrieval**: BM25 (rank-bm25)
- **PDF parsing**: pypdf
- **Deployment**: Render.com

## Run Locally

```bash
pip install -r requirements.txt
# Create .env with: GROQ_API_KEY=your_key
uvicorn api:app --reload
```

Open http://127.0.0.1:8000

## Deploy on Render

- Runtime: Python 3
- Build: `pip install -r requirements.txt`
- Start: `uvicorn api:app --host 0.0.0.0 --port 10000`
- Env var: `GROQ_API_KEY`
