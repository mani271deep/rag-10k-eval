"""FastAPI backend for the 10-K RAG pipeline."""
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generator import answer

app = FastAPI(title="10-K RAG API", version="1.0")

# CORS so the Streamlit frontend (different origin) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_RESULTS = os.path.join("eval", "results.json")


class QueryRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/")
def root():
    return {"status": "ok", "service": "10-K RAG API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query")
def query(req: QueryRequest):
    out = answer(req.query, k=req.k)
    # trim chunk text for transport; keep enough to display
    chunks = [
        {
            "chunk_id": c["chunk_id"],
            "company": c["company"],
            "section_label": c["section_label"],
            "score": round(c["score"], 4),
            "text": c["text"],
        }
        for c in out["chunks"]
    ]
    return {
        "query": out["query"],
        "answer": out["answer"],
        "chunks": chunks,
    }


@app.get("/eval")
def eval_summary():
    """Return the precomputed eval results for display in the dashboard."""
    if not os.path.exists(EVAL_RESULTS):
        return {"available": False}
    results = json.load(open(EVAL_RESULTS))
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    hit_rate = sum(r["hit@k"] for r in answerable) / len(answerable)
    mrr = sum(r["reciprocal_rank"] for r in answerable) / len(answerable)
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_correct = sum(r["correctness"] for r in results) / len(results)
    abstain = sum(1 for r in unanswerable if r["correctness"] >= 4)
    return {
        "available": True,
        "n_questions": len(results),
        "hit_at_k": round(hit_rate, 3),
        "mrr": round(mrr, 3),
        "faithfulness": round(avg_faith, 2),
        "correctness": round(avg_correct, 2),
        "abstention": f"{abstain}/{len(unanswerable)}",
        "results": results,
    }
