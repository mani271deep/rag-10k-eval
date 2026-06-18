"""Retrieval module for the 10-K RAG pipeline.

Exposes retrieve(query, k) as a clean, tool-shaped function so it can drop
into an agent's ReAct tool registry unchanged (Project 3 bridge). The
RETRIEVE_TOOL_SCHEMA below is the tool-calling spec to register with the agent.
"""
import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "data/index.faiss"
META_PATH = "data/chunks_meta.json"
EMB_MODEL = "text-embedding-3-small"

# Loaded once at import, reused across calls (efficient for API + eval loops)
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_index = faiss.read_index(INDEX_PATH)
with open(META_PATH, encoding="utf-8") as f:
    _meta = json.load(f)


def _embed(text):
    resp = _client.embeddings.create(model=EMB_MODEL, input=[text])
    v = np.array([resp.data[0].embedding], dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1e-8
    return v / n


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Retrieve the top-k most relevant 10-K chunks for a query.

    Returns a list of dicts, each with chunk metadata plus a similarity
    'score' (cosine, higher = more relevant), in descending score order.
    """
    qvec = _embed(query)
    scores, idxs = _index.search(qvec, k)
    results = []
    for i, s in zip(idxs[0], scores[0]):
        if i < 0:
            continue
        m = _meta[i]
        results.append({
            "chunk_id": m["chunk_id"],
            "company": m["company"],
            "section": m["section"],
            "section_label": m["section_label"],
            "score": float(s),
            "text": m["text"],
        })
    return results


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a citation-tagged context string for
    the generation step. Each chunk is labeled with a [source] tag the
    LLM can cite."""
    blocks = []
    for c in chunks:
        tag = f"{c['company'].title()} | {c['section_label']} | {c['chunk_id']}"
        blocks.append(f"[source: {tag}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


# Tool-calling schema — register this in the agent's ReAct tool registry
# (Project 3 bridge). Mirrors the OpenAI/Groq function-tool format.
RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the SEC 10-K corpus (Apple, JPMorgan, Walmart, FY2023; "
            "Business, Risk Factors, and MD&A sections) and return the most "
            "relevant passages for a natural-language query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}
