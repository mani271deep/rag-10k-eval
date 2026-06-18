"""Grounded generation for the 10-K RAG pipeline.

answer(query, k) retrieves top-k chunks and synthesizes a grounded answer
with citations, abstaining when the context does not contain the answer.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
from retriever import retrieve, format_context

load_dotenv()

GEN_MODEL = "gpt-4o-mini"
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a financial research assistant answering questions \
about SEC 10-K filings for Apple, JPMorgan, and Walmart (fiscal year 2023).

Rules you must follow:
1. Answer ONLY using the provided context passages. Do not use outside knowledge.
2. Cite the source(s) for each claim using the source tags shown in the context, \
e.g. (Apple | Item 1 - Business).
3. If the context does not contain enough information to answer the question, \
respond exactly with: "I don't have enough information in the provided 10-K \
excerpts to answer this." Do not guess or use prior knowledge.
4. Be concise and factual. Do not speculate beyond what the passages state."""

USER_TEMPLATE = """Context passages:

{context}

---

Question: {query}

Answer using only the context above, with source citations."""


def answer(query: str, k: int = 5) -> dict:
    chunks = retrieve(query, k=k)
    context = format_context(chunks)
    user_msg = USER_TEMPLATE.format(context=context, query=query)

    resp = _client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    answer_text = resp.choices[0].message.content.strip()

    return {
        "query": query,
        "answer": answer_text,
        "chunks": chunks,
        "context": context,
    }


if __name__ == "__main__":
    examples = [
        "What are Apple's main sources of revenue?",          # single-company factual
        "Which of these companies face supply chain risks?",  # cross-company
        "What is Apple's strategy for its cryptocurrency mining operations?",  # unanswerable
    ]
    for q in examples:
        out = answer(q, k=5)
        print("=" * 70)
        print("Q:", q)
        print("-" * 70)
        print(out["answer"])
        print()
        print("Retrieved:", ", ".join(
            f"{c['company']}/{c['section']}" for c in out["chunks"]))
        print()
