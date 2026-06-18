"""Streamlit dashboard for the 10-K RAG pipeline."""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="10-K RAG", page_icon="📑", layout="wide")

st.title("📑 SEC 10-K RAG with Evaluation")
st.caption(
    "Retrieval-augmented Q&A over FY2023 10-K filings "
    "(Apple, JPMorgan, Walmart) — Business, Risk Factors, and MD&A sections."
)

EXAMPLES = {
    "Single-company": "What are Walmart's three reportable segments?",
    "Cross-company": "Which companies identify supply chain risks?",
    "Unanswerable": "What is Apple's cryptocurrency mining strategy?",
}

# --- session state for the query box ---
if "query" not in st.session_state:
    st.session_state.query = ""

st.subheader("Ask a question")

cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.query = q

query = st.text_input(
    "Your question", value=st.session_state.query,
    placeholder="e.g. How does JPMorgan manage interest rate risk?")

k = st.slider("Passages to retrieve (k)", 3, 10, 5)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"query": query, "k": k},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    st.markdown("### Answer")
    st.write(data["answer"])

    st.markdown(f"### Retrieved passages ({len(data['chunks'])})")
    for i, c in enumerate(data["chunks"], 1):
        header = (f"{i}. {c['company'].title()} — {c['section_label']} "
                  f"(score {c['score']:.3f})")
        with st.expander(header):
            st.caption(f"chunk_id: {c['chunk_id']}")
            st.write(c["text"])

st.divider()

# --- Evaluation section ---
st.subheader("📊 Evaluation results")
try:
    ev = requests.get(f"{API_URL}/eval", timeout=30).json()
except Exception as e:
    ev = {"available": False}
    st.warning(f"Could not load eval results: {e}")

if ev.get("available"):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hit@k", f"{ev['hit_at_k']:.0%}")
    c2.metric("MRR", f"{ev['mrr']:.3f}")
    c3.metric("Faithfulness", f"{ev['faithfulness']}/5")
    c4.metric("Correctness", f"{ev['correctness']}/5")
    c5.metric("Abstention", ev["abstention"])

    st.caption(
        f"Evaluated on {ev['n_questions']} curated ground-truth questions "
        "(single-hop, cross-company, and unanswerable). Retrieval quality "
        "(Hit@k, MRR) is measured separately from answer quality "
        "(LLM-as-judge faithfulness & correctness, gpt-4o)."
    )

    with st.expander("Per-question results"):
        rows = []
        for r in ev["results"]:
            rows.append({
                "id": r["id"],
                "type": r["type"],
                "hit@k": r["hit@k"],
                "rr": r["reciprocal_rank"],
                "faith": r["faithfulness"],
                "correct": r["correctness"],
                "question": r["question"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.info(
        "**Documented failure (sh05):** For *'What was Walmart's total revenue "
        "in fiscal 2023?'* retrieval succeeded (the correct passage was "
        "returned at rank 1), but the generator reported net sales "
        "($605.9B) instead of total revenue ($611.3B) — both figures appear "
        "side-by-side in the same passage. This is a generation-side error, "
        "not a retrieval failure, which is exactly the distinction the "
        "two-dimensional eval is designed to expose."
    )
else:
    st.write("Eval results not available.")
