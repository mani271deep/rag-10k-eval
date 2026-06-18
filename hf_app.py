"""Hugging Face Spaces entry point — in-process Streamlit RAG demo.

Imports the pipeline directly (no FastAPI/HTTP layer) so the whole demo
runs in a single Space. The FastAPI backend (app.py) remains in the repo
as the standalone API implementation.
"""
import os
import json
import streamlit as st

# set_page_config must be the FIRST Streamlit command in the script.
st.set_page_config(page_title="10-K RAG", page_icon="📑", layout="wide")

# Build the index on first launch if it is not present (HF Spaces
# does not store the prebuilt binary; it is regenerated from chunks.jsonl).
@st.cache_resource
def _ensure_index():
    if not os.path.exists("data/index.faiss"):
        import build_index
        build_index.main()
    return True

_ensure_index()

from generator import answer

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
            out = answer(query, k=k)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.markdown("### Answer")
    st.write(out["answer"])

    st.markdown(f"### Retrieved passages ({len(out['chunks'])})")
    for i, c in enumerate(out["chunks"], 1):
        header = (f"{i}. {c['company'].title()} — {c['section_label']} "
                  f"(score {c['score']:.3f})")
        with st.expander(header):
            st.caption(f"chunk_id: {c['chunk_id']}")
            st.write(c["text"])

st.divider()

st.subheader("📊 Evaluation results")

RESULTS_PATH = os.path.join("eval", "results.json")
if os.path.exists(RESULTS_PATH):
    results = json.load(open(RESULTS_PATH))
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    hit = sum(r["hit@k"] for r in answerable) / len(answerable)
    mrr = sum(r["reciprocal_rank"] for r in answerable) / len(answerable)
    faith = sum(r["faithfulness"] for r in results) / len(results)
    correct = sum(r["correctness"] for r in results) / len(results)
    abstain = sum(1 for r in unanswerable if r["correctness"] >= 4)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hit@k", f"{hit:.0%}")
    c2.metric("MRR", f"{mrr:.3f}")
    c3.metric("Faithfulness", f"{faith:.2f}/5")
    c4.metric("Correctness", f"{correct:.2f}/5")
    c5.metric("Abstention", f"{abstain}/{len(unanswerable)}")

    st.caption(
        f"Evaluated on {len(results)} curated ground-truth questions "
        "(single-hop, cross-company, unanswerable). Retrieval quality "
        "(Hit@k, MRR) is measured separately from answer quality "
        "(LLM-as-judge faithfulness & correctness, gpt-4o)."
    )

    with st.expander("Per-question results"):
        rows = [{
            "id": r["id"], "type": r["type"], "hit@k": r["hit@k"],
            "rr": r["reciprocal_rank"], "faith": r["faithfulness"],
            "correct": r["correctness"], "question": r["question"],
        } for r in results]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.info(
        "**Documented failure (sh05):** For *'What was Walmart's total revenue "
        "in fiscal 2023?'* retrieval succeeded (correct passage at rank 1), but "
        "the generator reported net sales ($605.9B) instead of total revenue "
        "($611.3B) — both figures appear side-by-side in the same passage. A "
        "generation-side error, not a retrieval failure: exactly the distinction "
        "the two-dimensional eval is designed to expose."
    )
else:
    st.write("Eval results not available.")
