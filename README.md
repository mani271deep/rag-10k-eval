---
title: SEC 10-K RAG with Evaluation
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.2
app_file: hf_app.py
pinned: false
license: mit
short_description: RAG over SEC 10-K filings with evaluation
---

# SEC 10-K RAG with Evaluation

Retrieval-augmented generation over FY2023 10-K filings (Apple, JPMorgan, Walmart), with a two-dimensional evaluation harness measuring **retrieval quality** (Hit@k, MRR) separately from **answer quality** (LLM-as-judge faithfulness & correctness).

_Full documentation is added in Phase 9._
