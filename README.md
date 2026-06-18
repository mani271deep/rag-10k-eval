---
title: SEC 10-K RAG with Evaluation
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.2
app_file: streamlit_app.py
pinned: false
license: mit
short_description: RAG over SEC 10-K filings with evaluation
---

# SEC 10-K RAG with Evaluation

A retrieval-augmented generation (RAG) system that answers natural-language questions about SEC 10-K filings, paired with a rigorous evaluation harness that measures **retrieval quality and answer quality as separate dimensions** — not just "did it produce an answer."

**Live demo:** https://huggingface.co/spaces/manideep271/rag-10k-eval

Corpus: FY2023 10-K filings for **Apple, JPMorgan, and Walmart** — the three narrative sections of each (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A).

---

## What it does

Ask a question like *"What are Walmart's three reportable segments?"* or *"Which companies identify supply chain risks?"* and the system retrieves the most relevant passages from the filings and generates a grounded answer with source citations. If the answer isn't in the corpus, it abstains rather than hallucinating.

The dashboard shows the answer, the retrieved passages (with similarity scores and source tags), and the evaluation results.

---

## Architecture

```
Question
   │
   ▼
retrieve(query, k)            # embed query → FAISS cosine search → top-k chunks
   │
   ▼
format_context(chunks)        # citation-tagged context
   │
   ▼
answer(query, k)              # grounded generation with abstention contract
   │
   ▼
Answer + citations + retrieved passages
```

**Pipeline stages**

1. **Acquisition** — FY2023 10-Ks pulled from SEC EDGAR via the submissions API.
2. **Extraction** — the three narrative sections (Item 1 / 1A / 7) carved out per company.
3. **Chunking** — token-aware, paragraph-preserving, section-bounded chunks (~600 tokens, ~100 overlap) → 189 chunks.
4. **Embedding & index** — `text-embedding-3-small` (1536-dim), normalized vectors in a FAISS flat inner-product index (= exact cosine similarity).
5. **Retrieval** — `retrieve(query, k)` returns scored, metadata-rich chunks.
6. **Generation** — `gpt-4o-mini` synthesizes a grounded answer, cites sources, and abstains when the context lacks the answer.
7. **Evaluation** — retrieval metrics (Hit@k, MRR) + LLM-as-judge answer quality (`gpt-4o`).

**Model split:** generation uses `gpt-4o-mini` (cheap, reliable); the eval judge uses `gpt-4o` (stronger than the generator, which is the right relationship for a credible judge).

**Deployment shape:** the live demo runs as a single in-process Streamlit app (`streamlit_app.py`) on Hugging Face Spaces. A standalone FastAPI backend (`app.py`) is also included, exposing `/query`, `/eval`, and `/health` — the decoupled architecture, preserved for local/standalone use even though the hosted demo collapses both into one process for simplicity.

---

## Evaluation (the headline)

This project's purpose is the evaluation, not just the pipeline. The key idea: **retrieval quality and answer quality are different things and must be measured separately.** A system can retrieve the right passage but still generate a wrong answer (and vice versa). Conflating them hides exactly the failures that matter.

The harness runs against 14 curated ground-truth questions across three types:

- **Single-hop factual** — answer lives in a known company/section (clean retrieval label).
- **Cross-company** — tests whether retrieval discriminates across the three filings.
- **Unanswerable** — answer is *not* in the corpus; tests correct abstention.

**Metrics**

| Dimension | Metric | Result |
|---|---|---|
| Retrieval | Hit@5 | 100% |
| Retrieval | MRR | 1.000 |
| Answer quality | Faithfulness (gpt-4o judge, /5) | 4.64 |
| Answer quality | Correctness (gpt-4o judge, /5) | 4.29 |
| Abstention | Correct refusals on unanswerable | 3/3 |

**Retrieval** is scored against verified labels: each answerable question carries the acceptable company+section(s) where the answer genuinely lives. Hit@k asks whether a correct passage appears in the top k; MRR captures how highly it ranks.

**Answer quality** uses `gpt-4o` as judge, scoring faithfulness (is the answer grounded in the retrieved context, with no fabrication?) and correctness (does it match the reference?). For unanswerable questions, correct abstention scores full marks.

### Documented failure: sh05

The most instructive result is a failure. For *"What was Walmart's total revenue in fiscal 2023?"*:

- **Retrieval succeeded** — the passage containing the figure was returned at rank 1.
- **Generation failed** — the model answered **$605.9B (net sales)** instead of **$611.3B (total revenue)**. Both figures sit side-by-side in the same passage ("total revenues of $611.3 billion, which was comprised primarily of net sales of $605.9 billion").

This is a **generation-side error, not a retrieval failure** — precisely the distinction the two-dimensional eval is designed to expose. A single "task success" metric would have blurred it. (A candid footnote: the gpt-4o judge's written rationale for this item was itself slightly off, a reminder that LLM judges are strong but not infallible.)

The fix is deliberately *not* applied: tuning the prompt after seeing the failure would overfit to the test set. The honest move is to document it and treat the disambiguation of similar financial metrics as future work.

---

## Bridge to the agentic SQL project (Project 3)

Retrieval is implemented as a clean, tool-shaped function so it can drop into an agent's ReAct tool registry unchanged:

```python
def retrieve(query: str, k: int = 5) -> list[dict]: ...
```

The module also exports `RETRIEVE_TOOL_SCHEMA` — a function-calling spec in the same format the agent's other tools (`inspect_schema`, `run_sql`, `python_exec`) use. Wiring this RAG system into the agent later is registration, not redesign: the agent gains a `retrieve` tool and the same ReAct loop handles it.

---

## Engineering decisions & gotchas

Real filings and real platforms surfaced real problems. The interesting ones:

**Narrative-section scoping.** Full 10-Ks include exhibits, tables, and XBRL noise that make general-purpose parsing a rabbit hole. Scoping to the three prose-heavy narrative sections (where the answerable questions actually live) was a deliberate decision to avoid brittle parsing — documented as a sound boundary, not a limitation.

**Per-company extraction anchors.** The three filings are structured differently and could not share one parser:

- *Apple* uses clean titled headers ("Item 1. Business").
- *Walmart* uses all-caps two-line headers ("ITEM 1." / "BUSINESS") and repeats header text in its TOC.
- *JPMorgan* incorporates its MD&A **by reference** — the Item 7 header is a stub pointing to the real discussion located elsewhere in the document.

Each company got verified anchors; every extracted section was checked by word count *and* eyeballed for clean prose.

**EDGAR pagination.** JPMorgan files so frequently that its FY2023 10-K had paginated off the inline "recent" submissions array; the fetcher pages into the additional submission files to find it.

**Rebuild-on-startup index.** Hugging Face Spaces rejects committed binaries, so the FAISS index is *not* stored in git. Instead it's regenerated from `chunks.jsonl` (a text artifact) on first launch. The retriever loads lazily so importing the pipeline never requires a pre-existing index. This makes the deployment reproducible from text alone.

**Python 3.13 on the Space.** The hosted container runs Python 3.13, where the locally-pinned `faiss-cpu==1.9.0` has no wheel. The deployed `faiss-cpu`/`numpy` pins are relaxed to `>=` for 3.13 compatibility; local development uses 3.11 with exact pins.

**Secrets hygiene.** `.env` is gitignored and was verified absent from the entire git history before any push. API keys live only in the Space's encrypted secrets.

---

## Running locally

Environment note: build the virtualenv on Python 3.11 and call the venv interpreter explicitly (`./venv/bin/python`) to avoid any system/Anaconda Python shadowing it on PATH.

```bash
# setup
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt

# secrets
cp .env.example .env        # then fill in OPENAI_API_KEY, GROQ_API_KEY, EDGAR_USER_AGENT

# build the corpus + index (one-time)
./venv/bin/python fetch_filings.py       # pull 10-Ks from EDGAR
./venv/bin/python extract_sections.py    # carve narrative sections
./venv/bin/python chunk_sections.py      # → data/chunks.jsonl (189 chunks)
./venv/bin/python build_index.py         # → FAISS index + metadata

# run the evaluation
./venv/bin/python eval/run_eval.py

# run the API (optional standalone backend)
./venv/bin/python -m uvicorn app:app --port 8000

# run the dashboard
./venv/bin/python -m streamlit run streamlit_app.py
```

---

## Repository structure

```
fetch_filings.py      # acquire FY2023 10-Ks from SEC EDGAR
extract_sections.py   # per-company narrative-section extraction
chunk_sections.py     # token-aware, section-bounded chunking
build_index.py        # embed chunks → FAISS index (+ rebuilt on Space startup)
retriever.py          # retrieve() + RETRIEVE_TOOL_SCHEMA (Project 3 bridge)
generator.py          # grounded generation with abstention contract
app.py                # FastAPI backend (/query, /eval, /health)
streamlit_app.py      # Streamlit dashboard (in-process; Space entry point)
eval/
  ground_truth.json   # 14 curated, corpus-verified Q&A labels
  run_eval.py         # retrieval (Hit@k, MRR) + LLM-as-judge harness
  results.json        # eval output
data/
  chunks.jsonl        # 189 chunks with metadata (committed)
  sections/           # 9 extracted narrative sections (committed)
  # index.faiss + chunks_meta.json are rebuilt on startup, not committed
```

---

## Stack

Python 3.11 · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`, `gpt-4o`) · Groq · FAISS · FastAPI · Streamlit · Hugging Face Spaces
