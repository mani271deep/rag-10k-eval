import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = "data/chunks.jsonl"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/chunks_meta.json"
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(texts):
    # one batched call; OpenAI accepts a list of inputs
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    # resp.data is returned in input order
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    return vecs


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vecs / norms


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    print(f"Embedding with {EMB_MODEL} ...")
    vecs = embed_texts(texts)
    assert vecs.shape == (len(chunks), EMB_DIM), f"unexpected shape {vecs.shape}"
    vecs = normalize(vecs)

    # Flat inner-product index = cosine similarity on normalized vectors
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)
    print(f"Wrote index ({index.ntotal} vectors) -> {INDEX_PATH}")

    # Save metadata parallel to vector order (position i in index == meta[i])
    meta = [
        {k: c[k] for k in
         ("chunk_id", "company", "section", "section_label",
          "chunk_index", "token_count", "text")}
        for c in chunks
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"Wrote metadata ({len(meta)} records) -> {META_PATH}")

    # --- sanity query ---
    q = "What are the risk factors related to supply chain disru彫tion?".replace("彫", "")
    print(f"\nSanity query: {q!r}")
    qvec = normalize(embed_texts([q]))
    scores, idxs = index.search(qvec, 5)
    print("\nTop 5 retrieved:")
    for rank, (i, s) in enumerate(zip(idxs[0], scores[0]), 1):
        m = meta[i]
        preview = m["text"][:90].replace("\n", " ")
        print(f"  {rank}. [{s:.3f}] {m['company']}/{m['section']} "
              f"({m['chunk_id']})")
        print(f"       {preview}...")


if __name__ == "__main__":
    main()
