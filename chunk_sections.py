import os
import re
import json
import tiktoken

SECTIONS_DIR = "data/sections"
OUT_PATH = "data/chunks.jsonl"

TARGET_TOKENS = 600
OVERLAP_TOKENS = 100

ENC = tiktoken.get_encoding("cl100k_base")

# Human-readable section labels
SECTION_LABELS = {
    "item1_business": "Item 1 - Business",
    "item1a_riskfactors": "Item 1A - Risk Factors",
    "item7_mdna": "Item 7 - MD&A",
}


def n_tokens(text):
    return len(ENC.encode(text))


def parse_filename(fname):
    # e.g. "walmart_item1a_riskfactors.txt" -> ("walmart", "item1a_riskfactors")
    base = fname[:-4]  # strip .txt
    parts = base.split("_", 1)
    return parts[0], parts[1]


def split_paragraphs(text):
    # sections are newline-joined lines; treat each non-empty line as a paragraph unit
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def chunk_paragraphs(paras):
    """Accumulate paragraphs up to TARGET_TOKENS, then start a new chunk
    carrying ~OVERLAP_TOKENS of trailing text from the previous chunk."""
    chunks = []
    current = []
    current_tokens = 0

    for para in paras:
        pt = n_tokens(para)
        # If a single paragraph exceeds target, hard-split it by tokens.
        if pt > TARGET_TOKENS:
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            toks = ENC.encode(para)
            for i in range(0, len(toks), TARGET_TOKENS - OVERLAP_TOKENS):
                piece = ENC.decode(toks[i:i + TARGET_TOKENS])
                chunks.append(piece)
            continue

        if current_tokens + pt > TARGET_TOKENS and current:
            chunks.append("\n".join(current))
            # build overlap: take trailing paragraphs until ~OVERLAP_TOKENS
            overlap, ov_tokens = [], 0
            for p in reversed(current):
                t = n_tokens(p)
                if ov_tokens + t > OVERLAP_TOKENS:
                    break
                overlap.insert(0, p)
                ov_tokens += t
            current = overlap[:]
            current_tokens = ov_tokens

        current.append(para)
        current_tokens += pt

    if current:
        chunks.append("\n".join(current))
    return chunks


def main():
    files = sorted(f for f in os.listdir(SECTIONS_DIR) if f.endswith(".txt"))
    all_records = []
    summary = []

    for fname in files:
        company, section = parse_filename(fname)
        with open(os.path.join(SECTIONS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        paras = split_paragraphs(text)
        chunks = chunk_paragraphs(paras)

        for idx, ctext in enumerate(chunks):
            rec = {
                "chunk_id": f"{company}_{section}_{idx:03d}",
                "company": company,
                "section": section,
                "section_label": SECTION_LABELS.get(section, section),
                "chunk_index": idx,
                "token_count": n_tokens(ctext),
                "text": ctext,
            }
            all_records.append(rec)
        summary.append((company, section, len(chunks)))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_records)} chunks -> {OUT_PATH}\n")
    print(f"{'company':<10} {'section':<22} {'chunks':>6}")
    print("-" * 40)
    for company, section, n in summary:
        print(f"{company:<10} {section:<22} {n:>6}")

    tokens = [r["token_count"] for r in all_records]
    print(f"\ntoken stats: min={min(tokens)} max={max(tokens)} "
          f"avg={sum(tokens)//len(tokens)}")


if __name__ == "__main__":
    main()
