import re
import os
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

RAW_DIR = "data/raw"
OUT_DIR = "data/sections"
os.makedirs(OUT_DIR, exist_ok=True)

# Per-company anchors. Each section: (key, start_regex, [end_regexes], occurrence)
# occurrence: "last" = use last match (skips TOC), "first_after_N" handled via min_line.
# We match on cleaned, whitespace-collapsed lines, case-insensitive unless noted.
CONFIG = {
    "apple": {
        "min_line": 1035,  # skip TOC region
        "sections": [
            ("item1_business", r"^Item\s+1\.\s+Business", [r"^Item\s+1A\.\s+Risk\s+Factors"]),
            ("item1a_riskfactors", r"^Item\s+1A\.\s+Risk\s+Factors", [r"^Item\s+1B\.\s+Unresolved"]),
            ("item7_mdna", r"^Item\s+7\.\s+Management", [r"^Item\s+7A\.\s+Quantitative"]),
        ],
    },
    "jpmorgan": {
        "min_line": 9960,  # skip TOC region
        "sections": [
            ("item1_business", r"^Item\s+1\.\s+Business", [r"^Item\s+1A\.\s+Risk\s+Factors"]),
            ("item1a_riskfactors", r"^Item\s+1A\.\s+Risk\s+Factors", [r"^Item\s+2\.\s+Properties"]),
            # MD&A is incorporated by reference; real body starts at "The following is Management's discussion"
            ("item7_mdna", r"^The following is Management.s discussion and analysis",
             [r"^JPMorgan Chase.s Consolidated Financial Statements as of"]),
        ],
    },
    "walmart": {
        "min_line": 1780,  # skip TOC region; real all-caps sections start ~1783
        "sections": [
            # Walmart uses all-caps two-line headers: "ITEM 1." then "BUSINESS"
            ("item1_business", r"^ITEM\s+1\.$", [r"^ITEM\s+1A\.$"]),
            ("item1a_riskfactors", r"^ITEM\s+1A\.$", [r"^ITEM\s+2\.$", r"^ITEM\s+1B\.$"]),
            ("item7_mdna", r"^ITEM\s+7\.$", [r"^ITEM\s+7A\.$", r"^ITEM\s+8\.$"]),
        ],
    },
}


def clean_lines(html_path):
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in soup.get_text("\n").split("\n")]
    return [ln for ln in lines if ln]


def last_match_after(lines, pattern, min_line, case_insensitive=True):
    flags = re.IGNORECASE if case_insensitive else 0
    pat = re.compile(pattern, flags)
    hits = [i for i, ln in enumerate(lines) if i >= min_line and pat.match(ln)]
    return hits[-1] if hits else None


def first_after(lines, patterns, start_idx, case_insensitive=True):
    flags = re.IGNORECASE if case_insensitive else 0
    for i in range(start_idx + 1, len(lines)):
        for p in patterns:
            if re.match(p, lines[i], flags):
                return i
    return None


def main():
    for company, cfg in CONFIG.items():
        html_path = os.path.join(RAW_DIR, f"{company}_10k.html")
        lines = clean_lines(html_path)
        min_line = cfg["min_line"]
        # Walmart all-caps anchors are case-sensitive to avoid matching prose
        ci = (company != "walmart")
        print(f"\n=== {company} ({len(lines):,} lines, min_line={min_line}) ===")
        for key, start_pat, end_pats in cfg["sections"]:
            start = last_match_after(lines, start_pat, min_line, ci)
            if start is None:
                print(f"  {key}: START NOT FOUND")
                continue
            end = first_after(lines, end_pats, start, ci)
            if end is None:
                print(f"  {key}: END NOT FOUND (start {start})")
                continue
            body = "\n".join(lines[start:end])
            wc = len(body.split())
            out_path = os.path.join(OUT_DIR, f"{company}_{key}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(body)
            flag = "  <-- SUSPICIOUS" if wc < 500 else ""
            print(f"  {key}: lines {start}-{end}, {wc:,} words{flag}")


if __name__ == "__main__":
    main()
