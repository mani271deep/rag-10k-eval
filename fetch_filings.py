import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

UA = os.getenv("EDGAR_USER_AGENT")
if not UA or UA == "REPLACE_ME":
    sys.exit("ERROR: set EDGAR_USER_AGENT in .env to 'Your Name your@email.com'")

HEADERS = {"User-Agent": UA}
RAW_DIR = "data/raw"

COMPANIES = {
    "apple": 320193,
    "jpmorgan": 19617,
    "walmart": 104169,
}

TARGET_FY = "2023"


def get_json(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    return get_json(url)


def scan_block(recent):
    """Yield 10-K filings from a 'recent'-shaped dict."""
    out = []
    forms = recent["form"]
    for i, form in enumerate(forms):
        if form == "10-K":
            out.append({
                "accession": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_doc": recent["primaryDocument"][i],
            })
    return out


def find_10k(subs):
    # 1. check inline recent array
    candidates = scan_block(subs["filings"]["recent"])
    for c in candidates:
        if c["report_date"].startswith(TARGET_FY):
            return c
    # 2. page through additional files if needed
    for f in subs["filings"].get("files", []):
        url = f"https://data.sec.gov/submissions/{f['name']}"
        block = get_json(url)
        # additional files are shaped like the 'recent' dict directly
        more = scan_block(block)
        for c in more:
            if c["report_date"].startswith(TARGET_FY):
                return c
        time.sleep(0.3)
    return None


def download_doc(cik, accession, primary_doc, out_path):
    acc_nodash = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{acc_nodash}/{primary_doc}"
    )
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(r.text)
    return url, len(r.text)


def main():
    for name, cik in COMPANIES.items():
        print(f"\n=== {name} (CIK {cik}) ===")
        subs = get_submissions(cik)
        print(f"  resolved entity: {subs.get('name')}")
        filing = find_10k(subs)
        if not filing:
            print(f"  ERROR: no FY{TARGET_FY} 10-K found")
            continue
        print(f"  10-K filing date: {filing['filing_date']}")
        print(f"  fiscal period end: {filing['report_date']}")
        print(f"  accession: {filing['accession']}")
        out_path = os.path.join(RAW_DIR, f"{name}_10k.html")
        url, size = download_doc(
            cik, filing["accession"], filing["primary_doc"], out_path
        )
        print(f"  downloaded: {url}")
        print(f"  saved {size:,} chars -> {out_path}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
