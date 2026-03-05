"""
refetch_abstracts.py

Scans all JSONL files under paper_db/, finds papers with empty abstracts,
retries fetching them (up to 3 times) via:
  - Semantic Scholar API       (if doi_url is present, as general fallback)
  - ACM Digital Library HTML   (if doi_url is an ACM DOI: 10.1145/...)
  - IEEE Xplore REST API       (if doi_url is an IEEE DOI: 10.1109/...)
  - USENIX HTML parser         (if dblp_url contains usenix.org)
Then writes the updated records back to the original file.
Papers that still have no abstract after all retries are saved to
'missing_abstracts.jsonl' in the db directory.

Usage:
    uv run python src/refetch_abstracts.py
    uv run python src/refetch_abstracts.py --db-dir paper_db
"""

import argparse
import json
import logging
import os
import sys

# Make scraper importable when running from project root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper import (
    fetch_abstract_from_semantic_scholar,
    fetch_abstract_from_usenix,
    fetch_abstract_from_acm,
    fetch_abstract_from_ieee,
    fetch_abstract_from_openalex,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_all_jsonl(db_dir: str):
    """Recursively yield all .jsonl file paths under db_dir."""
    for root, _, files in os.walk(db_dir):
        for f in sorted(files):
            if f.endswith(".jsonl"):
                yield os.path.join(root, f)


def refetch_file(path: str, max_retries: int = 3):
    """
    Read a JSONL file, retry fetching abstracts for papers that have none,
    and rewrite the file if any abstracts were updated.
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    papers = [json.loads(line) for line in lines]
    empty = [p for p in papers if not p.get("abstract")]

    if not empty:
        logging.info(f"[SKIP] {path} — all abstracts present")
        return 0, []

    logging.info(f"[FILE] {path} — {len(empty)}/{len(papers)} papers missing abstracts")
    updated = 0

    still_missing = []

    for paper in empty:
        doi_url = paper.get("doi_url", "")
        pub_url = paper.get("dblp_url", "")
        title_short = paper.get("title", "")[:60]
        abstract = ""

        # Determine which fetcher(s) to use based on available URLs
        def _source_label():
            if doi_url:
                if "10.1145/" in doi_url:
                    return "ACM"
                elif "10.1109/" in doi_url:
                    return "IEEE"
                else:
                    return "S2"
            elif pub_url and "usenix.org" in pub_url:
                return "USENIX"
            return None

        source = _source_label()
        if source is None:
            logging.debug(f"  No usable URL for: {title_short}")
            still_missing.append(paper)
            continue

        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"  [{attempt}/{max_retries}] {source} → {title_short}...")
                if source == "ACM":
                    # Try chain: S2 → OpenAlex → ACM DL (403 likely but worth a shot)
                    abstract = fetch_abstract_from_semantic_scholar(doi_url)
                    if not abstract:
                        logging.info(f"  S2 miss, trying OpenAlex for ACM paper...")
                        abstract = fetch_abstract_from_openalex(doi_url)
                    if not abstract:
                        logging.info(f"  OpenAlex miss, trying ACM DL directly...")
                        abstract = fetch_abstract_from_acm(doi_url, max_retries=1)
                elif source == "IEEE":
                    # Try chain: IEEE API → S2 → OpenAlex
                    abstract = fetch_abstract_from_ieee(doi_url, max_retries=1)
                    if not abstract:
                        abstract = fetch_abstract_from_semantic_scholar(doi_url)
                    if not abstract:
                        logging.info(f"  S2 miss, trying OpenAlex for IEEE paper...")
                        abstract = fetch_abstract_from_openalex(doi_url)
                elif source == "USENIX":
                    abstract = fetch_abstract_from_usenix(pub_url, max_retries=1)
                else:  # S2 as general fallback for other DOIs
                    abstract = fetch_abstract_from_semantic_scholar(doi_url)

                if abstract:
                    break  # success
                logging.warning(f"  Attempt {attempt} returned empty, retrying...")
            except Exception as e:
                logging.warning(f"  Attempt {attempt} failed: {e}")

        if abstract:
            paper["abstract"] = abstract
            updated += 1
            logging.info(f"  ✓ Abstract found ({len(abstract)} chars)")
        else:
            logging.warning(f"  ✗ No abstract after {max_retries} attempts: {title_short}")
            still_missing.append(paper)

    if updated > 0:
        with open(path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")
        logging.info(f"  → Wrote {updated} updated abstract(s) back to {path}")
    else:
        logging.info(f"  → No updates for {path}")

    return updated, still_missing


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Retry fetching missing abstracts in paper_db.")
    parser.add_argument(
        "--db-dir", "-d",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper_db",
        ),
        help="Path to the paper_db directory (default: project root/paper_db)",
    )
    parser.add_argument(
        "--retries", "-r", type=int, default=3,
        help="Max retry attempts per paper (default: 3)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        logging.error(f"paper_db directory not found: {args.db_dir}")
        sys.exit(1)

    logging.info(f"Scanning {args.db_dir} for JSONL files...")
    total_updated = 0
    all_missing = []

    for jsonl_path in find_all_jsonl(args.db_dir):
        n_updated, missing = refetch_file(jsonl_path, max_retries=args.retries)
        total_updated += n_updated
        all_missing.extend(missing)

    logging.info(f"Done. Total abstracts updated: {total_updated}")

    if all_missing:
        missing_path = os.path.join(args.db_dir, "missing_abstracts.jsonl")
        with open(missing_path, "w", encoding="utf-8") as f:
            for paper in all_missing:
                record = {
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "venue_abbr": paper.get("venue_abbr", ""),
                    "doi_url": paper.get("doi_url", ""),
                    "dblp_url": paper.get("dblp_url", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(all_missing)} unfetchable papers to {missing_path}")


if __name__ == "__main__":
    main()
