import argparse
import logging
import os
from scraper import fetch_metadata
from storage import save_to_db

# Project root is one level up from this src/ directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "paper_db")

def parse_args():
    parser = argparse.ArgumentParser(description="Paper metadata and abstract crawler.")
    
    parser.add_argument("--venues-file", "-f", type=str, required=True, 
                        help="Path to a file containing a list of venue abbreviations (one per line).")
    parser.add_argument("--year", "-y", type=str, nargs='+', required=True, 
                        help="Year(s) of publication (e.g., 2025 or 2023 2024 2025).")
    parser.add_argument("--type", "-t", type=str, choices=['conf', 'journal'], required=True, 
                        help="Type of venue: 'conf' or 'journal'.")
    parser.add_argument("--out-dir", "-o", type=str, default=_DEFAULT_OUT_DIR, 
                        help="Base directory to output the database.")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="Maximum number of papers to scrape per venue (0 means no limit).")
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    setup_logging()
    args = parse_args()
    
    if not os.path.exists(args.venues_file):
        logging.error(f"Venues file not found: {args.venues_file}")
        return
        
    with open(args.venues_file, 'r', encoding='utf-8') as f:
        venues = [line.strip().lower() for line in f if line.strip()]
        
    logging.info(f"Loaded {len(venues)} venues and {len(args.year)} year(s) to process.")
    
    for year in args.year:
        for venue in venues:
            logging.info(f"Processing venue: {venue} for year {year}")
            try:
                papers = fetch_metadata(venue, year, args.type, args.limit)
                if papers:
                    save_to_db(venue, year, papers, base_dir=args.out_dir)
                else:
                    logging.warning(f"No data returned for venue: {venue} ({year})")
            except Exception as e:
                logging.error(f"Error processing {venue} ({year}): {e}")

if __name__ == "__main__":
    main()
