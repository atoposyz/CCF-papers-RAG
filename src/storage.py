import os
import json
import logging

def save_to_db(venue, year, papers, base_dir="../paper_db"):
    """
    Saves a list of paper dictionaries to a JSONL file.
    
    Args:
        venue (str): The abbreviation of the venue (e.g., 'dac', 'kdd').
        year (str): The publication year.
        papers (list[dict]): A list of dictionaries, where each dict is a paper's metadata.
        base_dir (str): The base directory for the database.
    """
    if not papers:
        logging.warning(f"No papers to save for {venue} {year}.")
        return

    # Create directory for the specific year
    year_dir = os.path.join(base_dir, year)
    os.makedirs(year_dir, exist_ok=True)
    
    # Define file path
    file_path = os.path.join(year_dir, f"{venue.lower()}.jsonl")
    
    logging.info(f"Saving {len(papers)} papers for {venue} {year} to {file_path}")
    
    # Write to JSONL file
    with open(file_path, 'w', encoding='utf-8') as f:
        for paper in papers:
            json_line = json.dumps(paper, ensure_ascii=False)
            f.write(json_line + '\n')
            
    logging.info("Save complete.")
