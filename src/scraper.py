import requests
from bs4 import BeautifulSoup
import logging
import re
import time

DBLP_BASE_URL = "https://dblp.org/db/"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"

DBLP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

SS_HEADERS = {
    "User-Agent": "paper-db-crawler/1.0",
}


def get_dblp_url(venue, year, venue_type):
    if venue_type == 'conf':
        return f"{DBLP_BASE_URL}conf/{venue}/{venue}{year}.html"
    elif venue_type == 'journal':
        return f"{DBLP_BASE_URL}journals/{venue}/{venue}{year}.html"
    return None


def fetch_abstract_from_semantic_scholar(doi_url: str) -> str:
    """
    Fetch abstract from Semantic Scholar API using a DOI URL.
    Extracts the DOI path from a full doi.org URL, then queries the API.
    Returns the abstract string, or empty string if not found.
    """
    # Extract raw DOI from URL like "https://doi.org/10.1109/..."
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi:
        return ""

    api_url = f"{SEMANTIC_SCHOLAR_API}/DOI:{doi}"
    params = {"fields": "abstract"}

    try:
        resp = requests.get(api_url, headers=SS_HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("abstract") or ""
            return abstract
        elif resp.status_code == 404:
            logging.debug(f"Semantic Scholar: paper not found for DOI {doi}")
        elif resp.status_code == 429:
            logging.warning("Semantic Scholar rate limit hit. Waiting 10 seconds...")
            time.sleep(10)
            # Retry once
            resp = requests.get(api_url, headers=SS_HEADERS, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("abstract") or ""
        else:
            logging.warning(f"Semantic Scholar returned {resp.status_code} for DOI {doi}")
    except Exception as e:
        logging.error(f"Error querying Semantic Scholar for DOI {doi}: {e}")

    return ""


def fetch_abstract_from_usenix(url: str, max_retries: int = 3) -> str:
    """
    Fetch abstract directly from a USENIX presentation page.
    USENIX stores the abstract in a static HTML div with class
    'field-name-field-paper-description'.
    Retries up to max_retries times on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            timeout = 15 * attempt  # 15s, 30s, 45s
            resp = requests.get(url, headers=DBLP_HEADERS, timeout=timeout)
            if resp.status_code != 200:
                logging.warning(f"USENIX page returned {resp.status_code} for {url}")
                return ""
            soup = BeautifulSoup(resp.text, 'html.parser')
            desc = soup.find('div', class_='field-name-field-paper-description')
            if desc:
                return desc.get_text(separator=' ', strip=True)
            return ""
        except Exception as e:
            logging.warning(f"USENIX fetch attempt {attempt}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(5)
    return ""


def fetch_abstract_from_acm(doi_url: str, max_retries: int = 3) -> str:
    """
    Fetch abstract from ACM Digital Library given a doi.org URL.
    ACM DL pages embed the abstract in a <div class="abstractSection ..."> block.
    Falls back to <meta name="description"> if the section is not found.
    """
    # Build the ACM DL page URL: https://dl.acm.org/doi/<doi-path>
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi:
        return ""
    acm_url = f"https://dl.acm.org/doi/{doi}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    for attempt in range(1, max_retries + 1):
        try:
            timeout = 20 * attempt
            resp = requests.get(acm_url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                logging.warning(f"ACM DL returned {resp.status_code} for {acm_url} (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(5)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Primary: <div class="abstractSection abstractInFull"> or similar
            abstract_div = soup.find("div", class_=re.compile(r"abstractSection"))
            if abstract_div:
                # Remove any nested heading tags (e.g., "Abstract")
                for tag in abstract_div.find_all(["h2", "h3", "h4"]):
                    tag.decompose()
                text = abstract_div.get_text(separator=" ", strip=True)
                if text:
                    return text

            # Fallback: <meta name="description">
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                return meta["content"].strip()

            return ""
        except Exception as e:
            logging.warning(f"ACM fetch attempt {attempt}/{max_retries} failed for {acm_url}: {e}")
            if attempt < max_retries:
                time.sleep(5)
    return ""


def fetch_abstract_from_ieee(doi_url: str, max_retries: int = 3) -> str:
    """
    Fetch abstract from IEEE Xplore using their public metadata API.
    The DOI is used to look up the article number, then the abstract is
    retrieved from the Xplore REST API endpoint.
    """
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi:
        return ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://ieeexplore.ieee.org/",
    }

    # IEEE Xplore search API — search by DOI to get the article number
    search_url = "https://ieeexplore.ieee.org/rest/search"
    search_params = {
        "queryText": f'"{doi}"',
        "newsearch": "true",
        "highlight": "true",
        "returnFacets": "ALL",
        "returnType": "SEARCH",
        "matchPubs": "true",
    }

    for attempt in range(1, max_retries + 1):
        try:
            timeout = 20 * attempt
            resp = requests.get(search_url, headers=headers, params=search_params, timeout=timeout)
            if resp.status_code != 200:
                logging.warning(
                    f"IEEE search returned {resp.status_code} for DOI {doi} (attempt {attempt})"
                )
                if attempt < max_retries:
                    time.sleep(5)
                continue

            data = resp.json()
            records = data.get("records") or []
            if not records:
                logging.debug(f"IEEE: no records found for DOI {doi}")
                return ""

            article_number = records[0].get("articleNumber") or records[0].get("article_number")
            if not article_number:
                logging.debug(f"IEEE: no articleNumber in first record for DOI {doi}")
                return ""

            # Fetch full metadata for this article
            meta_url = f"https://ieeexplore.ieee.org/rest/document/{article_number}/abstract"
            meta_resp = requests.get(meta_url, headers=headers, timeout=timeout)
            if meta_resp.status_code == 200:
                meta_data = meta_resp.json()
                abstract = meta_data.get("abstract") or ""
                if abstract:
                    # Strip HTML tags that IEEE sometimes includes
                    abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
                    abstract = re.sub(r"\s+", " ", abstract)
                    return abstract

            # Fallback: abstract may be in the search record itself
            abstract = records[0].get("abstract") or ""
            if abstract:
                abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
                return abstract

            return ""
        except Exception as e:
            logging.warning(f"IEEE fetch attempt {attempt}/{max_retries} failed for DOI {doi}: {e}")
            if attempt < max_retries:
                time.sleep(5)
    return ""


def fetch_abstract_from_openalex(doi_url: str, max_retries: int = 3) -> str:
    """
    Fetch abstract from OpenAlex API using a DOI URL.
    OpenAlex is free, requires no API key, and has excellent coverage of ACM/IEEE papers.
    Abstracts are stored as an inverted index and must be reconstructed here.
    """
    doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi:
        return ""

    # OpenAlex can be queried directly by DOI URL
    api_url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    headers = {
        "User-Agent": "CCF-papers-RAG/1.0 (mailto:research@example.com)",
        "Accept": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            timeout = 15 * attempt
            resp = requests.get(api_url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                logging.debug(f"OpenAlex: paper not found for DOI {doi}")
                return ""
            if resp.status_code != 200:
                logging.warning(
                    f"OpenAlex returned {resp.status_code} for DOI {doi} (attempt {attempt})"
                )
                if attempt < max_retries:
                    time.sleep(3)
                continue

            data = resp.json()

            # OpenAlex stores abstracts as an inverted index:
            # {"word": [pos1, pos2, ...], ...}
            inverted = data.get("abstract_inverted_index")
            if not inverted:
                return ""

            # Rebuild the abstract by sorting words by their first position
            max_pos = max(pos for positions in inverted.values() for pos in positions)
            word_list = [""] * (max_pos + 1)
            for word, positions in inverted.items():
                for pos in positions:
                    word_list[pos] = word

            abstract = " ".join(w for w in word_list if w).strip()
            return abstract

        except Exception as e:
            logging.warning(f"OpenAlex fetch attempt {attempt}/{max_retries} failed for DOI {doi}: {e}")
            if attempt < max_retries:
                time.sleep(3)
    return ""


def fetch_metadata(venue, year, venue_type, limit=0):
    """
    Fetches the list of paper metadata from DBLP for a specific venue and year.
    Then enriches each paper with abstract from Semantic Scholar.
    Returns: list[dict]
    """
    url = get_dblp_url(venue, year, venue_type)
    if not url:
        logging.error("Invalid venue type.")
        return []

    logging.info(f"Fetching DBLP page: {url}")

    try:
        response = requests.get(url, headers=DBLP_HEADERS, timeout=60)
        if response.status_code != 200:
            logging.error(f"Failed to fetch DBLP page. Status code: {response.status_code}")

            fallback_url = url.replace(".html", "-1.html")
            logging.info(f"Trying fallback URL: {fallback_url}")
            response = requests.get(fallback_url, headers=DBLP_HEADERS, timeout=30)
            if response.status_code != 200:
                return []

        soup = BeautifulSoup(response.text, 'html.parser')

        venue_full_name = ""
        header = soup.find('h1')
        if header:
            venue_full_name = header.get_text(strip=True)

        entry_class = "inproceedings" if venue_type == 'conf' else "article"
        paper_entries = soup.find_all('li', class_=f"entry {entry_class}")

        papers = []
        logging.info(f"Found {len(paper_entries)} potential papers. Extracting DBLP details...")

        for i, entry in enumerate(paper_entries):
            paper_data = {
                "title": "",
                "authors": [],
                "year": year,
                "venue_abbr": venue,
                "venue_full_name": venue_full_name,
                "dblp_url": "",
                "doi_url": "",
                "abstract": "",
                "keywords": []
            }

            title_span = entry.find('span', class_='title')
            if title_span:
                paper_data["title"] = title_span.get_text(strip=True).rstrip('.')

            author_spans = entry.find_all('span', itemprop='author')
            for a_span in author_spans:
                name = a_span.find('span', itemprop='name')
                if name:
                    paper_data["authors"].append(name.get_text(strip=True))

            ee_links = entry.find_all('li', class_='ee')
            for ee in ee_links:
                a_tag = ee.find('a')
                if a_tag and a_tag.has_attr('href'):
                    href = a_tag['href']
                    if 'doi.org' in href:
                        paper_data["doi_url"] = href
                    else:
                        if not paper_data["dblp_url"]:
                            paper_data["dblp_url"] = href

            if not paper_data["dblp_url"]:
                dblp_link = entry.find('a', href=re.compile(r'dblp.org/rec/'))
                if dblp_link:
                    paper_data["dblp_url"] = dblp_link['href']

            papers.append(paper_data)

            if limit > 0 and len(papers) >= limit:
                logging.info(f"Reached limit of {limit} papers. Stopping early.")
                break

        # Fetch abstracts: Semantic Scholar (with DOI) or USENIX HTML fallback
        logging.info("Starting abstract collection...")
        for i, paper in enumerate(papers):
            doi_url = paper.get("doi_url", "")
            pub_url = paper.get("dblp_url", "")
            abstract = ""

            if doi_url:
                logging.info(f"[{i+1}/{len(papers)}] Fetching abstract (S2): {paper['title'][:60]}...")
                abstract = fetch_abstract_from_semantic_scholar(doi_url)
                time.sleep(1)
            elif pub_url and "usenix.org" in pub_url:
                logging.info(f"[{i+1}/{len(papers)}] Fetching abstract (USENIX): {paper['title'][:60]}...")
                abstract = fetch_abstract_from_usenix(pub_url)
                time.sleep(0.5)
            else:
                logging.debug(f"[{i+1}/{len(papers)}] No URL available for abstract: {paper['title'][:60]}")

            if abstract:
                paper["abstract"] = abstract
                logging.info(f"  -> Abstract found ({len(abstract)} chars)")
            else:
                logging.warning(f"  -> No abstract found for: {paper['title'][:60]}")

        return papers

    except Exception as e:
        logging.error(f"Error while fetching metadata from DBLP: {e}", exc_info=True)
        return []
