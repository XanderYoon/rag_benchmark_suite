
import os
import time
import json
import requests
import feedparser
from tqdm import tqdm
from urllib.parse import quote


############################################
# Configuration
############################################

SAVE_DIR = "rag_corpus_pdf"
METADATA_FILE = "metadata.json"
MAX_RESULTS_PER_QUERY = 200
START_YEAR = 2019   # filter by year
ARXIV_API = "http://export.arxiv.org/api/query?"


TOPICS = [
    "retrieval augmented generation"]
#     "dense passage retrieval",
#     "agentic workflow llm",
#     "chain of thought reasoning",
#     "llm prompting techniques",
#     "multi hop question answering",
#     "tool augmented language models",
#     "autonomous agents large language models"
# ]


############################################
# Utilities
############################################

def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)


def build_query(search_term):
    return f"search_query=all:{quote(search_term)}&start=0&max_results={MAX_RESULTS_PER_QUERY}"


def parse_year(published):
    return int(published.split("-")[0])


def download_pdf(pdf_url, filename):
    if os.path.exists(filename):
        return

    try:
        r = requests.get(pdf_url, stream=True, timeout=20)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        time.sleep(3)  # arXiv rate limit safety
    except Exception as e:
        print(f"Failed: {pdf_url}", e)


############################################
# Main Scraper
############################################

def scrape_arxiv():
    ensure_dirs()
    metadata = []

    for topic in TOPICS:
        print(f"\nSearching for: {topic}")
        query_url = ARXIV_API + build_query(topic)
        feed = feedparser.parse(query_url)

        for entry in tqdm(feed.entries):
            year = parse_year(entry.published)

            if year < START_YEAR:
                continue

            paper_id = entry.id.split("/abs/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            filename = os.path.join(SAVE_DIR, f"{paper_id}.pdf")

            download_pdf(pdf_url, filename)

            metadata.append({
                "id": paper_id,
                "title": entry.title,
                "authors": [a.name for a in entry.authors],
                "summary": entry.summary,
                "published": entry.published,
                "pdf_url": pdf_url,
                "topic_query": topic
            })

    with open(os.path.join(SAVE_DIR, METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Downloaded {len(metadata)} papers.")


if __name__ == "__main__":
    scrape_arxiv()
