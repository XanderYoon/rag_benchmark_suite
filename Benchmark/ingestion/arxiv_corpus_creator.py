from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote


DEFAULT_METADATA_FILE = "metadata.json"
DEFAULT_START_YEAR = 2019
ARXIV_API = "http://export.arxiv.org/api/query?"

ProgressCallback = Callable[[int, int, str], None]


def parse_year(published: str) -> int:
    """Extract the publication year from an arXiv timestamp."""
    return int(published.split("-", 1)[0])


def build_query(*, search_term: str, max_results: int) -> str:
    """Build the arXiv API query string for a topic."""
    return f"search_query=all:{quote(search_term)}&start=0&max_results={max_results}"


def download_pdf(*, pdf_url: str, destination_path: Path, requests_module: object, delay_seconds: float) -> bool:
    """Download one PDF unless it already exists locally."""
    if destination_path.exists():
        return False

    response = requests_module.get(pdf_url, stream=True, timeout=20)
    response.raise_for_status()
    with destination_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)
    time.sleep(delay_seconds)
    return True


def scrape_arxiv_corpus(
    *,
    topic_rows: list[dict[str, object]],
    save_dir: Path,
    start_year: int = DEFAULT_START_YEAR,
    metadata_file: str = DEFAULT_METADATA_FILE,
    delay_seconds: float = 3.0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Scrape recent arXiv PDFs for topic rows and persist metadata.

    Args:
        topic_rows: Topic dictionaries with `topic` and `docs_per_topic`.
        save_dir: Directory where PDFs and metadata should be written.
        start_year: Minimum publication year to keep.
        metadata_file: Metadata JSON filename written under `save_dir`.
        delay_seconds: Pause after each PDF download to respect rate limits.
        progress_callback: Optional callback receiving completed count, total count, and status text.

    Returns:
        Tuple of downloaded PDF count and metadata rows for all matched papers.

    Raises:
        ValueError: When topic rows or options are invalid.
        RuntimeError: When required dependencies are not installed.
    """
    if not topic_rows:
        raise ValueError("At least one topic row is required.")
    if start_year < 1991:
        raise ValueError(f"start_year must be 1991 or later, got {start_year}")
    if delay_seconds < 0:
        raise ValueError(f"delay_seconds must be non-negative, got {delay_seconds}")

    try:
        import feedparser
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("feedparser is required. Install with: pip install feedparser") from exc

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("requests is required. Install with: pip install requests") from exc

    save_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    downloaded = 0
    total_requested = sum(int(row.get("docs_per_topic", 0)) for row in topic_rows)
    processed = 0

    for row in topic_rows:
        topic = str(row.get("topic", "")).strip()
        docs_per_topic = int(row.get("docs_per_topic", 0))
        if not topic:
            raise ValueError("Each topic row must include a non-empty 'topic'.")
        if docs_per_topic <= 0:
            raise ValueError(
                f"Topic '{topic}' has invalid docs_per_topic={docs_per_topic}. Expected a positive integer."
            )

        feed = feedparser.parse(ARXIV_API + build_query(search_term=topic, max_results=docs_per_topic))
        topic_count = 0
        for entry in feed.entries:
            if parse_year(entry.published) < start_year:
                continue
            if topic_count >= docs_per_topic:
                break

            paper_id = entry.id.split("/abs/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            filename = save_dir / f"{paper_id}.pdf"
            if download_pdf(
                pdf_url=pdf_url,
                destination_path=filename,
                requests_module=requests,
                delay_seconds=delay_seconds,
            ):
                downloaded += 1

            metadata.append(
                {
                    "id": paper_id,
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "summary": entry.summary,
                    "published": entry.published,
                    "pdf_url": pdf_url,
                    "topic_query": topic,
                }
            )
            topic_count += 1
            processed += 1
            if progress_callback is not None:
                progress_callback(
                    processed,
                    total_requested,
                    f"Processed {processed}/{total_requested}: {paper_id} ({topic})",
                )

    (save_dir / metadata_file).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return downloaded, metadata
