from __future__ import annotations

from pathlib import Path

from Benchmark.config import AppConfig
from Benchmark.domain.models import Paper


class PaperService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def list_pdfs(self) -> list[Path]:
        """Return all PDF files in the configured corpus directory."""
        return sorted(self.config.corpus_dir.glob("*.pdf"))

    def list_papers(self) -> list[Paper]:
        """Return paper records derived from available PDFs."""
        papers: list[Paper] = []
        for pdf in self.list_pdfs():
            papers.append(Paper(paper_id=pdf.stem, source_path=str(pdf)))
        return papers
