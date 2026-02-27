from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from Benchmark.config import AppConfig
from Benchmark.domain.enums import DifficultyLabel
from Benchmark.domain.models import BenchmarkRecord, Chunk
from Benchmark.ingestion.chunk_store import ChunkStore
from Benchmark.ingestion.chunker import Chunker
from Benchmark.ingestion.pdf_loader import PdfLoader
from Benchmark.ingestion.text_cleaner import TextCleaner
from Benchmark.persistence.benchmark_store import BenchmarkStore
from Benchmark.services.paper_service import PaperService
from Benchmark.services.question_service import QuestionService
from Benchmark.verification.audit_log import AuditLog


class PipelineService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.pdf_loader = PdfLoader()
        self.text_cleaner = TextCleaner()
        self.chunker = Chunker(config.chunk_size_tokens, config.chunk_overlap_tokens)
        self.chunk_store = ChunkStore(config.chunk_dir)

        self.paper_service = PaperService(config)
        self.question_service = QuestionService(config)

        self.benchmark_store = BenchmarkStore(config)
        self.audit_log = AuditLog()

    def ingest_paper(self, pdf_path: Path) -> list[Chunk]:
        """Ingest one PDF into cached text and chunk files."""
        paper_id = pdf_path.stem
        if self.chunk_store.has_chunks(paper_id):
            existing_chunks = self.chunk_store.read_chunks(paper_id)
            self.audit_log.append(
                "paper_ingest_skipped",
                {"paper_id": paper_id, "chunk_count": len(existing_chunks)},
            )
            return existing_chunks

        text = self.pdf_loader.load(pdf_path)
        cleaned = self.text_cleaner.clean(text)

        cache_path = self.config.text_cache_dir / f"{paper_id}.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(cleaned, encoding="utf-8", errors="replace")

        chunks = self.chunker.chunk_text(paper_id=paper_id, text=cleaned)
        self.chunk_store.write_chunks(paper_id, chunks)
        self.chunk_store.update_manifest(
            manifest_path=self.config.chunk_dir / "manifest.json",
            paper_id=paper_id,
            source_path=pdf_path,
            chunk_count=len(chunks),
        )
        self.audit_log.append("paper_ingested", {"paper_id": paper_id, "chunk_count": len(chunks)})
        return chunks

    def ingest_all(
        self,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, int]:
        """Ingest all PDFs in the configured corpus directory."""
        summary: dict[str, int] = {}
        pdfs = self.paper_service.list_pdfs()
        total = len(pdfs)
        for index, pdf in enumerate(pdfs, start=1):
            summary[pdf.stem] = len(self.ingest_paper(pdf))
            if progress_callback is not None:
                progress_callback(
                    index,
                    total,
                    f"Processed {index}/{total}: {pdf.stem}",
                )
        return summary

    def load_chunks(self, paper_id: str) -> list[Chunk]:
        """Load stored chunks for a paper."""
        return self.chunk_store.read_chunks(paper_id)

    def generate_for_paper(self, paper_id: str) -> list[BenchmarkRecord]:
        """Generate benchmark questions for a paper from existing chunks."""
        chunks = self.load_chunks(paper_id)
        records = self.question_service.generate_records(paper_id, chunks)
        for record in records:
            self.benchmark_store.append_draft(record)
            self.audit_log.append("question_generated", record.to_dict())
        return records

    def regenerate_question(
        self,
        paper_id: str,
        target_difficulty: DifficultyLabel,
        slot_index: int,
        feedback: str,
        avoid_questions: list[str],
    ) -> BenchmarkRecord:
        """Regenerate one question for a paper with reviewer feedback."""
        chunks = self.load_chunks(paper_id)
        record = self.question_service.regenerate_record(
            paper_id=paper_id,
            chunks=chunks,
            target_difficulty=target_difficulty,
            slot_index=slot_index,
            feedback=feedback,
            avoid_questions=avoid_questions,
        )
        self.benchmark_store.append_draft(record)
        self.audit_log.append(
            "question_regenerated",
            {
                "paper_id": paper_id,
                "target_difficulty": target_difficulty.value,
                "slot_index": slot_index,
                "feedback": feedback,
                "record": record.to_dict(),
            },
        )
        return record
