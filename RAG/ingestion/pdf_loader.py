from __future__ import annotations

from pathlib import Path


class PdfLoader:
    def load(self, pdf_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pypdf is required for PDF extraction") from exc

        reader = PdfReader(str(pdf_path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages)
