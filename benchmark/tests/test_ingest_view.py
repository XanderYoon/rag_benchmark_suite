from __future__ import annotations

from pathlib import Path

from UI.views import ingest_view


def test_render_pdf_viewer_passes_bytes_to_streamlit_pdf(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_bytes = b"%PDF-1.4\nmock\n"
    pdf_path.write_bytes(pdf_bytes)
    pdf_calls: list[tuple[bytes, int]] = []
    download_calls: list[dict[str, object]] = []

    def fake_pdf(data: bytes, *, height: int) -> None:
        pdf_calls.append((data, height))

    monkeypatch.setattr(ingest_view.st, "pdf", fake_pdf, raising=False)
    monkeypatch.setattr(
        ingest_view.st,
        "download_button",
        lambda label, data, file_name, mime, key: download_calls.append(
            {
                "label": label,
                "data": data,
                "file_name": file_name,
                "mime": mime,
                "key": key,
            }
        ),
    )

    ingest_view._render_pdf_viewer(pdf_path)

    assert pdf_calls == [(pdf_bytes, 720)]
    assert download_calls == [
        {
            "label": "Download PDF",
            "data": pdf_bytes,
            "file_name": "paper.pdf",
            "mime": "application/pdf",
            "key": "download_pdf_paper",
        }
    ]


def test_render_pdf_viewer_warns_when_streamlit_pdf_errors(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_bytes = b"%PDF-1.4\nmock\n"
    pdf_path.write_bytes(pdf_bytes)
    warning_messages: list[str] = []

    def failing_pdf(data: bytes, *, height: int) -> None:
        raise RuntimeError("preview failed")

    monkeypatch.setattr(ingest_view.st, "pdf", failing_pdf, raising=False)
    monkeypatch.setattr(ingest_view.st, "warning", lambda message: warning_messages.append(message))
    monkeypatch.setattr(ingest_view.st, "download_button", lambda *args, **kwargs: None)

    ingest_view._render_pdf_viewer(pdf_path)

    assert len(warning_messages) == 1
    assert "Inline PDF preview is unavailable" in warning_messages[0]
    assert "preview failed" in warning_messages[0]


def test_render_pdf_viewer_warns_when_streamlit_pdf_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nmock\n")
    warning_messages: list[str] = []

    monkeypatch.delattr(ingest_view.st, "pdf", raising=False)
    monkeypatch.setattr(ingest_view.st, "warning", lambda message: warning_messages.append(message))
    monkeypatch.setattr(ingest_view.st, "download_button", lambda *args, **kwargs: None)

    ingest_view._render_pdf_viewer(pdf_path)

    assert len(warning_messages) == 1
    assert "streamlit[pdf]" in warning_messages[0]
