from __future__ import annotations

from UI.views import ingest_view


def test_build_pdf_embed_html_uses_blob_url_instead_of_data_iframe_src() -> None:
    html = ingest_view._build_pdf_embed_html(pdf_base64="cGRm")

    assert "URL.createObjectURL" in html
    assert 'frame.src = blobUrl + "#toolbar=1&navpanes=0&scrollbar=1&view=FitH"' in html
    assert 'src="data:application/pdf;base64,' not in html
