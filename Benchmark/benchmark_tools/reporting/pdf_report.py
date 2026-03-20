from __future__ import annotations

from typing import Any


def build_pdf_report(*, report_model: dict[str, Any]) -> bytes:
    """Build a compact PDF report from a normalized report view model."""
    if not isinstance(report_model, dict):
        raise ValueError("Invalid report model: expected a dictionary.")

    lines = _build_report_lines(report_model=report_model)
    if not lines:
        raise ValueError("Report model is empty: expected non-empty report sections.")

    return _render_pdf(lines=lines)


def _build_report_lines(*, report_model: dict[str, Any]) -> list[str]:
    """Format report model sections into printable lines."""
    lines: list[str] = ["RAG Benchmark Report", ""]
    lines.append(f"Generated (UTC): {report_model.get('generated_at_utc', 'unknown')}")
    lines.append("")

    lines.append("Run Configuration")
    run_config = dict(report_model.get("run_config", {}))
    for key in [
        "embedded_chunks_path",
        "corpus",
        "embedding_model",
        "retrieval_model",
        "evaluation_model",
        "max_cases",
        "top_k",
    ]:
        lines.append(f"{key}: {run_config.get(key)}")
    tools = run_config.get("tools", [])
    methods = run_config.get("retrieval_methods", [])
    lines.append(f"tools: {_to_csv(tools)}")
    lines.append(f"retrieval_methods: {_to_csv(methods)}")
    lines.append("")

    lines.append("Timing")
    timing = dict(report_model.get("timing", {}))
    if timing:
        for key, value in timing.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("No timing data in snapshot.")
    lines.append("")

    lines.append("Baselines by Source/Method")
    source_baselines = list(report_model.get("source_baselines", []))
    if source_baselines:
        for row in source_baselines:
            baseline = dict(dict(row).get("baseline", {}))
            lines.append(f"source={row.get('source')} method={row.get('retrieval_method')}")
            if baseline:
                for metric, value in baseline.items():
                    lines.append(f"  {metric}: {value}")
            else:
                lines.append("  No baseline metrics.")
    else:
        lines.append("No baseline data in snapshot.")
    lines.append("")

    lines.append("Tool Summaries")
    tool_summaries = list(report_model.get("tool_summaries", []))
    if tool_summaries:
        for row in tool_summaries:
            lines.append(
                f"source={row.get('source')} method={row.get('retrieval_method')} "
                f"tool={row.get('tool')} status={row.get('status')}"
            )
            summary = dict(dict(row).get("summary", {}))
            if summary:
                for metric, value in summary.items():
                    lines.append(f"  {metric}: {value}")
            else:
                lines.append("  No numeric summary metrics.")
    else:
        lines.append("No tool summary data in snapshot.")

    return lines


def _to_csv(values: Any) -> str:
    """Convert list-like values into a compact comma-separated string."""
    if not isinstance(values, list):
        return str(values)
    return ", ".join(str(item) for item in values) if values else "(none)"


def _render_pdf(*, lines: list[str]) -> bytes:
    """Render text lines into a minimal valid PDF document."""
    lines_per_page = 46
    pages = [lines[index : index + lines_per_page] for index in range(0, len(lines), lines_per_page)]

    objects: list[bytes] = []
    page_object_ids: list[int] = []
    next_id = 1

    catalog_id = next_id
    next_id += 1
    pages_root_id = next_id
    next_id += 1
    font_id = next_id
    next_id += 1

    page_entries: list[tuple[int, int]] = []
    for _ in pages:
        page_id = next_id
        next_id += 1
        content_id = next_id
        next_id += 1
        page_entries.append((page_id, content_id))
        page_object_ids.append(page_id)

    objects.append(_obj(catalog_id, f"<< /Type /Catalog /Pages {pages_root_id} 0 R >>".encode("ascii")))
    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects.append(
        _obj(
            pages_root_id,
            f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("ascii"),
        )
    )
    objects.append(_obj(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    for page_lines, (page_id, content_id) in zip(pages, page_entries):
        content_stream = _content_stream(page_lines=page_lines)
        content_header = f"<< /Length {len(content_stream)} >>\nstream\n".encode("ascii")
        content_footer = b"\nendstream"
        objects.append(_obj(content_id, content_header + content_stream + content_footer))
        page_body = (
            f"<< /Type /Page /Parent {pages_root_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        objects.append(_obj(page_id, page_body.encode("ascii")))

    objects.sort(key=lambda item: int(item.split(b" ", 1)[0]))
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    payload = bytearray(header)
    for item in objects:
        offsets.append(len(payload))
        payload.extend(item)
        payload.extend(b"\n")

    xref_start = len(payload)
    payload.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    payload.extend(
        (
            "trailer\n"
            f"<< /Size {len(offsets)} /Root {catalog_id} 0 R >>\n"
            "startxref\n"
            f"{xref_start}\n"
            "%%EOF"
        ).encode("ascii")
    )
    return bytes(payload)


def _content_stream(*, page_lines: list[str]) -> bytes:
    """Encode page text lines into a PDF content stream."""
    stream_lines = ["BT", "/F1 10 Tf", "50 760 Td", "12 TL"]
    for line in page_lines:
        stream_lines.append(f"({_escape_pdf_text(line)}) Tj")
        stream_lines.append("T*")
    stream_lines.append("ET")
    return "\n".join(stream_lines).encode("latin-1", errors="replace")


def _escape_pdf_text(value: str) -> str:
    """Escape PDF text operators and keep printable latin-1 output."""
    escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return escaped.encode("latin-1", errors="replace").decode("latin-1")


def _obj(object_id: int, body: bytes) -> bytes:
    """Build one PDF indirect object."""
    return f"{object_id} 0 obj\n".encode("ascii") + body + b"\nendobj"
