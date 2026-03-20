from __future__ import annotations

import csv
import html
import json
import re
import time
from datetime import datetime, timezone
from math import cos, pi, sin
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import streamlit as st

from Benchmark.benchmark_tools import estimate_benchmark_runtime, run_retrieval_benchmarks
from Benchmark.benchmark_tools.probe_sources import count_verified_questions
from Benchmark.config import DEFAULT_CONFIG
from Benchmark.embedding.build_faiss_rag_index import build_faiss_index
from UI.state.session_state import (
    get_benchmark_snapshot,
    set_benchmark_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
OLLAMA_EMBEDDING_MODELS = ["nomic-embed-text"]
OPENAI_EVALUATION_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
]
DEFAULT_EVALUATION_MODEL = "gpt-4o-mini"
DEFAULT_BENCHMARK_TOOLS = ["ragas"]
RETRIEVAL_METHODS = {
    "faiss": "FAISS (required baseline)",
    "graphrag": "GraphRAG (optional)",
    "lightrag": "LightRAG (optional)",
}
VERIFIED_QUESTIONS_PATH = PROJECT_ROOT / "data" / "verified_questions.json"
METRIC_TOOLTIPS = {
    "Probe cases": "Number of synthetic retrieval test queries generated for this run.",
    "Top-k retrieved chunks": "How many chunks are returned per probe query and used to compute retrieval metrics.",
    "Cases": "Number of benchmark probe queries included in this run.",
    "Hit@1": "Share of probes where the expected chunk was the top-ranked retrieval result.",
    "Hit@3": "Share of probes where the expected chunk appeared within the top 3 retrieved results.",
    "MRR": "Mean reciprocal rank. Higher is better and rewards correct chunks appearing earlier.",
    "Avg Top Score": "Average similarity score assigned to the top retrieved chunk across all probes.",
    "context_precision": "How much of the retrieved context is relevant to the query or reference answer.",
    "context_recall": "How much of the needed reference context was successfully retrieved.",
    "faithfulness": "Whether the produced answer stays grounded in the retrieved context.",
    "answer_relevancy": "How directly the answer addresses the benchmark query.",
    "response_relevancy": "How relevant the generated response is to the prompt in RAGAS-compatible scoring.",
}
BENCHMARK_RUN_IN_PROGRESS_KEY = "benchmark_run_in_progress"
BENCHMARK_PENDING_RUN_REQUEST_KEY = "benchmark_pending_run_request"
BENCHMARK_LAST_RUN_NOTICE_KEY = "benchmark_last_run_notice"
BENCHMARK_EXPORTS_DIR = DATA_ROOT / "benchmark_runs" / "csv_exports"


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Benchmarking")
    else:
        st.subheader("Benchmarking")


def _display_path(path: Path) -> str:
    """Return a short display path relative to the repository root."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sanitize_slug(value: str) -> str:
    """Convert a free-form string into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "default"


def _find_chunk_corpora() -> list[Path]:
    """Discover chunk-root directories that contain stored chunk files."""
    roots: set[Path] = set()
    if DATA_ROOT.exists():
        for chunk_file in DATA_ROOT.rglob("*_chunk_*.txt"):
            roots.add(chunk_file.parent.parent.resolve())

    default_root = DATA_ROOT / "rag_corpus_chunked"
    if default_root.exists():
        roots.add(default_root.resolve())

    return sorted(roots)


def _find_faiss_index_dirs() -> list[Path]:
    """Discover FAISS index directories that contain required artifacts."""
    if not DATA_ROOT.exists():
        return []
    discovered: set[Path] = set()
    for manifest_path in DATA_ROOT.rglob("index_manifest.json"):
        parent = manifest_path.parent.resolve()
        if (parent / "chunks.faiss").exists() and (parent / "chunks_metadata.jsonl").exists():
            discovered.add(parent)
    return sorted(discovered)


def _render_model_picker(
    *,
    label: str,
    options: list[str],
    select_key: str,
    custom_key: str,
    disabled: bool = False,
) -> str:
    """Render a preset-or-custom model picker."""
    select_options = [*options, "Custom"]
    selected = st.selectbox(label, options=select_options, key=select_key, disabled=disabled)
    if selected == "Custom":
        custom_value = st.text_input(
            f"{label} (custom)",
            value=str(st.session_state.get(custom_key, "")),
            key=custom_key,
            disabled=disabled,
        ).strip()
        return custom_value

    st.session_state[custom_key] = selected
    return selected


def _render_fixed_model_picker(*, label: str, options: list[str], select_key: str, disabled: bool = False) -> str:
    """Render a model picker constrained to predefined options."""
    return str(st.selectbox(label, options=options, key=select_key, disabled=disabled))


def _resolve_evaluation_model(*, raw_model: str | None) -> str:
    """Resolve evaluation model with a safe default for blank UI values."""
    resolved = str(raw_model or "").strip()
    return resolved or DEFAULT_EVALUATION_MODEL


def _benchmark_enabled_providers() -> list[str]:
    """Return enabled providers configured in sidebar settings."""
    raw = st.session_state.get("llm_providers", ["openai"])
    providers = [str(provider).strip().lower() for provider in raw if str(provider).strip()]
    deduped: list[str] = []
    for provider in providers:
        if provider not in deduped and provider in {"openai", "ollama"}:
            deduped.append(provider)
    return deduped or ["openai"]


def _benchmark_embedding_model_options() -> list[str]:
    """Return embedding model options based on enabled providers."""
    providers = _benchmark_enabled_providers()
    options: list[str] = []
    if "openai" in providers:
        options.extend(OPENAI_EMBEDDING_MODELS)
    if "ollama" in providers:
        options.extend(OLLAMA_EMBEDDING_MODELS)
    return options or list(OPENAI_EMBEDDING_MODELS)


def _benchmark_evaluation_model_options() -> list[str]:
    """Return evaluation model options based on enabled providers."""
    providers = _benchmark_enabled_providers()
    options: list[str] = []
    if "openai" in providers:
        options.extend(OPENAI_EVALUATION_MODELS)
    if "ollama" in providers:
        ollama_model = str(st.session_state.get("ollama_model", "")).strip()
        if ollama_model:
            options.append(ollama_model)
    return options or list(OPENAI_EVALUATION_MODELS)


def _embedding_provider_for_model(model: str) -> str:
    """Infer embedding provider from model naming convention."""
    normalized = str(model).strip().lower()
    if normalized.startswith("text-embedding-"):
        return "openai"
    return "ollama"


def _index_output_dir(corpus_root: Path, embedding_model: str) -> Path:
    """Return the persistent index output directory for a corpus/model pair."""
    corpus_slug = _sanitize_slug(_display_path(corpus_root))
    embedding_slug = _sanitize_slug(embedding_model)
    return DATA_ROOT / "benchmark_runs" / "retrieval_indexes" / corpus_slug / embedding_slug


def _render_index_build(
    *,
    corpus_root: Path,
    output_dir: Path,
    embedding_model: str,
    rebuild_index: bool,
) -> dict[str, Any]:
    """Build or reuse a FAISS index for the selected benchmark run."""
    manifest_path = output_dir / "index_manifest.json"
    if manifest_path.exists() and not rebuild_index:
        return {
            "status": "reused",
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
        }

    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_placeholder.info("Building FAISS index for selected corpus...")

    def update_progress(progress: float, message: str) -> None:
        progress_bar.progress(min(max(progress, 0.0), 1.0))
        status_placeholder.info(message)

    try:
        embedding_provider = _embedding_provider_for_model(embedding_model)
        result = build_faiss_index(
            chunks_root=corpus_root,
            output_dir=output_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            batch_size=64,
            metric="cosine",
            overwrite=True,
            ollama_base_url=str(st.session_state.get("ollama_base_url", "http://localhost:11434")).strip(),
            progress_callback=update_progress,
        )
    except Exception:
        progress_bar.empty()
        status_placeholder.empty()
        raise

    progress_bar.progress(1.0)
    status_placeholder.success("FAISS index ready.")
    return result


def _render_baseline_metrics(result: dict[str, Any]) -> None:
    """Render baseline retrieval summary cards and charts."""
    baseline = dict(result.get("baseline", {}))
    if not baseline:
        return

    st.subheader("Information Retrieval Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cases", int(baseline.get("num_cases", 0)), help=METRIC_TOOLTIPS["Cases"])
    c2.metric("Hit@1", _format_score(baseline.get("hit_at_1", 0.0)), help=METRIC_TOOLTIPS["Hit@1"])
    c3.metric("Hit@3", _format_score(baseline.get("hit_at_3", 0.0)), help=METRIC_TOOLTIPS["Hit@3"])
    c4.metric("MRR", _format_score(baseline.get("mrr", 0.0)), help=METRIC_TOOLTIPS["MRR"])
    c5.metric(
        "Avg Top Score",
        _format_score(baseline.get("average_top_score", 0.0)),
        help=METRIC_TOOLTIPS["Avg Top Score"],
    )

    chart_rows = [
        {"metric": "Hit@1", "value": float(baseline.get("hit_at_1", 0.0))},
        {"metric": "Hit@3", "value": float(baseline.get("hit_at_3", 0.0))},
        {"metric": "MRR", "value": float(baseline.get("mrr", 0.0))},
        {"metric": "Avg Top Score", "value": float(baseline.get("average_top_score", 0.0))},
    ]
    chart_mode = _render_chart_mode_toggle(
        label="Chart type",
        key="ir_metrics_chart_mode",
    )
    _render_metric_chart(
        metric_rows=chart_rows,
        chart_mode=chart_mode,
        chart_key="ir_metrics_chart",
        color="#1f77b4",
    )
    _render_metric_guide(["Hit@1", "Hit@3", "MRR", "Avg Top Score"])


def _render_tool_results(result: dict[str, Any]) -> None:
    """Render per-tool benchmark summaries and status messages."""
    tool_results = dict(result.get("tool_results", {}))
    if not tool_results:
        return

    st.subheader("RAGAS Metrics")
    summary_rows: list[dict[str, Any]] = []
    for tool_name, tool_result in tool_results.items():
        status = str(tool_result.get("status", "unknown"))
        title = tool_name.upper()
        if status == "completed":
            st.markdown(f"**{title}**")
            numeric_summary = {
                key: float(value)
                for key, value in dict(tool_result.get("summary", {})).items()
                if isinstance(value, (int, float))
            }
            if numeric_summary:
                metric_columns = st.columns(min(len(numeric_summary), 4))
                for index, (key, value) in enumerate(numeric_summary.items()):
                    metric_columns[index % len(metric_columns)].metric(
                        key,
                        _format_score(value),
                        help=METRIC_TOOLTIPS.get(key, f"{key} score reported by {title}."),
                    )
                for key, value in numeric_summary.items():
                    summary_rows.append(
                        {
                            "tool": tool_name,
                            "metric": key,
                            "value": value,
                        }
                    )
                _render_metric_guide(list(numeric_summary.keys()))
        elif status == "skipped":
            st.warning(f"{title}: {tool_result.get('details', {}).get('message', 'Skipped.')}")
            _render_tool_debug_details(tool_name=tool_name, tool_result=tool_result)
        else:
            details = dict(tool_result.get("details", {}))
            error_text = str(details.get("error", "Benchmark failed."))
            if "timeout" in error_text.lower():
                st.warning(
                    f"{title}: timeout detected while running the evaluator. "
                    "Open debug details below for model/backend context and traceback."
                )
            st.error(f"{title}: {error_text}")
            _render_tool_debug_details(tool_name=tool_name, tool_result=tool_result)

    if summary_rows:
        chart_mode = _render_chart_mode_toggle(
            label="Chart type",
            key="ragas_metrics_chart_mode",
        )
        _render_metric_chart(
            metric_rows=summary_rows,
            chart_mode=chart_mode,
            chart_key="ragas_metrics_chart",
            color="#2a9d8f",
        )


def _render_case_chart(result: dict[str, Any]) -> None:
    """Render benchmark case details."""
    cases = list(result.get("cases", []))
    if not cases:
        return

    with st.expander("Benchmark Case Details"):
        st.caption(
            "Why a score might be `None`: RAGAS may skip a probe-level metric when that row does not "
            "produce a usable numeric value, for example due to weak retrieved context, missing support, "
            "or a row-level evaluator failure."
        )
        st.dataframe(_build_case_detail_rows(result=result, cases=cases), width="stretch")


def _render_tool_debug_details(*, tool_name: str, tool_result: dict[str, Any]) -> None:
    """Render structured debug details for skipped or failed tool executions."""
    details = dict(tool_result.get("details", {}))
    if not details:
        return

    debug_rows: list[dict[str, Any]] = []
    error_type = str(details.get("error_type", "")).strip()
    error_message = str(details.get("error_message", "")).strip()
    debug_log_file = str(details.get("debug_log_file", "")).strip()
    debug_context = details.get("debug_context")
    if error_type:
        debug_rows.append({"field": "error_type", "value": error_type})
    if error_message:
        debug_rows.append({"field": "error_message", "value": error_message})
    if debug_log_file:
        debug_rows.append({"field": "debug_log_file", "value": debug_log_file})
    if isinstance(debug_context, dict):
        for field_name in (
            "retrieval_model",
            "evaluation_model",
            "retrieval_case_count",
            "metric_names",
        ):
            if field_name in debug_context:
                debug_rows.append({"field": field_name, "value": debug_context.get(field_name)})

    if not debug_rows and not str(details.get("traceback", "")).strip():
        return

    with st.expander(f"{tool_name.upper()} debug details", expanded=False):
        if debug_rows:
            st.dataframe(debug_rows, width="stretch")
        traceback_text = str(details.get("traceback", "")).strip()
        if traceback_text:
            st.code(traceback_text, language="text")


def _format_score(value: Any) -> str:
    """Format numeric scores for metric display."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _render_chart_mode_toggle(*, label: str, key: str) -> str:
    """Render a compact chart-type toggle."""
    return str(
        st.radio(
            label,
            options=["Bar", "Radar"],
            horizontal=True,
            key=key,
        )
    ).strip().lower()


def _render_metric_chart(
    *,
    metric_rows: list[dict[str, Any]],
    chart_mode: str,
    chart_key: str,
    color: str,
) -> None:
    """Render either a bar chart or a radar chart for metric rows."""
    if chart_mode == "radar":
        _render_radar_chart(metric_rows=metric_rows, chart_key=chart_key, color=color)
        return

    st.vega_lite_chart(
        {
            "data": {"values": metric_rows},
            "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
            "encoding": {
                "x": {"field": "metric", "type": "nominal", "axis": {"labelAngle": 0}},
                "y": {"field": "value", "type": "quantitative", "scale": {"domain": [0, 1]}},
                "color": {"value": color, "legend": None},
            },
            "height": 300,
        },
        width="stretch",
    )


def _render_radar_chart(*, metric_rows: list[dict[str, Any]], chart_key: str, color: str) -> None:
    """Render a simple radar chart by projecting metrics into 2D coordinates."""
    if not metric_rows:
        return

    polygon_points, label_points, ring_points = _build_radar_chart_rows(metric_rows=metric_rows)
    st.vega_lite_chart(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "width": 420,
            "height": 420,
            "layer": [
                {
                    "data": {"values": ring_points},
                    "mark": {"type": "line", "stroke": "#d1d5db"},
                    "encoding": {
                        "detail": {"field": "ring"},
                        "x": {"field": "x", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "y": {"field": "y", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "order": {"field": "order"},
                    },
                },
                {
                    "data": {"values": polygon_points},
                    "mark": {"type": "area", "color": color, "opacity": 0.2},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "y": {"field": "y", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "order": {"field": "order"},
                    },
                },
                {
                    "data": {"values": polygon_points},
                    "mark": {"type": "line", "point": True, "strokeWidth": 3, "color": color},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "y": {"field": "y", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "order": {"field": "order"},
                        "tooltip": [
                            {"field": "metric", "type": "nominal"},
                            {"field": "value", "type": "quantitative", "format": ".3f"},
                        ],
                    },
                },
                {
                    "data": {"values": label_points},
                    "mark": {"type": "text", "fontSize": 12, "color": "#374151"},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "y": {"field": "y", "type": "quantitative", "axis": None, "scale": {"domain": [-1.2, 1.2]}},
                        "text": {"field": "metric"},
                    },
                },
            ],
            "config": {"view": {"stroke": None}},
        },
        width="stretch",
    )


def _build_radar_chart_rows(*, metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build polygon, label, and ring rows for the radar chart renderer."""
    total_metrics = len(metric_rows)
    polygon_points: list[dict[str, Any]] = []
    label_points: list[dict[str, Any]] = []
    ring_points: list[dict[str, Any]] = []

    for index, row in enumerate(metric_rows):
        angle = (2 * pi * index / total_metrics) - (pi / 2)
        value = max(0.0, min(float(row.get("value", 0.0)), 1.0))
        polygon_points.append(
            {
                "metric": row.get("metric"),
                "value": value,
                "x": value * cos(angle),
                "y": value * sin(angle),
                "order": index,
            }
        )
        label_points.append(
            {
                "metric": row.get("metric"),
                "x": 1.1 * cos(angle),
                "y": 1.1 * sin(angle),
            }
        )

    polygon_points.append(dict(polygon_points[0], order=total_metrics))

    steps = 36
    for ring in (0.25, 0.5, 0.75, 1.0):
        for step in range(steps + 1):
            angle = (2 * pi * step / steps) - (pi / 2)
            ring_points.append(
                {
                    "ring": f"{ring:.2f}",
                    "x": ring * cos(angle),
                    "y": ring * sin(angle),
                    "order": step,
                }
            )

    return polygon_points, label_points, ring_points


def _build_case_detail_rows(*, result: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine retrieval rows with stored per-case RAGAS metrics for detail display."""
    ragas_scores_by_case = _ragas_scores_by_case(result=result)
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        retrieved_chunks = list(case.get("retrieved_chunks", []))
        retrieved_contexts = [
            str(chunk.get("text", "")).strip()
            for chunk in retrieved_chunks
            if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
        ]
        score_row = dict(ragas_scores_by_case.get(case_id, {}))
        rows.append(
            {
                "case_id": case.get("case_id"),
                "question": case.get("query"),
                "answer": retrieved_contexts[0] if retrieved_contexts else "",
                "contexts": "\n\n".join(retrieved_contexts),
                "reference_contexts": "\n\n".join(
                    str(context).strip()
                    for context in list(case.get("reference_contexts", []))
                    if str(context).strip()
                ),
                "expected_chunk_id": case.get("expected_chunk_id"),
                "top_hit_chunk_id": case.get("top_hit_chunk_id"),
                "hit_at_1": case.get("hit_at_1"),
                "hit_at_3": case.get("hit_at_3"),
                "reciprocal_rank": case.get("reciprocal_rank"),
                "context_precision": score_row.get("context_precision"),
                "context_recall": score_row.get("context_recall"),
                "faithfulness": score_row.get("faithfulness"),
                "answer_relevancy": score_row.get("answer_relevancy"),
            }
        )
    return rows


def _ragas_scores_by_case(*, result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index stored per-case RAGAS scores by benchmark case id."""
    tool_results = dict(result.get("tool_results", {}))
    ragas_result = dict(tool_results.get("ragas", {}))
    details = dict(ragas_result.get("details", {}))
    raw_case_scores = details.get("case_scores", [])
    if not isinstance(raw_case_scores, list):
        return {}

    scores_by_case: dict[str, dict[str, Any]] = {}
    for row in raw_case_scores:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        scores_by_case[case_id] = dict(row)
    return scores_by_case


def _render_metric_guide(metric_names: list[str]) -> None:
    """Render a compact metric explanation block for visible metrics."""
    unique_names: list[str] = []
    for name in metric_names:
        if name not in unique_names:
            unique_names.append(name)

    descriptions = [
        f"`{name}`: {METRIC_TOOLTIPS[name]}"
        for name in unique_names
        if name in METRIC_TOOLTIPS
    ]
    if descriptions:
        st.caption("Metric guide: " + " | ".join(descriptions))


def _run_benchmark_with_loading_feedback(
    *,
    selected_corpus: Path,
    output_dir: Path,
    embedding_model: str,
    rebuild_index: bool,
    retrieval_model: str,
    evaluation_model: str | None,
    auto_probe_count: int,
    include_auto: bool,
    include_verified: bool,
    verified_count: int,
    selected_tools: list[str],
    selected_method: str,
    top_k: int,
    estimate_request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run index+benchmark stages with visible progress, elapsed time, and ETA."""
    progress_bar = st.progress(0.0)
    stage_message = st.empty()
    timing_message = st.empty()

    estimated_seconds = 0.0
    try:
        estimate = estimate_benchmark_runtime(request=estimate_request)
        estimated_seconds = float(estimate.get("expected_seconds", 0.0))
    except Exception:
        estimated_seconds = 0.0

    stage_message.info("🧱 Stage 1/2: Preparing retrieval index...")
    index_result = _render_index_build(
        corpus_root=selected_corpus,
        output_dir=output_dir,
        embedding_model=embedding_model,
        rebuild_index=rebuild_index,
    )
    progress_bar.progress(0.20)
    stage_message.info("🧪 Stage 2/2: Running retrieval + evaluator benchmarks...")

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_retrieval_benchmarks,
            embedded_chunks_path=output_dir,
            retrieval_model=retrieval_model,
            evaluation_model=evaluation_model,
            max_cases=auto_probe_count if include_auto else max(verified_count, 1),
            top_k=top_k,
            tools=selected_tools,
            include_auto_probes=include_auto,
            include_verified_probes=include_verified,
            verified_questions_path=VERIFIED_QUESTIONS_PATH,
            retrieval_methods=[selected_method],
        )

        while not future.done():
            elapsed = time.perf_counter() - start_time
            if estimated_seconds > 0:
                predicted_remaining = max(estimated_seconds - elapsed, 0.0)
                progress = min(0.20 + 0.75 * min(elapsed / estimated_seconds, 1.0), 0.95)
                timing_message.caption(
                    f"⏱️ Elapsed: {elapsed:.1f}s | Estimated remaining: {predicted_remaining:.1f}s"
                )
            else:
                progress = 0.30
                timing_message.caption(f"⏱️ Elapsed: {elapsed:.1f}s | Estimating remaining time...")
            progress_bar.progress(progress)
            time.sleep(0.25)

        benchmark_result = future.result()

    progress_bar.progress(1.0)
    total_elapsed = time.perf_counter() - start_time
    stage_message.success("✅ Benchmark execution completed.")
    timing_message.caption(f"⏱️ Benchmark stage runtime: {total_elapsed:.1f}s")
    return benchmark_result, index_result


def _list_source_method_options(result: dict[str, Any]) -> list[tuple[str, str]]:
    """Return available (source, retrieval_method) pairs from source results."""
    source_results = result.get("source_results", {})
    if not isinstance(source_results, dict):
        return []

    options: list[tuple[str, str]] = []
    for source_name, source_bucket in source_results.items():
        if not isinstance(source_bucket, dict):
            continue
        methods = source_bucket.get("methods", {})
        if not isinstance(methods, dict):
            continue
        for method_name, method_payload in methods.items():
            if isinstance(method_payload, dict):
                options.append((str(source_name), str(method_name)))
    return options


def _method_status_for_slice(*, result: dict[str, Any], source: str, retrieval_method: str) -> str:
    """Return execution status for one source/method slice."""
    source_results = result.get("source_results", {})
    if not isinstance(source_results, dict):
        return "unknown"
    source_bucket = source_results.get(source, {})
    if not isinstance(source_bucket, dict):
        return "unknown"
    methods = source_bucket.get("methods", {})
    if not isinstance(methods, dict):
        return "unknown"
    method_payload = methods.get(retrieval_method, {})
    if not isinstance(method_payload, dict):
        return "unknown"
    return str(method_payload.get("status", "unknown")).strip().lower() or "unknown"


def _default_slice_index(
    *,
    result: dict[str, Any],
    source_method_options: list[tuple[str, str]],
    prefer_verified: bool,
) -> int:
    """Choose a default slice, preferring completed results over failed ones."""
    if not source_method_options:
        return 0

    if prefer_verified:
        for index, (source_name, method_name) in enumerate(source_method_options):
            if source_name != "verified":
                continue
            if _method_status_for_slice(
                result=result,
                source=source_name,
                retrieval_method=method_name,
            ) == "completed":
                return index

    for index, (source_name, method_name) in enumerate(source_method_options):
        if _method_status_for_slice(
            result=result,
            source=source_name,
            retrieval_method=method_name,
        ) == "completed":
            return index

    return 0


def _build_display_result_for_source_method(
    *,
    result: dict[str, Any],
    source: str,
    retrieval_method: str,
) -> dict[str, Any]:
    """Project one source+method result into the legacy display fields used by this view."""
    source_results = result.get("source_results", {})
    if not isinstance(source_results, dict):
        return dict(result)

    source_bucket = source_results.get(source, {})
    if not isinstance(source_bucket, dict):
        return dict(result)
    methods = source_bucket.get("methods", {})
    if not isinstance(methods, dict):
        return dict(result)

    method_result = methods.get(retrieval_method, {})
    if not isinstance(method_result, dict):
        return dict(result)

    projected = dict(result)
    projected["baseline"] = dict(method_result.get("baseline", {}))
    projected["cases"] = list(method_result.get("cases", []))
    projected["tool_results"] = dict(method_result.get("tool_results", {}))
    projected["display_source"] = source
    projected["display_retrieval_method"] = retrieval_method
    projected["status"] = str(method_result.get("status", "unknown"))
    projected["error"] = method_result.get("error")
    return projected


def _render_execution_failure_details(*, result: dict[str, Any], display_result: dict[str, Any]) -> None:
    """Render clear failure reasons when selected benchmark slice did not complete."""
    selected_status = str(display_result.get("status", "completed")).strip().lower()
    if selected_status == "completed":
        return

    selected_source = str(display_result.get("display_source", "unknown"))
    selected_method = str(display_result.get("display_retrieval_method", "unknown"))
    selected_error = str(display_result.get("error", "")).strip()
    st.error(
        f"Selected slice `{selected_source} / {selected_method}` did not complete (status: {selected_status})."
    )
    if selected_error:
        st.caption(f"Error: {selected_error}")

    failed_jobs: list[dict[str, Any]] = []
    for row in list(result.get("jobs", [])):
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).strip().lower() == "completed":
            continue
        failed_jobs.append(
            {
                "job_id": row.get("job_id"),
                "source": row.get("source"),
                "retrieval_method": row.get("retrieval_method"),
                "status": row.get("status"),
                "error": row.get("error"),
            }
        )
    if failed_jobs:
        st.dataframe(failed_jobs, width="stretch")


def _load_verified_question_count(*, verified_path: Path) -> tuple[int, str | None]:
    """Load verified question count and return a user-facing error on failure."""
    try:
        return count_verified_questions(verified_path=verified_path), None
    except Exception as exc:
        return 0, str(exc)


def _build_runtime_estimate_request(
    *,
    output_dir: Path,
    retrieval_model: str,
    evaluation_model: str | None,
    selected_tools: list[str],
    include_auto: bool,
    auto_probe_count: int | None,
    include_verified: bool,
    verified_count: int,
    retrieval_method: str,
    top_k: int,
) -> dict[str, Any]:
    """Build a benchmark request payload suitable for runtime estimation."""
    return {
        "embedded_chunks_path": output_dir,
        "retrieval_model": retrieval_model,
        "evaluation_model": evaluation_model,
        "top_k": top_k,
        "tools": list(selected_tools),
        "include_auto": include_auto,
        "auto_probe_count": auto_probe_count,
        "include_verified": include_verified,
        "verified_questions_path": VERIFIED_QUESTIONS_PATH,
        "retrieval_methods": [retrieval_method],
        "estimated_verified_cases": verified_count if include_verified else 0,
    }


def _build_ui_run_signature(
    *,
    selected_corpus_label: str,
    embedding_model: str,
    retrieval_model: str,
    evaluation_model: str | None,
    include_auto: bool,
    auto_probe_count: int,
    include_verified: bool,
    verified_count: int,
    top_k: int,
    selected_tools: list[str],
    selected_method: str,
) -> dict[str, Any]:
    """Build a stable run signature used for snapshot comparison and display."""
    return {
        "corpus": selected_corpus_label,
        "embedding_model": embedding_model,
        "retrieval_model": retrieval_model,
        "evaluation_model": evaluation_model,
        "include_auto": include_auto,
        "auto_probe_count": auto_probe_count if include_auto else None,
        "include_verified": include_verified,
        "verified_available": verified_count,
        "top_k": top_k,
        "tools": list(selected_tools),
        "retrieval_methods": [selected_method],
        "retrieval_method": selected_method,
    }


def _probe_details_schema() -> list[dict[str, str]]:
    """Return the probe-details CSV schema shown before benchmark export."""
    return [
        {"column": "run_id", "type": "string", "description": "Benchmark run identifier."},
        {"column": "probe_source", "type": "string", "description": "Probe bucket used for the case."},
        {"column": "retrieval_method", "type": "string", "description": "Retriever used for this case row."},
        {"column": "case_id", "type": "string", "description": "Stable probe case identifier."},
        {"column": "query", "type": "string", "description": "Benchmark question text."},
        {"column": "expected_chunk_id", "type": "string", "description": "Expected supporting chunk id."},
        {"column": "expected_answer", "type": "string", "description": "Expected benchmark answer."},
        {"column": "top_hit_chunk_id", "type": "string", "description": "Top retrieved chunk id."},
        {"column": "hit_at_1", "type": "boolean", "description": "Whether the expected chunk ranked first."},
        {"column": "hit_at_3", "type": "boolean", "description": "Whether the expected chunk ranked in top three."},
        {"column": "reciprocal_rank", "type": "float", "description": "Reciprocal rank for the expected chunk."},
        {"column": "retrieved_chunk_count", "type": "integer", "description": "Number of retrieved chunks stored for the case."},
        {"column": "retrieved_chunk_ids_json", "type": "json-string", "description": "Ordered retrieved chunk ids."},
        {"column": "retrieved_chunk_scores_json", "type": "json-string", "description": "Ordered retrieved chunk scores."},
        {"column": "reference_contexts_json", "type": "json-string", "description": "Reference contexts used by the probe."},
        {"column": "retrieved_contexts_json", "type": "json-string", "description": "Retrieved chunk texts for the case."},
        {"column": "answer_text", "type": "string", "description": "Top retrieved chunk text used as answer surrogate."},
        {"column": "ragas_context_precision", "type": "float", "description": "Stored per-case RAGAS context precision score."},
        {"column": "ragas_context_recall", "type": "float", "description": "Stored per-case RAGAS context recall score."},
        {"column": "ragas_faithfulness", "type": "float", "description": "Stored per-case RAGAS faithfulness score."},
        {"column": "ragas_answer_relevancy", "type": "float", "description": "Stored per-case RAGAS answer relevancy score."},
        {"column": "ragas_response_relevancy", "type": "float", "description": "Stored per-case RAGAS response relevancy score."},
    ]


def _metadata_schema() -> list[dict[str, str]]:
    """Return the metadata CSV schema shown before benchmark export."""
    return [
        {"column": "run_id", "type": "string", "description": "Benchmark run identifier."},
        {"column": "run_completed_at_utc", "type": "datetime", "description": "UTC timestamp when the benchmark finished."},
        {"column": "corpus", "type": "string", "description": "Selected corpus label."},
        {"column": "embedding_model", "type": "string", "description": "Embedding model used to build/reuse the index."},
        {"column": "retrieval_model", "type": "string", "description": "Embedding model used for retrieval queries."},
        {"column": "evaluation_model", "type": "string", "description": "Evaluation model passed to RAGAS."},
        {"column": "probe_source", "type": "string", "description": "Probe source for this metadata row."},
        {"column": "probe_sources_json", "type": "json-string", "description": "Enabled probe sources for the run."},
        {"column": "retrieval_method", "type": "string", "description": "Retriever evaluated for this row."},
        {"column": "tools_json", "type": "json-string", "description": "Benchmark tool ids used in the run."},
        {"column": "chunk_size_tokens", "type": "integer", "description": "Configured chunk size token count."},
        {"column": "chunk_overlap_tokens", "type": "integer", "description": "Configured chunk overlap token count."},
        {"column": "benchmark_duration_seconds", "type": "float", "description": "Total benchmark runtime from telemetry."},
        {"column": "job_duration_seconds", "type": "float", "description": "Duration of the source/method branch."},
        {"column": "top_k", "type": "integer", "description": "Retrieved chunks requested per probe."},
        {"column": "rebuild_index_used", "type": "boolean", "description": "Whether the index was rebuilt before the run."},
        {"column": "index_output_dir", "type": "string", "description": "FAISS directory used for the run."},
        {"column": "index_build_status", "type": "string", "description": "Index build outcome for the run."},
        {"column": "auto_probe_cases_used", "type": "integer", "description": "Number of auto probe cases included."},
        {"column": "verified_probes_used", "type": "integer", "description": "Number of verified probes included."},
        {"column": "total_cases", "type": "integer", "description": "Total cases in the benchmark run."},
        {"column": "status", "type": "string", "description": "Job completion status."},
    ]


def _render_csv_export_schema_preview() -> None:
    """Show the CSV schemas before any benchmark artifacts are written."""
    with st.expander("CSV Export Schemas", expanded=False):
        st.caption(
            "These are the exact columns written after a fully successful benchmark run. "
            "Token columns reflect chunk configuration, not provider token billing."
        )
        st.markdown("**Probe details CSV**")
        st.dataframe(_probe_details_schema(), width="stretch")
        st.markdown("**Metadata CSV**")
        st.dataframe(_metadata_schema(), width="stretch")


def _serialize_json_cell(value: Any) -> str:
    """Convert list-like benchmark fields into stable CSV-safe JSON strings."""
    return json.dumps(value, ensure_ascii=True)


def _is_successful_benchmark_result(*, result: dict[str, Any]) -> bool:
    """Return True when every benchmark job completed successfully."""
    jobs = result.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        return False
    return all(str(job.get("status", "")).strip().lower() == "completed" for job in jobs if isinstance(job, dict))


def _build_probe_detail_export_rows(*, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten completed benchmark cases into probe-detail CSV rows."""
    rows: list[dict[str, Any]] = []
    for source_name, source_bucket in dict(result.get("source_results", {})).items():
        if not isinstance(source_bucket, dict):
            continue
        methods = dict(source_bucket.get("methods", {}))
        for retrieval_method, method_payload in methods.items():
            if not isinstance(method_payload, dict):
                continue
            if str(method_payload.get("status", "")).strip().lower() != "completed":
                continue
            ragas_scores_by_case = _ragas_scores_by_case(result=method_payload)
            for case in list(method_payload.get("cases", [])):
                if not isinstance(case, dict):
                    continue
                retrieved_chunks = [
                    chunk for chunk in list(case.get("retrieved_chunks", [])) if isinstance(chunk, dict)
                ]
                retrieved_contexts = [str(chunk.get("text", "")) for chunk in retrieved_chunks]
                ragas_scores = dict(ragas_scores_by_case.get(str(case.get("case_id", "")), {}))
                rows.append(
                    {
                        "run_id": str(result.get("timing", {}).get("run_id", "")),
                        "probe_source": str(source_name),
                        "retrieval_method": str(retrieval_method),
                        "case_id": str(case.get("case_id", "")),
                        "query": str(case.get("query", "")),
                        "expected_chunk_id": str(case.get("expected_chunk_id", "")),
                        "expected_answer": str(case.get("expected_answer", "")),
                        "top_hit_chunk_id": str(case.get("top_hit_chunk_id", "")),
                        "hit_at_1": bool(case.get("hit_at_1", False)),
                        "hit_at_3": bool(case.get("hit_at_3", False)),
                        "reciprocal_rank": float(case.get("reciprocal_rank", 0.0) or 0.0),
                        "retrieved_chunk_count": len(retrieved_chunks),
                        "retrieved_chunk_ids_json": _serialize_json_cell(
                            [str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks]
                        ),
                        "retrieved_chunk_scores_json": _serialize_json_cell(
                            [chunk.get("score") for chunk in retrieved_chunks]
                        ),
                        "reference_contexts_json": _serialize_json_cell(list(case.get("reference_contexts", []))),
                        "retrieved_contexts_json": _serialize_json_cell(retrieved_contexts),
                        "answer_text": retrieved_contexts[0] if retrieved_contexts else "",
                        "ragas_context_precision": ragas_scores.get("context_precision"),
                        "ragas_context_recall": ragas_scores.get("context_recall"),
                        "ragas_faithfulness": ragas_scores.get("faithfulness"),
                        "ragas_answer_relevancy": ragas_scores.get("answer_relevancy"),
                        "ragas_response_relevancy": ragas_scores.get("response_relevancy"),
                    }
                )
    return rows


def _build_metadata_export_rows(*, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten completed benchmark jobs into metadata CSV rows."""
    signature = dict(result.get("ui_run_signature", {}))
    timing = dict(result.get("timing", {}))
    index_build = dict(result.get("index_build", {}))
    probe_breakdown = dict(result.get("probe_source_breakdown", {}))
    enabled_sources = [
        source_name
        for source_name, enabled in (
            ("auto", bool(signature.get("include_auto", True))),
            ("verified", bool(signature.get("include_verified", False))),
        )
        if enabled
    ]

    rows: list[dict[str, Any]] = []
    for job in list(result.get("jobs", [])):
        if not isinstance(job, dict):
            continue
        if str(job.get("status", "")).strip().lower() != "completed":
            continue
        rows.append(
            {
                "run_id": str(timing.get("run_id", "")),
                "run_completed_at_utc": str(result.get("run_completed_at_utc", "")),
                "corpus": str(signature.get("corpus", "")),
                "embedding_model": str(signature.get("embedding_model", "")),
                "retrieval_model": str(result.get("retrieval_model", "")),
                "evaluation_model": str(result.get("evaluation_model", "")),
                "probe_source": str(job.get("source", "")),
                "probe_sources_json": _serialize_json_cell(enabled_sources),
                "retrieval_method": str(job.get("retrieval_method", "")),
                "tools_json": _serialize_json_cell(list(signature.get("tools", []))),
                "chunk_size_tokens": int(DEFAULT_CONFIG.chunk_size_tokens),
                "chunk_overlap_tokens": int(DEFAULT_CONFIG.chunk_overlap_tokens),
                "benchmark_duration_seconds": float(timing.get("actual_total_seconds", 0.0) or 0.0),
                "job_duration_seconds": float(job.get("duration_seconds", 0.0) or 0.0),
                "top_k": int(signature.get("top_k", 0) or 0),
                "rebuild_index_used": bool(index_build.get("status") != "reused"),
                "index_output_dir": str(index_build.get("output_dir", "")),
                "index_build_status": str(index_build.get("status", "")),
                "auto_probe_cases_used": int(probe_breakdown.get("auto_cases", 0) or 0),
                "verified_probes_used": int(probe_breakdown.get("verified_cases", 0) or 0),
                "total_cases": int(probe_breakdown.get("total_cases", 0) or 0),
                "status": str(job.get("status", "")),
            }
        )
    return rows


def _write_csv(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with a stable header order derived from the first row."""
    if not rows:
        raise ValueError(f"No rows available for CSV export: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_benchmark_csvs(*, result: dict[str, Any]) -> dict[str, Any]:
    """Persist probe-details and metadata CSV artifacts for a successful run."""
    run_id = str(result.get("timing", {}).get("run_id", "")).strip()
    if not run_id:
        raise ValueError("Benchmark result is missing timing.run_id required for CSV export.")

    probe_rows = _build_probe_detail_export_rows(result=result)
    metadata_rows = _build_metadata_export_rows(result=result)
    if not probe_rows or not metadata_rows:
        raise ValueError("Benchmark result did not contain completed rows for CSV export.")

    probe_path = BENCHMARK_EXPORTS_DIR / f"{run_id}_probe_details.csv"
    metadata_path = BENCHMARK_EXPORTS_DIR / f"{run_id}_metadata.csv"
    _write_csv(path=probe_path, rows=probe_rows)
    _write_csv(path=metadata_path, rows=metadata_rows)
    return {
        "probe_details_csv": str(probe_path),
        "metadata_csv": str(metadata_path),
        "probe_details_columns": list(probe_rows[0].keys()),
        "metadata_columns": list(metadata_rows[0].keys()),
    }


def _read_csv_rows(*, path: Path) -> list[dict[str, str]]:
    """Read CSV rows into dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _compare_run_label(*, metadata_row: dict[str, str]) -> str:
    """Build a human-readable compare label for one exported run."""
    run_completed = str(metadata_row.get("run_completed_at_utc", "")).strip() or "unknown-time"
    corpus = str(metadata_row.get("corpus", "")).strip() or "unknown-corpus"
    retrieval_model = str(metadata_row.get("retrieval_model", "")).strip() or "unknown-model"
    run_id = str(metadata_row.get("run_id", "")).strip()
    short_run_id = run_id.replace("benchmark_", "")[:8] if run_id else "unknown"
    return f"{run_completed} | {corpus} | {retrieval_model} | {short_run_id}"


def _load_exported_run_catalog() -> list[dict[str, Any]]:
    """Load exported benchmark runs available for comparison."""
    runs: list[dict[str, Any]] = []
    if not BENCHMARK_EXPORTS_DIR.exists():
        return runs

    for metadata_path in sorted(BENCHMARK_EXPORTS_DIR.glob("*_metadata.csv"), reverse=True):
        try:
            metadata_rows = _read_csv_rows(path=metadata_path)
        except Exception:
            continue
        if not metadata_rows:
            continue
        first_row = metadata_rows[0]
        run_id = str(first_row.get("run_id", "")).strip()
        if not run_id:
            continue
        probe_path = BENCHMARK_EXPORTS_DIR / f"{run_id}_probe_details.csv"
        runs.append(
            {
                "run_id": run_id,
                "label": _compare_run_label(metadata_row=first_row),
                "run_completed_at_utc": str(first_row.get("run_completed_at_utc", "")),
                "metadata_path": metadata_path,
                "probe_path": probe_path,
                "metadata_preview": dict(first_row),
            }
        )
    return runs


def _to_float(value: Any) -> float | None:
    """Convert CSV cell to float when possible."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _to_bool(value: Any) -> bool:
    """Convert stored string/bool values into booleans."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: list[float]) -> float:
    """Return arithmetic mean with zero fallback."""
    return sum(values) / float(len(values)) if values else 0.0


def _build_compare_summary_rows(
    *,
    selected_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate exported run artifacts into comparison rows."""
    rows: list[dict[str, Any]] = []
    for run_entry in selected_runs:
        metadata_rows = _read_csv_rows(path=Path(str(run_entry["metadata_path"])))
        probe_rows = _read_csv_rows(path=Path(str(run_entry["probe_path"])))
        if not metadata_rows or not probe_rows:
            continue

        first_metadata = metadata_rows[0]
        reciprocal_ranks = [value for value in (_to_float(row.get("reciprocal_rank")) for row in probe_rows) if value is not None]
        context_precision = [value for value in (_to_float(row.get("ragas_context_precision")) for row in probe_rows) if value is not None]
        context_recall = [value for value in (_to_float(row.get("ragas_context_recall")) for row in probe_rows) if value is not None]
        faithfulness = [value for value in (_to_float(row.get("ragas_faithfulness")) for row in probe_rows) if value is not None]
        answer_relevancy = [value for value in (_to_float(row.get("ragas_answer_relevancy")) for row in probe_rows) if value is not None]
        response_relevancy = [value for value in (_to_float(row.get("ragas_response_relevancy")) for row in probe_rows) if value is not None]
        hit_at_1_rate = _mean([1.0 if _to_bool(row.get("hit_at_1")) else 0.0 for row in probe_rows])
        hit_at_3_rate = _mean([1.0 if _to_bool(row.get("hit_at_3")) else 0.0 for row in probe_rows])
        total_job_duration = sum(
            value
            for value in (_to_float(row.get("job_duration_seconds")) for row in metadata_rows)
            if value is not None
        )

        rows.append(
            {
                "label": str(run_entry["label"]),
                "run_id": str(run_entry["run_id"]),
                "run_completed_at_utc": str(first_metadata.get("run_completed_at_utc", "")),
                "corpus": str(first_metadata.get("corpus", "")),
                "embedding_model": str(first_metadata.get("embedding_model", "")),
                "retrieval_model": str(first_metadata.get("retrieval_model", "")),
                "evaluation_model": str(first_metadata.get("evaluation_model", "")),
                "probe_sources": str(first_metadata.get("probe_sources_json", "")),
                "retrieval_methods": _serialize_json_cell(
                    sorted({str(row.get("retrieval_method", "")) for row in metadata_rows if str(row.get("retrieval_method", "")).strip()})
                ),
                "cases": len(probe_rows),
                "auto_probe_cases_used": int(_to_float(first_metadata.get("auto_probe_cases_used")) or 0),
                "verified_probes_used": int(_to_float(first_metadata.get("verified_probes_used")) or 0),
                "top_k": int(_to_float(first_metadata.get("top_k")) or 0),
                "benchmark_duration_seconds": float(_to_float(first_metadata.get("benchmark_duration_seconds")) or 0.0),
                "job_duration_seconds_total": float(total_job_duration),
                "hit_at_1": hit_at_1_rate,
                "hit_at_3": hit_at_3_rate,
                "mrr": _mean(reciprocal_ranks),
                "context_precision": _mean(context_precision),
                "context_recall": _mean(context_recall),
                "faithfulness": _mean(faithfulness),
                "answer_relevancy": _mean(answer_relevancy),
                "response_relevancy": _mean(response_relevancy),
            }
        )
    return rows


def _compare_color_palette() -> list[str]:
    """Return a stable high-contrast palette for compare dashboards."""
    return [
        "#0f766e",
        "#c2410c",
        "#1d4ed8",
        "#b45309",
        "#be123c",
    ]


def _assign_compare_colors(*, labels: list[str]) -> dict[str, str]:
    """Assign one consistent color per selected run label."""
    palette = _compare_color_palette()
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(labels)
    }


def _assign_compare_aliases(*, labels: list[str]) -> dict[str, str]:
    """Assign compact Run A / Run B aliases in display order."""
    aliases: dict[str, str] = {}
    for index, label in enumerate(labels):
        aliases[label] = f"Run {chr(ord('A') + index)}"
    return aliases


def _render_compare_legend(
    *,
    color_by_label: dict[str, str],
    alias_by_label: dict[str, str],
) -> None:
    """Render a compact legend so colors stay interpretable across charts."""
    if not color_by_label:
        return

    legend_html = "".join(
        (
            "<div style='display:flex;align-items:center;gap:0.5rem;"
            "padding:0.45rem 0.65rem;border:1px solid #e5e7eb;border-radius:999px;"
            "background:#ffffff;'>"
            f"<span style='width:0.9rem;height:0.9rem;border-radius:999px;background:{color};display:inline-block;'></span>"
            f"<span style='font-size:0.9rem;color:#111827;font-weight:700;'>{alias_by_label.get(label, label)}</span>"
            f"<span style='font-size:0.85rem;color:#4b5563;'>{label}</span>"
            "</div>"
        )
        for label, color in color_by_label.items()
    )
    st.markdown(
        (
            "<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin:0.25rem 0 0.75rem 0;'>"
            f"{legend_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_compare_dashboard_chart(
    *,
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str,
    color_by_label: dict[str, str],
    height: int = 320,
    y_domain: list[float] | None = None,
) -> None:
    """Render a grouped dashboard bar chart with stable run colors."""
    if not rows:
        return

    color_domain = list(color_by_label.keys())
    color_range = [color_by_label[label] for label in color_domain]
    encoding_y: dict[str, Any] = {"field": y_field, "type": "quantitative"}
    if y_domain is not None:
        encoding_y["scale"] = {"domain": y_domain}

    st.vega_lite_chart(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": rows},
            "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
            "encoding": {
                "x": {
                    "field": x_field,
                    "type": "nominal",
                    "axis": {
                        "labelAngle": 0,
                        "title": x_field,
                        "labelLimit": 1000,
                    },
                },
                "xOffset": {"field": "run"},
                "y": {
                    **encoding_y,
                    "axis": {
                        "title": y_field,
                        "labelLimit": 1000,
                    },
                },
                "color": {
                    "field": "run",
                    "type": "nominal",
                    "scale": {"domain": color_domain, "range": color_range},
                    "legend": None,
                },
                "tooltip": [
                    {"field": "run", "type": "nominal"},
                    {"field": x_field, "type": "nominal"},
                    {"field": y_field, "type": "quantitative", "format": ".3f"},
                ],
            },
            "height": height,
            "title": title,
            "config": {"view": {"stroke": None}},
        },
        width="stretch",
    )


def _build_compare_export_html(
    *,
    summary_rows: list[dict[str, Any]],
    alias_by_label: dict[str, str],
    color_by_label: dict[str, str],
) -> str:
    """Build a standalone HTML snapshot of the compare dashboard."""
    legend_items = []
    for row in summary_rows:
        label = str(row["label"])
        alias = alias_by_label.get(label, label)
        color = color_by_label.get(label, "#1f77b4")
        legend_items.append(
            "<div style='display:flex;align-items:center;gap:10px;padding:12px 14px;"
            "border:1px solid #dbe4ea;border-radius:14px;background:#fff;'>"
            f"<span style='width:14px;height:14px;border-radius:999px;background:{color};display:inline-block;'></span>"
            f"<strong>{html.escape(alias)}</strong>"
            f"<span style='color:#475569;'>{html.escape(label)}</span>"
            "</div>"
        )

    def _metric_block(title: str, value: str) -> str:
        return (
            "<div style='padding:16px;border:1px solid #dbe4ea;border-radius:16px;background:#fff;'>"
            f"<div style='font-size:12px;letter-spacing:0.04em;color:#64748b;text-transform:uppercase;'>{html.escape(title)}</div>"
            f"<div style='margin-top:8px;font-size:22px;font-weight:700;color:#0f172a;'>{html.escape(value)}</div>"
            "</div>"
        )

    best_hit = max(summary_rows, key=lambda row: float(row.get("hit_at_1", 0.0) or 0.0))
    best_mrr = max(summary_rows, key=lambda row: float(row.get("mrr", 0.0) or 0.0))
    fastest = min(summary_rows, key=lambda row: float(row.get("benchmark_duration_seconds", 0.0) or 0.0))
    best_faithfulness = max(summary_rows, key=lambda row: float(row.get("faithfulness", 0.0) or 0.0))

    quality_headers = ["Run", "Hit@1", "Hit@3", "MRR", "Ctx Precision", "Ctx Recall", "Faithfulness", "Ans Relevancy", "Resp Relevancy"]
    quality_rows = []
    for row in summary_rows:
        alias = alias_by_label.get(str(row["label"]), str(row["label"]))
        color = color_by_label.get(str(row["label"]), "#1f77b4")
        quality_rows.append(
            "<tr>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'><span style='display:inline-flex;align-items:center;gap:8px;'><span style='width:10px;height:10px;border-radius:999px;background:{color};display:inline-block;'></span><strong>{html.escape(alias)}</strong></span></td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('hit_at_1', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('hit_at_3', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('mrr', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('context_precision', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('context_recall', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('faithfulness', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('answer_relevancy', 0.0) or 0.0):.3f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('response_relevancy', 0.0) or 0.0):.3f}</td>"
            "</tr>"
        )

    config_headers = ["Run", "Duration (s)", "Top-k", "Auto Probes", "Verified Probes", "Cases", "Embedding", "Retrieval", "Evaluation"]
    config_rows = []
    for row in summary_rows:
        alias = alias_by_label.get(str(row["label"]), str(row["label"]))
        color = color_by_label.get(str(row["label"]), "#1f77b4")
        config_rows.append(
            "<tr>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'><span style='display:inline-flex;align-items:center;gap:8px;'><span style='width:10px;height:10px;border-radius:999px;background:{color};display:inline-block;'></span><strong>{html.escape(alias)}</strong></span></td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{float(row.get('benchmark_duration_seconds', 0.0) or 0.0):.1f}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{int(row.get('top_k', 0) or 0)}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{int(row.get('auto_probe_cases_used', 0) or 0)}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{int(row.get('verified_probes_used', 0) or 0)}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{int(row.get('cases', 0) or 0)}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{html.escape(str(row.get('embedding_model', '')))}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{html.escape(str(row.get('retrieval_model', '')))}</td>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #e5e7eb;'>{html.escape(str(row.get('evaluation_model', '')))}</td>"
            "</tr>"
        )

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Benchmark Compare Dashboard</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:32px;}"
        "h1,h2{margin:0 0 16px 0;} .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;}"
        ".legend{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0 26px 0;} .section{margin-top:28px;}"
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dbe4ea;border-radius:16px;overflow:hidden;}"
        "th{background:#eef6fb;color:#0f172a;text-align:left;padding:12px;border-bottom:1px solid #dbe4ea;font-size:13px;}"
        "td{font-size:13px;} .subtle{color:#475569;margin-top:6px;}"
        "</style></head><body>"
        "<h1>Benchmark Compare Dashboard</h1>"
        "<div class='subtle'>Downloaded comparison snapshot from the Streamlit compare view.</div>"
        f"<div class='legend'>{''.join(legend_items)}</div>"
        "<div class='grid'>"
        f"{_metric_block('Best Hit@1', alias_by_label.get(str(best_hit['label']), str(best_hit['label'])))}"
        f"{_metric_block('Best MRR', alias_by_label.get(str(best_mrr['label']), str(best_mrr['label'])))}"
        f"{_metric_block('Fastest Runtime', alias_by_label.get(str(fastest['label']), str(fastest['label'])))}"
        f"{_metric_block('Highest Faithfulness', alias_by_label.get(str(best_faithfulness['label']), str(best_faithfulness['label'])))}"
        "</div>"
        "<div class='section'><h2>Quality Metrics</h2><table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in quality_headers)
        + "</tr></thead><tbody>"
        + "".join(quality_rows)
        + "</tbody></table></div>"
        "<div class='section'><h2>Runtime and Configuration</h2><table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in config_headers)
        + "</tr></thead><tbody>"
        + "".join(config_rows)
        + "</tbody></table></div>"
        "</body></html>"
    )


def _render_compare_view(*, show_title: bool) -> None:
    """Render historical benchmark comparison from stored CSV artifacts."""
    _show_title(show_title)
    st.caption("Compare up to five exported benchmark runs using the saved CSV artifacts.")

    exported_runs = _load_exported_run_catalog()
    if not exported_runs:
        st.info("No exported benchmark runs found yet. Complete a successful benchmark run first.")
        return

    run_options = {str(entry["label"]): entry for entry in exported_runs}
    default_labels = list(run_options.keys())[: min(2, len(run_options))]
    selected_labels = st.multiselect(
        "Runs to compare",
        options=list(run_options.keys()),
        default=default_labels,
        max_selections=5,
        key="benchmark_compare_runs",
        help="Select between one and five completed benchmark runs.",
    )
    if not selected_labels:
        st.info("Select at least one run to compare.")
        return

    selected_runs = [run_options[label] for label in selected_labels]
    summary_rows = _build_compare_summary_rows(selected_runs=selected_runs)
    if not summary_rows:
        st.warning("Selected runs did not produce readable compare data.")
        return

    ordered_labels = [str(row["label"]) for row in summary_rows]
    color_by_label = _assign_compare_colors(labels=ordered_labels)
    alias_by_label = _assign_compare_aliases(labels=ordered_labels)
    for row in summary_rows:
        row["run_alias"] = alias_by_label.get(str(row["label"]), str(row["label"]))

    st.subheader("Dashboard")
    _render_compare_legend(color_by_label=color_by_label, alias_by_label=alias_by_label)

    export_html = _build_compare_export_html(
        summary_rows=summary_rows,
        alias_by_label=alias_by_label,
        color_by_label=color_by_label,
    )
    st.download_button(
        label="Download Dashboard Snapshot",
        data=export_html,
        file_name="benchmark_compare_dashboard.html",
        mime="text/html",
        key="benchmark_compare_download",
        width="stretch",
    )

    headline_metrics = [
        ("Best Hit@1", max(summary_rows, key=lambda row: float(row.get("hit_at_1", 0.0) or 0.0))),
        ("Best MRR", max(summary_rows, key=lambda row: float(row.get("mrr", 0.0) or 0.0))),
        (
            "Fastest Runtime",
            min(summary_rows, key=lambda row: float(row.get("benchmark_duration_seconds", 0.0) or 0.0)),
        ),
        (
            "Highest Faithfulness",
            max(summary_rows, key=lambda row: float(row.get("faithfulness", 0.0) or 0.0)),
        ),
    ]
    metric_columns = st.columns(len(headline_metrics))
    for index, (title, row) in enumerate(headline_metrics):
        metric_columns[index].metric(title, str(row["run_alias"]))

    st.subheader("Top-k")
    top_k_columns = st.columns(len(summary_rows))
    for index, row in enumerate(summary_rows):
        top_k_columns[index].metric(str(row["run_alias"]), int(row.get("top_k", 0) or 0))

    st.subheader("Run Comparison")
    st.dataframe(summary_rows, width="stretch")

    quality_chart_rows = [
        {
            "run": str(row["run_alias"]),
            "metric": metric_name,
            "value": float(row.get(metric_name, 0.0) or 0.0),
        }
        for row in summary_rows
        for metric_name in [
            "hit_at_1",
            "hit_at_3",
            "mrr",
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "response_relevancy",
        ]
    ]
    duration_chart_rows = [
        {
            "run": str(row["run_alias"]),
            "metric": "benchmark_duration_seconds",
            "value": float(row.get("benchmark_duration_seconds", 0.0) or 0.0),
        }
        for row in summary_rows
    ]
    probe_mix_chart_rows = [
        {
            "run": str(row["run_alias"]),
            "metric": metric_name,
            "value": float(row.get(metric_name, 0.0) or 0.0),
        }
        for row in summary_rows
        for metric_name in ["auto_probe_cases_used", "verified_probes_used"]
    ]

    _render_compare_dashboard_chart(
        rows=quality_chart_rows,
        x_field="metric",
        y_field="value",
        title="Retrieval and RAGAS Metrics",
        color_by_label={alias_by_label[label]: color for label, color in color_by_label.items()},
        y_domain=[0, 1],
        height=340,
    )

    duration_col, probe_mix_col = st.columns(2)
    with duration_col:
        _render_compare_dashboard_chart(
            rows=duration_chart_rows,
            x_field="metric",
            y_field="value",
            title="Benchmark Duration",
            color_by_label={alias_by_label[label]: color for label, color in color_by_label.items()},
            height=280,
        )
    with probe_mix_col:
        _render_compare_dashboard_chart(
            rows=probe_mix_chart_rows,
            x_field="metric",
            y_field="value",
            title="Probe Mix",
            color_by_label={alias_by_label[label]: color for label, color in color_by_label.items()},
            height=280,
        )

    st.subheader("Branch Metadata")
    branch_rows: list[dict[str, Any]] = []
    for run_entry in selected_runs:
        metadata_rows = _read_csv_rows(path=Path(str(run_entry["metadata_path"])))
        for row in metadata_rows:
            branch_rows.append(dict(row))
    if branch_rows:
        st.dataframe(branch_rows, width="stretch")


def _render_runtime_estimate_preview(*, request: dict[str, Any]) -> None:
    """Render the pre-run benchmark runtime estimate from telemetry history."""
    try:
        estimate = estimate_benchmark_runtime(request=request)
    except Exception as exc:
        st.info(f"Expected runtime estimate unavailable: {exc}")
        return

    expected = float(estimate.get("expected_seconds", 0.0))
    low = float(estimate.get("low_seconds", 0.0))
    high = float(estimate.get("high_seconds", 0.0))
    st.caption(
        "Expected duration: "
        f"{expected:.1f}s (range {low:.1f}s to {high:.1f}s) based on recent benchmark telemetry."
    )


def _render_timing_summary(*, result: dict[str, Any]) -> None:
    """Render actual benchmark total and per-stage timing details."""
    timing = dict(result.get("timing", {}))
    if not timing:
        return

    st.subheader("Timing")
    total_seconds = timing.get("actual_total_seconds")
    if isinstance(total_seconds, (int, float)):
        st.metric("Actual total duration", f"{float(total_seconds):.2f} s")

    stage_seconds = timing.get("stages_seconds", {})
    if isinstance(stage_seconds, dict) and stage_seconds:
        rows = [
            {"stage": stage_name, "seconds": float(seconds)}
            for stage_name, seconds in stage_seconds.items()
            if isinstance(seconds, (int, float))
        ]
        if rows:
            st.dataframe(rows, width="stretch")


def _render_benchmark_tool_info() -> None:
    """Render a compact RAGAS explainer for benchmark users."""
    with st.expander("About RAGAS"):
        st.markdown("**Who made it**")
        st.markdown("- RAGAS was created by the Exploding Gradients team and open-source community.")
        st.markdown("- It is a retrieval-augmented generation evaluation framework focused on RAG-specific quality signals.")

        st.markdown("**Metrics and how they are calculated**")
        st.markdown("- RAGAS evaluates each benchmark case with LLM-based judges using the question, generated answer, expected answer, retrieved chunks, and reference contexts.")
        st.markdown("- `context_precision`: estimates how much of the retrieved context is relevant by judging whether returned chunks are useful rather than noisy.")
        st.markdown("- `context_recall`: estimates how much necessary support was retrieved by comparing retrieved chunks against the reference context or expected answer.")
        st.markdown("- `faithfulness`: checks whether statements in the generated answer are supported by the retrieved context.")
        st.markdown("- `answer_relevancy`: measures how directly the generated answer responds to the benchmark question.")
        st.markdown("- `response_relevancy`: measures how well the produced response aligns with the prompt and task.")
        st.markdown("- The app reports these as aggregate scores averaged across all evaluated benchmark cases.")


def render(show_title: bool = True) -> None:
    subpage = str(st.session_state.get("nav_subpage", "Run Benchmarks"))
    if subpage == "Compare Benchmarks":
        _render_compare_view(show_title=show_title)
        return

    _show_title(show_title)

    run_notice = st.session_state.pop(BENCHMARK_LAST_RUN_NOTICE_KEY, None)
    if isinstance(run_notice, dict):
        level = str(run_notice.get("level", "info"))
        message = str(run_notice.get("message", ""))
        if message:
            if level == "success":
                st.success(message)
            elif level == "error":
                st.error(message)
            else:
                st.info(message)

    is_benchmark_running = bool(st.session_state.get(BENCHMARK_RUN_IN_PROGRESS_KEY, False))

    corpora = _find_chunk_corpora()
    if not corpora:
        st.info("No chunk corpora found. Run ingestion first.")
        return

    corpus_labels = {_display_path(path): path for path in corpora}
    default_corpus = _display_path(corpora[0])
    selected_corpus_label = st.selectbox(
        "Corpus (chunk root)",
        options=list(corpus_labels.keys()),
        index=list(corpus_labels.keys()).index(default_corpus),
        disabled=is_benchmark_running,
    )
    selected_corpus = corpus_labels[selected_corpus_label]
    embedding_model_options = _benchmark_embedding_model_options()
    evaluation_model_options = _benchmark_evaluation_model_options()

    left_col, right_col = st.columns(2)
    with left_col:
        embedding_model = _render_fixed_model_picker(
            label="Embedding model",
            options=embedding_model_options,
            select_key="benchmark_embedding_model_select",
            disabled=is_benchmark_running,
        )
    with right_col:
        retrieval_model = _render_fixed_model_picker(
            label="Retrieval model",
            options=embedding_model_options,
            select_key="benchmark_retrieval_model_select",
            disabled=is_benchmark_running,
        )

    if not embedding_model or not retrieval_model:
        st.warning("Select both an embedding model and a retrieval model.")
        return

    selected_tools = list(DEFAULT_BENCHMARK_TOOLS)
    _render_benchmark_tool_info()

    verified_count, verified_count_error = _load_verified_question_count(verified_path=VERIFIED_QUESTIONS_PATH)
    if verified_count_error:
        st.warning(f"Verified probe source unavailable: {verified_count_error}")

    st.markdown("**Probe Sources**")
    source_col_1, source_col_2 = st.columns(2)
    with source_col_1:
        include_auto = st.checkbox(
            "Auto probes",
            value=True,
            key="benchmark_include_auto_probes",
            disabled=is_benchmark_running,
        )
    with source_col_2:
        include_verified = st.checkbox(
            f"Verified probes ({verified_count} available)",
            value=False,
            key="benchmark_include_verified_probes",
            disabled=verified_count <= 0 or is_benchmark_running,
            help="All available verified questions are included when enabled.",
        )

    if not include_auto and not include_verified:
        st.warning("Enable at least one probe source.")
        return

    st.markdown("**Retrieval Method**")
    selected_method = str(
        st.selectbox(
            "Retriever",
            options=list(RETRIEVAL_METHODS.keys()),
            index=list(RETRIEVAL_METHODS.keys()).index("faiss"),
            key="benchmark_retrieval_method_select",
            format_func=lambda method_id: RETRIEVAL_METHODS[method_id],
            disabled=is_benchmark_running,
        )
    ).strip().lower()
    if selected_method in {"graphrag", "lightrag"}:
        st.caption(
            "GraphRAG and LightRAG adapters are optional and may report planned/not-implemented failures."
        )

    control_col_1, control_col_2, control_col_3 = st.columns(3)
    with control_col_1:
        auto_probe_count = int(
            st.slider(
                "Auto probe cases",
                min_value=4,
                max_value=48,
                value=24,
                step=4,
                disabled=not include_auto or is_benchmark_running,
                help=METRIC_TOOLTIPS["Probe cases"],
            )
        )
    with control_col_2:
        top_k = int(
            st.slider(
                "Top-k retrieved chunks",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                disabled=is_benchmark_running,
                help=METRIC_TOOLTIPS["Top-k retrieved chunks"],
            )
        )
    with control_col_3:
        rebuild_index = st.checkbox("Rebuild index before run", value=False, disabled=is_benchmark_running)

    with st.expander("Advanced Evaluation Settings"):
        evaluation_model_raw = _render_model_picker(
            label="Evaluation model",
            options=evaluation_model_options,
            select_key="benchmark_evaluation_model_select",
            custom_key="benchmark_evaluation_model_custom",
            disabled=is_benchmark_running,
        )
        evaluation_model = _resolve_evaluation_model(raw_model=evaluation_model_raw)
        st.caption(
            "Embedding model builds the corpus index. Retrieval model embeds benchmark queries. "
            "Evaluation model is passed to RAGAS/DeepEval when supported."
        )

    suggested_output_dir = _index_output_dir(selected_corpus, embedding_model)
    discovered_faiss_dirs = _find_faiss_index_dirs()
    faiss_directory_options: list[str] = [str(suggested_output_dir)]
    for path in discovered_faiss_dirs:
        as_str = str(path)
        if as_str not in faiss_directory_options:
            faiss_directory_options.append(as_str)
    default_faiss_directory = str(
        st.session_state.get("benchmark_faiss_directory_select", str(suggested_output_dir))
    )
    if default_faiss_directory not in faiss_directory_options:
        default_faiss_directory = str(suggested_output_dir)
    selected_faiss_directory = st.selectbox(
        "FAISS directory",
        options=faiss_directory_options,
        index=faiss_directory_options.index(default_faiss_directory),
        key="benchmark_faiss_directory_select",
        disabled=is_benchmark_running,
    )
    output_dir = Path(selected_faiss_directory)
    st.caption(f"Index output: {_display_path(output_dir)}")

    estimate_request = _build_runtime_estimate_request(
        output_dir=output_dir,
        retrieval_model=retrieval_model,
        evaluation_model=evaluation_model,
        selected_tools=list(selected_tools),
        include_auto=include_auto,
        auto_probe_count=auto_probe_count if include_auto else None,
        include_verified=include_verified,
        verified_count=verified_count,
        retrieval_method=selected_method,
        top_k=top_k,
    )
    _render_runtime_estimate_preview(request=estimate_request)

    if is_benchmark_running:
        st.warning("Benchmark is currently running. Navigation is locked until it finishes.")

    run_clicked = st.button(
        "Run retrieval benchmarks",
        key="run_retrieval_benchmarks",
        width="stretch",
        disabled=is_benchmark_running,
    )

    if run_clicked:
        st.session_state[BENCHMARK_PENDING_RUN_REQUEST_KEY] = {
            "selected_corpus_path": str(selected_corpus),
            "selected_corpus_label": selected_corpus_label,
            "output_dir": str(output_dir),
            "embedding_model": embedding_model,
            "rebuild_index": rebuild_index,
            "retrieval_model": retrieval_model,
            "evaluation_model": evaluation_model,
            "auto_probe_count": auto_probe_count,
            "include_auto": include_auto,
            "include_verified": include_verified,
            "verified_count": verified_count,
            "selected_tools": list(selected_tools),
            "selected_method": selected_method,
            "top_k": top_k,
            "estimate_request": estimate_request,
            "ui_run_signature": _build_ui_run_signature(
                selected_corpus_label=selected_corpus_label,
                embedding_model=embedding_model,
                retrieval_model=retrieval_model,
                evaluation_model=evaluation_model,
                include_auto=include_auto,
                auto_probe_count=auto_probe_count,
                include_verified=include_verified,
                verified_count=verified_count,
                top_k=top_k,
                selected_tools=list(selected_tools),
                selected_method=selected_method,
            ),
        }
        st.session_state[BENCHMARK_RUN_IN_PROGRESS_KEY] = True
        st.session_state["nav_section"] = "Benchmarking"
        st.session_state["nav_subpage"] = "Run Benchmarks"
        st.rerun()

    pending_run_request = st.session_state.get(BENCHMARK_PENDING_RUN_REQUEST_KEY)
    if is_benchmark_running and isinstance(pending_run_request, dict):
        try:
            benchmark_result, index_result = _run_benchmark_with_loading_feedback(
                selected_corpus=Path(str(pending_run_request["selected_corpus_path"])),
                output_dir=Path(str(pending_run_request["output_dir"])),
                embedding_model=str(pending_run_request["embedding_model"]),
                rebuild_index=bool(pending_run_request["rebuild_index"]),
                retrieval_model=str(pending_run_request["retrieval_model"]),
                evaluation_model=(
                    None
                    if pending_run_request.get("evaluation_model") is None
                    else str(pending_run_request["evaluation_model"])
                ),
                auto_probe_count=int(pending_run_request["auto_probe_count"]),
                include_auto=bool(pending_run_request["include_auto"]),
                include_verified=bool(pending_run_request["include_verified"]),
                verified_count=int(pending_run_request["verified_count"]),
                selected_tools=list(pending_run_request["selected_tools"]),
                selected_method=str(pending_run_request["selected_method"]),
                top_k=int(pending_run_request["top_k"]),
                estimate_request=dict(pending_run_request["estimate_request"]),
            )
            benchmark_result["index_build"] = index_result
            benchmark_result["ui_run_signature"] = dict(pending_run_request["ui_run_signature"])
            benchmark_result["run_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            if _is_successful_benchmark_result(result=benchmark_result):
                benchmark_result["csv_export"] = _export_benchmark_csvs(result=benchmark_result)
            st.session_state["benchmark_result"] = benchmark_result
            set_benchmark_snapshot(snapshot=benchmark_result)
            st.session_state[BENCHMARK_LAST_RUN_NOTICE_KEY] = {
                "level": "success",
                "message": (
                    "Benchmark run completed and CSV exports were saved."
                    if "csv_export" in benchmark_result
                    else "Benchmark run completed."
                ),
            }
        except Exception as exc:
            st.session_state[BENCHMARK_LAST_RUN_NOTICE_KEY] = {
                "level": "error",
                "message": f"Benchmark run failed: {exc}",
            }
        finally:
            st.session_state[BENCHMARK_RUN_IN_PROGRESS_KEY] = False
            st.session_state.pop(BENCHMARK_PENDING_RUN_REQUEST_KEY, None)
            st.rerun()

    result = get_benchmark_snapshot()
    if result is None:
        legacy_result = st.session_state.get("benchmark_result")
        if isinstance(legacy_result, dict):
            set_benchmark_snapshot(snapshot=legacy_result)
            result = legacy_result
    if not result:
        st.info("Configure a run and click 'Run retrieval benchmarks' to generate results.")
        return

    expected_signature = _build_ui_run_signature(
        selected_corpus_label=selected_corpus_label,
        embedding_model=embedding_model,
        retrieval_model=retrieval_model,
        evaluation_model=evaluation_model,
        include_auto=include_auto,
        auto_probe_count=auto_probe_count,
        include_verified=include_verified,
        verified_count=verified_count,
        top_k=top_k,
        selected_tools=list(selected_tools),
        selected_method=selected_method,
    )
    if result.get("ui_run_signature") != expected_signature:
        st.info(
            "Current selections differ from the stored benchmark run. "
            "Showing the most recent stored snapshot; run again to refresh."
        )

    stored_signature = dict(result.get("ui_run_signature", {}))

    st.subheader("Run Summary")
    st.json(
        {
            "corpus": stored_signature.get("corpus", selected_corpus_label),
            "embedding_model": stored_signature.get("embedding_model", embedding_model),
            "retrieval_model": result.get("retrieval_model"),
            "evaluation_model": result.get("evaluation_model"),
            "probe_sources": {
                "auto_enabled": stored_signature.get("include_auto", True),
                "auto_probe_count": stored_signature.get("auto_probe_count"),
                "verified_enabled": stored_signature.get("include_verified", False),
                "verified_available": stored_signature.get("verified_available", verified_count),
            },
            "retrieval_methods": stored_signature.get(
                "retrieval_methods",
                result.get("retrieval_methods", [stored_signature.get("retrieval_method", "faiss")]),
            ),
            "tools": stored_signature.get("tools", list(result.get("tool_results", {}).keys())),
            "index_build": result.get("index_build", {}),
            "probe_source_breakdown": result.get("probe_source_breakdown", {}),
            "csv_export": result.get("csv_export", {}),
        }
    )
    _render_timing_summary(result=result)

    display_result = dict(result)
    source_method_options = _list_source_method_options(result)
    if source_method_options:
        option_labels = []
        for source_name, method_name in source_method_options:
            status = _method_status_for_slice(
                result=result,
                source=source_name,
                retrieval_method=method_name,
            )
            option_labels.append(f"{source_name} / {method_name} ({status})")
        default_index = _default_slice_index(
            result=result,
            source_method_options=source_method_options,
            prefer_verified=include_verified,
        )

        selected_label = st.selectbox(
            "Displayed evaluation slice (probe source / retrieval method)",
            options=option_labels,
            index=default_index,
            key="benchmark_display_slice",
        )
        selected_index = option_labels.index(selected_label)
        selected_source, selected_method = source_method_options[selected_index]
        display_result = _build_display_result_for_source_method(
            result=result,
            source=selected_source,
            retrieval_method=selected_method,
        )
        st.caption(
            "Showing metrics/cases for "
            f"`{selected_source}` probes with `{selected_method}` retrieval."
        )

    _render_execution_failure_details(result=result, display_result=display_result)
    _render_baseline_metrics(display_result)
    _render_tool_results(display_result)
    _render_case_chart(display_result)
