from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from Benchmark.benchmark_tools import run_retrieval_benchmarks
from Benchmark.embedding.build_faiss_rag_index import build_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
COMMON_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
COMMON_EVALUATION_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
]
BENCHMARK_TOOLS = ["ragas", "deepeval", "langsmith"]
METRIC_TOOLTIPS = {
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
    "contextual_precision": "DeepEval score for how much retrieved context is relevant rather than noisy.",
    "contextual_recall": "DeepEval score for how much required supporting context was retrieved.",
    "contextual_relevancy": "DeepEval score for overall usefulness of retrieved context to answer the query.",
    "retrieval_hit": "Binary evaluator score for whether the expected chunk matched the predicted top hit.",
}


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


def _render_model_picker(
    *,
    label: str,
    options: list[str],
    select_key: str,
    custom_key: str,
) -> str:
    """Render a preset-or-custom model picker."""
    select_options = [*options, "Custom"]
    selected = st.selectbox(label, options=select_options, key=select_key)
    if selected == "Custom":
        custom_value = st.text_input(
            f"{label} (custom)",
            value=str(st.session_state.get(custom_key, "")),
            key=custom_key,
        ).strip()
        return custom_value

    st.session_state[custom_key] = selected
    return selected


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
        result = build_faiss_index(
            chunks_root=corpus_root,
            output_dir=output_dir,
            embedding_model=embedding_model,
            batch_size=64,
            metric="cosine",
            overwrite=True,
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

    st.subheader("Baseline Retrieval")
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
    st.vega_lite_chart(
        {
            "data": {"values": chart_rows},
            "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
            "encoding": {
                "x": {"field": "metric", "type": "nominal", "axis": {"labelAngle": 0}},
                "y": {"field": "value", "type": "quantitative", "scale": {"domain": [0, 1]}},
                "color": {"value": "#1f77b4"},
            },
            "height": 260,
        },
        use_container_width=True,
    )
    _render_metric_guide(["Hit@1", "Hit@3", "MRR", "Avg Top Score"])


def _render_tool_results(result: dict[str, Any]) -> None:
    """Render per-tool benchmark summaries and status messages."""
    tool_results = dict(result.get("tool_results", {}))
    if not tool_results:
        return

    st.subheader("Tool Benchmarks")
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
            details = dict(tool_result.get("details", {}))
            if details:
                st.caption(str(details))
        elif status == "skipped":
            st.warning(f"{title}: {tool_result.get('details', {}).get('message', 'Skipped.')}")
        else:
            st.error(f"{title}: {tool_result.get('details', {}).get('error', 'Benchmark failed.')}")

    if summary_rows:
        st.vega_lite_chart(
            {
                "data": {"values": summary_rows},
                "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
                "encoding": {
                    "x": {"field": "metric", "type": "nominal", "axis": {"labelAngle": 0}},
                    "y": {"field": "value", "type": "quantitative"},
                    "xOffset": {"field": "tool"},
                    "color": {"field": "tool", "type": "nominal"},
                },
                "height": 300,
            },
            use_container_width=True,
        )


def _render_case_chart(result: dict[str, Any]) -> None:
    """Render a chart for per-case reciprocal rank and top-hit behavior."""
    cases = list(result.get("cases", []))
    if not cases:
        return

    st.subheader("Per-Case Performance")
    chart_rows = [
        {
            "case_id": str(case.get("case_id", "")),
            "reciprocal_rank": float(case.get("reciprocal_rank", 0.0)),
            "hit_at_1": 1.0 if bool(case.get("hit_at_1")) else 0.0,
        }
        for case in cases
    ]

    st.vega_lite_chart(
        {
            "data": {"values": chart_rows},
            "layer": [
                {
                    "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                    "encoding": {
                        "x": {"field": "case_id", "type": "nominal", "sort": None, "axis": {"labelAngle": -35}},
                        "y": {
                            "field": "reciprocal_rank",
                            "type": "quantitative",
                            "scale": {"domain": [0, 1]},
                        },
                        "color": {"value": "#4c78a8"},
                    },
                },
                {
                    "mark": {"type": "line", "strokeWidth": 2, "point": True},
                    "encoding": {
                        "x": {"field": "case_id", "type": "nominal", "sort": None},
                        "y": {"field": "hit_at_1", "type": "quantitative", "scale": {"domain": [0, 1]}},
                        "color": {"value": "#f58518"},
                    },
                },
            ],
            "height": 320,
        },
        use_container_width=True,
    )

    with st.expander("Benchmark Case Details"):
        st.dataframe(
            [
                {
                    "case_id": case.get("case_id"),
                    "expected_chunk_id": case.get("expected_chunk_id"),
                    "top_hit_chunk_id": case.get("top_hit_chunk_id"),
                    "hit_at_1": case.get("hit_at_1"),
                    "hit_at_3": case.get("hit_at_3"),
                    "reciprocal_rank": case.get("reciprocal_rank"),
                }
                for case in cases
            ],
            use_container_width=True,
        )


def _format_score(value: Any) -> str:
    """Format numeric scores for metric display."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


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


def render(show_title: bool = True) -> None:
    _show_title(show_title)

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
    )
    selected_corpus = corpus_labels[selected_corpus_label]

    left_col, right_col = st.columns(2)
    with left_col:
        embedding_model = _render_model_picker(
            label="Embedding model",
            options=COMMON_EMBEDDING_MODELS,
            select_key="benchmark_embedding_model_select",
            custom_key="benchmark_embedding_model_custom",
        )
    with right_col:
        retrieval_model = _render_model_picker(
            label="Retrieval model",
            options=COMMON_EMBEDDING_MODELS,
            select_key="benchmark_retrieval_model_select",
            custom_key="benchmark_retrieval_model_custom",
        )

    if not embedding_model or not retrieval_model:
        st.warning("Select both an embedding model and a retrieval model.")
        return

    selected_tools = st.multiselect(
        "Benchmark tools",
        options=BENCHMARK_TOOLS,
        default=BENCHMARK_TOOLS,
        help="Choose one or more benchmarking frameworks to run.",
    )
    if not selected_tools:
        st.info("Select at least one benchmarking tool to run.")
        return

    control_col_1, control_col_2, control_col_3 = st.columns(3)
    with control_col_1:
        max_cases = int(st.slider("Probe cases", min_value=4, max_value=48, value=24, step=4))
    with control_col_2:
        top_k = int(st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=5, step=1))
    with control_col_3:
        rebuild_index = st.checkbox("Rebuild index before run", value=False)

    with st.expander("Advanced Evaluation Settings"):
        evaluation_model = _render_model_picker(
            label="Evaluation model",
            options=COMMON_EVALUATION_MODELS,
            select_key="benchmark_evaluation_model_select",
            custom_key="benchmark_evaluation_model_custom",
        )
        st.caption(
            "Embedding model builds the corpus index. Retrieval model embeds benchmark queries. "
            "Evaluation model is passed to RAGAS/DeepEval when supported."
        )

    output_dir = _index_output_dir(selected_corpus, embedding_model)
    st.caption(f"Index output: {_display_path(output_dir)}")

    if st.button("Run retrieval benchmarks", key="run_retrieval_benchmarks", use_container_width=True):
        try:
            with st.spinner("Preparing FAISS index..."):
                index_result = _render_index_build(
                    corpus_root=selected_corpus,
                    output_dir=output_dir,
                    embedding_model=embedding_model,
                    rebuild_index=rebuild_index,
                )

            with st.spinner("Running benchmark tools..."):
                benchmark_result = run_retrieval_benchmarks(
                    embedded_chunks_path=output_dir,
                    retrieval_model=retrieval_model,
                    evaluation_model=evaluation_model,
                    max_cases=max_cases,
                    top_k=top_k,
                    tools=selected_tools,
                )
            benchmark_result["index_build"] = index_result
            benchmark_result["ui_run_signature"] = {
                "corpus": selected_corpus_label,
                "embedding_model": embedding_model,
                "retrieval_model": retrieval_model,
                "evaluation_model": evaluation_model,
                "max_cases": max_cases,
                "top_k": top_k,
                "tools": list(selected_tools),
            }
            st.session_state["benchmark_result"] = benchmark_result
            st.success("Benchmark run completed.")
        except Exception as exc:
            st.error(f"Benchmark run failed: {exc}")
            return

    result = st.session_state.get("benchmark_result")
    if not result:
        st.info("Configure a run and click 'Run retrieval benchmarks' to generate results.")
        return

    expected_signature = {
        "corpus": selected_corpus_label,
        "embedding_model": embedding_model,
        "retrieval_model": retrieval_model,
        "evaluation_model": evaluation_model,
        "max_cases": max_cases,
        "top_k": top_k,
        "tools": list(selected_tools),
    }
    if result.get("ui_run_signature") != expected_signature:
        st.info("Current selections differ from the stored benchmark run. Run again to refresh the charts.")
        return

    st.subheader("Run Summary")
    st.json(
        {
            "corpus": selected_corpus_label,
            "embedding_model": embedding_model,
            "retrieval_model": result.get("retrieval_model"),
            "evaluation_model": result.get("evaluation_model"),
            "tools": list(result.get("tool_results", {}).keys()),
            "index_build": result.get("index_build", {}),
        }
    )
    _render_baseline_metrics(result)
    _render_tool_results(result)
    _render_case_chart(result)
