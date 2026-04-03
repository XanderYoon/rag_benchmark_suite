from __future__ import annotations

import os
import time
import traceback
from inspect import signature
from typing import Any

from benchmark.benchmark_tools.models import RetrievalCaseResult
from benchmark.benchmark_tools.observability import append_debug_event


def run_ragas_benchmark(
    *,
    results: list[RetrievalCaseResult],
    retrieval_model: str,
    evaluation_model: str | None = None,
) -> dict[str, Any]:
    """Run RAGAS over retrieval case results when the SDK is installed."""
    debug_context = _build_ragas_debug_context(
        results=results,
        retrieval_model=retrieval_model,
        evaluation_model=evaluation_model,
    )
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas import metrics as ragas_metrics
    except ImportError as exc:
        return skipped_tool_result(
            tool_name="ragas",
            message="RAGAS dependencies are missing. Install with: pip install ragas datasets",
            error=str(exc),
        )

    if _is_embedding_model_name(model_name=evaluation_model):
        return skipped_tool_result(
            tool_name="ragas",
            message=(
                "RAGAS requires an LLM-style evaluation model. "
                f"Received embedding model '{evaluation_model}'."
            ),
            error="Set evaluation_model to a chat/completion model (for example, gpt-4o-mini).",
        )

    rows = []
    for result in results:
        rows.append(
            {
                "question": result.query,
                "answer": result.actual_answer,
                "ground_truth": result.expected_answer,
                "reference": result.expected_answer,
                "contexts": [chunk.text for chunk in result.retrieved_chunks],
                "reference_contexts": list(result.reference_contexts),
            }
        )

    if not rows:
        return completed_tool_result(tool_name="ragas", summary={"num_cases": 0}, details={})

    metric_names = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
    ]
    resolved_metrics = []
    seen_names: set[str] = set()
    for name in metric_names:
        metric = getattr(ragas_metrics, name, None)
        if metric is None or name in seen_names:
            continue
        resolved_metrics.append(_configure_ragas_metric(metric))
        seen_names.add(name)
    if not resolved_metrics:
        return skipped_tool_result(
            tool_name="ragas",
            message="No supported RAGAS metrics were available in the installed package.",
            error="Unable to resolve context_precision/context_recall/faithfulness/relevancy metrics.",
        )

    try:
        debug_context["metric_names"] = list(metric_names)
        dataset = Dataset.from_list(rows)
        evaluate_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "metrics": resolved_metrics,
        }
        embeddings = _build_langchain_embeddings(model=retrieval_model)
        llm = _build_langchain_chat_model(model=evaluation_model) if evaluation_model else None
        if embeddings is None:
            backend_error = _embeddings_backend_error(model=retrieval_model)
            return skipped_tool_result(
                tool_name="ragas",
                message=f"RAGAS embeddings backend is not available. {backend_error}",
                error=backend_error,
            )
        if llm is None:
            backend_error = _chat_backend_error(model=evaluation_model)
            return skipped_tool_result(
                tool_name="ragas",
                message=(
                    "RAGAS evaluation LLM could not be initialized from advanced settings. "
                    f"Configured evaluation_model='{evaluation_model}'. {backend_error}"
                ),
                error=backend_error,
            )

        evaluate_kwargs["embeddings"] = embeddings
        evaluate_kwargs["llm"] = llm
        debug_context["chat_model_kwargs"] = _safe_json_value(
            _build_chat_model_debug_context(model=evaluation_model or "")
        )
        debug_context["started_at_unix"] = time.time()

        evaluate_signature = signature(evaluate)
        if "raise_exceptions" in evaluate_signature.parameters:
            evaluate_kwargs["raise_exceptions"] = False
        if "show_progress" in evaluate_signature.parameters:
            evaluate_kwargs["show_progress"] = False

        score_result = evaluate(**evaluate_kwargs)
        parsed_scores = _extract_ragas_scores(score_result)
        case_scores = _extract_ragas_case_scores(score_result=score_result, results=results)
        debug_file = append_debug_event(
            event_type="ragas_completed",
            payload={
                **debug_context,
                "summary": _safe_json_value(parsed_scores),
                "case_scores_count": len(case_scores),
            },
        )
        return completed_tool_result(
            tool_name="ragas",
            summary=parsed_scores,
            details={
                "num_cases": len(rows),
                "case_scores": case_scores,
                "debug_log_file": str(debug_file),
                "debug_context": debug_context,
            },
        )
    except Exception as exc:
        debug_file = append_debug_event(
            event_type="ragas_failed",
            payload={
                **debug_context,
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            },
        )
        return failed_tool_result(
            "ragas",
            exc,
            details={
                "debug_log_file": str(debug_file),
                "debug_context": debug_context,
            },
        )


def _build_langchain_embeddings(model: str) -> Any | None:
    """Build a LangChain embeddings wrapper for OpenAI or Ollama."""
    if _is_openai_embedding_model(model_name=model):
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            return None
        return OpenAIEmbeddings(model=model)

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        return None
    return OllamaEmbeddings(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


def _build_langchain_chat_model(model: str) -> Any | None:
    """Build a LangChain chat wrapper for OpenAI or Ollama."""
    if _is_embedding_model_name(model_name=model):
        return None
    if _is_openai_chat_model(model_name=model):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None
        return ChatOpenAI(model=model)

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return None
    return ChatOllama(**_build_ollama_chat_kwargs(model=model))


def _build_ollama_chat_kwargs(*, model: str) -> dict[str, Any]:
    """Build Ollama chat kwargs with stable defaults for evaluator use."""
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": 0,
    }
    normalized_model = str(model).strip().lower()
    if normalized_model.startswith("qwen"):
        # Disable reasoning output so RAGAS gets concise judge responses.
        kwargs["reasoning"] = False
    return kwargs


def _build_chat_model_debug_context(*, model: str) -> dict[str, Any]:
    """Return non-sensitive chat-model config used for benchmark evaluation."""
    if _is_openai_chat_model(model_name=model):
        return {
            "provider": "openai",
            "model": model,
        }
    if not model:
        return {"provider": "unknown", "model": ""}
    return {
        "provider": "ollama",
        "model": model,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": 0,
        "reasoning": False if str(model).strip().lower().startswith("qwen") else None,
    }


def _build_ragas_debug_context(
    *,
    results: list[RetrievalCaseResult],
    retrieval_model: str,
    evaluation_model: str | None,
) -> dict[str, Any]:
    """Build a structured debug context for one RAGAS adapter invocation."""
    sample_case_ids = [result.case_id for result in results[:3]]
    sample_queries = [result.query for result in results[:2]]
    return {
        "tool": "ragas",
        "retrieval_model": retrieval_model,
        "evaluation_model": str(evaluation_model or ""),
        "retrieval_case_count": len(results),
        "sample_case_ids": sample_case_ids,
        "sample_queries": sample_queries,
    }


def _safe_json_value(value: Any) -> Any:
    """Convert nested values into JSON-serializable debug payloads."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _safe_json_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return str(value)


def _embeddings_backend_error(*, model: str) -> str:
    """Return actionable setup guidance for retrieval embeddings backend."""
    if _is_openai_embedding_model(model_name=model):
        return (
            "Configured retrieval_model uses OpenAI embeddings. "
            "Install langchain-openai and set OPENAI_API_KEY."
        )
    return (
        "Configured retrieval_model appears to be Ollama/local embeddings. "
        "Install langchain-ollama and ensure OLLAMA_BASE_URL is reachable."
    )


def _chat_backend_error(*, model: str | None) -> str:
    """Return actionable setup guidance for evaluation chat backend."""
    if not model:
        return "Set evaluation_model under Advanced Evaluation Settings."
    normalized = str(model).strip().lower()
    if _is_openai_chat_model(model_name=normalized):
        return (
            f"Configured evaluation_model='{model}' requires OpenAI chat backend. "
            "Install langchain-openai and set OPENAI_API_KEY."
        )
    return (
        f"Configured evaluation_model='{model}' requires Ollama chat backend. "
        "Install langchain-ollama and ensure OLLAMA_BASE_URL is reachable."
    )


def _is_embedding_model_name(*, model_name: str | None) -> bool:
    """Return True when model name appears to be an embeddings-only model id."""
    if not model_name:
        return False
    lowered = str(model_name).strip().lower()
    return lowered.startswith("text-embedding-") or "embed" in lowered


def _is_openai_embedding_model(*, model_name: str | None) -> bool:
    """Return True when the embedding model id belongs to OpenAI."""
    if not model_name:
        return False
    return str(model_name).strip().lower().startswith("text-embedding-")


def _is_openai_chat_model(*, model_name: str | None) -> bool:
    """Return True when the chat model id belongs to OpenAI."""
    if not model_name:
        return False
    return str(model_name).strip().lower().startswith("gpt-")


def _extract_ragas_scores(score_result: Any) -> dict[str, float]:
    """Normalize RAGAS output into a flat numeric dictionary."""
    if isinstance(score_result, dict):
        return {
            str(key): float(value)
            for key, value in score_result.items()
            if isinstance(value, (int, float))
        }

    to_pandas = getattr(score_result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if hasattr(frame, "columns") and len(frame.index) > 0:
            numeric_scores: dict[str, float] = {}
            for column in frame.columns:
                series = frame[column]
                try:
                    numeric_scores[str(column)] = float(series.mean())
                except Exception:
                    continue
            if numeric_scores:
                return numeric_scores

    repr_dict = getattr(score_result, "_repr_dict", None)
    if isinstance(repr_dict, dict):
        return {
            str(key): float(value)
            for key, value in repr_dict.items()
            if isinstance(value, (int, float))
        }

    return {}


def _extract_ragas_case_scores(
    *,
    score_result: Any,
    results: list[RetrievalCaseResult],
) -> list[dict[str, Any]]:
    """Extract per-case RAGAS metric rows aligned to benchmark case ids."""
    to_pandas = getattr(score_result, "to_pandas", None)
    if not callable(to_pandas):
        return []

    try:
        frame = to_pandas()
    except Exception:
        return []

    frame_columns = getattr(frame, "columns", None)
    if frame_columns is None:
        return []

    metric_names = (
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
    )
    available_metrics = [name for name in metric_names if name in frame_columns]
    if not available_metrics:
        return []

    case_scores: list[dict[str, Any]] = []
    for index, case_result in enumerate(results):
        if index >= len(frame.index):
            break
        row_payload: dict[str, Any] = {"case_id": case_result.case_id}
        has_metric = False
        for metric_name in available_metrics:
            try:
                metric_value = frame.iloc[index][metric_name]
            except Exception:
                continue
            if isinstance(metric_value, (int, float)):
                row_payload[metric_name] = float(metric_value)
                has_metric = True
        if has_metric:
            case_scores.append(row_payload)
    return case_scores


def _configure_ragas_metric(metric: Any) -> Any:
    """Reduce RAGAS sampling where supported to avoid repeated multi-generation warnings."""
    for attribute_name in ("strictness", "n", "_n"):
        if hasattr(metric, attribute_name):
            try:
                setattr(metric, attribute_name, 1)
            except Exception:
                continue
    return metric


def completed_tool_result(
    tool_name: str,
    summary: dict[str, Any],
    details: dict[str, Any],
) -> dict[str, Any]:
    """Return a normalized successful tool result."""
    return {
        "tool": tool_name,
        "status": "completed",
        "summary": summary,
        "details": details,
    }


def skipped_tool_result(
    *,
    tool_name: str,
    message: str,
    error: str,
) -> dict[str, Any]:
    """Return a normalized skipped tool result."""
    return {
        "tool": tool_name,
        "status": "skipped",
        "summary": {},
        "details": {
            "message": message,
            "error": error,
        },
    }


def failed_tool_result(
    tool_name: str,
    exc: Exception,
    *,
    summary: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a normalized failed tool result."""
    payload = {
        "tool": tool_name,
        "status": "failed",
        "summary": summary or {},
        "details": details or {},
    }
    payload["details"]["error"] = f"{type(exc).__name__}: {exc}"
    payload["details"]["error_type"] = type(exc).__name__
    payload["details"]["error_message"] = str(exc)
    payload["details"]["traceback"] = traceback.format_exc()
    return payload
