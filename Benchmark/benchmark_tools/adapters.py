from __future__ import annotations

from inspect import signature
from typing import Any

from Benchmark.benchmark_tools.models import RetrievalCaseResult


def run_ragas_benchmark(
    *,
    results: list[RetrievalCaseResult],
    retrieval_model: str,
    evaluation_model: str | None = None,
) -> dict[str, Any]:
    """Run RAGAS over retrieval case results when the SDK is installed."""
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
        dataset = Dataset.from_list(rows)
        evaluate_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "metrics": resolved_metrics,
        }
        embeddings = _build_langchain_embeddings(model=retrieval_model)
        llm = _build_langchain_chat_model(model=evaluation_model) if evaluation_model else None
        if embeddings is not None:
            evaluate_kwargs["embeddings"] = embeddings
        if llm is not None:
            evaluate_kwargs["llm"] = llm

        evaluate_signature = signature(evaluate)
        if "raise_exceptions" in evaluate_signature.parameters:
            evaluate_kwargs["raise_exceptions"] = False
        if "show_progress" in evaluate_signature.parameters:
            evaluate_kwargs["show_progress"] = False

        score_result = evaluate(**evaluate_kwargs)
        parsed_scores = _extract_ragas_scores(score_result)
        return completed_tool_result(
            tool_name="ragas",
            summary=parsed_scores,
            details={"num_cases": len(rows)},
        )
    except Exception as exc:
        return failed_tool_result("ragas", exc)


def run_deepeval_benchmark(
    *,
    results: list[RetrievalCaseResult],
    evaluation_model: str | None = None,
) -> dict[str, Any]:
    """Run DeepEval metrics over retrieval case results when installed."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        return skipped_tool_result(
            tool_name="deepeval",
            message="DeepEval dependencies are missing. Install with: pip install deepeval",
            error=str(exc),
        )

    metric_defs = [
        ("answer_relevancy", AnswerRelevancyMetric),
        ("faithfulness", FaithfulnessMetric),
        ("contextual_precision", ContextualPrecisionMetric),
        ("contextual_recall", ContextualRecallMetric),
        ("contextual_relevancy", ContextualRelevancyMetric),
    ]

    aggregate_scores: dict[str, list[float]] = {name: [] for name, _ in metric_defs}
    metric_reasons: dict[str, list[str]] = {name: [] for name, _ in metric_defs}

    try:
        for result in results:
            test_case = _build_deepeval_test_case(LLMTestCase=LLMTestCase, result=result)
            for name, metric_cls in metric_defs:
                metric = _instantiate_metric(metric_cls, evaluation_model=evaluation_model)
                metric.measure(test_case)
                aggregate_scores[name].append(float(getattr(metric, "score", 0.0)))
                reason = getattr(metric, "reason", None)
                if reason:
                    metric_reasons[name].append(str(reason))

        summary = {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in aggregate_scores.items()
        }
        details = {
            "num_cases": len(results),
            "reasons": {
                name: reasons[:3]
                for name, reasons in metric_reasons.items()
                if reasons
            },
        }
        return completed_tool_result(tool_name="deepeval", summary=summary, details=details)
    except Exception as exc:
        return failed_tool_result("deepeval", exc)


def run_langsmith_benchmark(
    *,
    results: list[RetrievalCaseResult],
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Run a simple LangSmith evaluation flow when the SDK is installed."""
    try:
        from langsmith import Client
        from langsmith import evaluate as langsmith_evaluate
    except ImportError as exc:
        return skipped_tool_result(
            tool_name="langsmith",
            message="LangSmith dependencies are missing. Install with: pip install langsmith",
            error=str(exc),
        )

    payload = [
        {
            "inputs": {"question": result.query},
            "outputs": {
                "answer": result.actual_answer,
                "top_hit_chunk_id": result.top_hit_chunk_id,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in result.retrieved_chunks],
            },
            "reference_outputs": {
                "expected_chunk_id": result.expected_chunk_id,
                "expected_answer": result.expected_answer,
            },
        }
        for result in results
    ]

    summary = {
        "num_cases": len(results),
        "hit_at_1": sum(1 for result in results if result.hit_at_1) / len(results) if results else 0.0,
        "hit_at_3": sum(1 for result in results if result.hit_at_3) / len(results) if results else 0.0,
    }

    try:
        client = Client()
        evaluate_method = langsmith_evaluate if callable(langsmith_evaluate) else getattr(client, "evaluate", None)
        if callable(evaluate_method):
            def target(example: dict[str, Any]) -> dict[str, Any]:
                return dict(example.get("outputs", {}))

            def retrieval_hit_evaluator(run: Any, example: Any) -> dict[str, Any]:
                predicted = None
                expected = None

                if hasattr(run, "outputs") and isinstance(run.outputs, dict):
                    predicted = run.outputs.get("top_hit_chunk_id")
                elif isinstance(run, dict):
                    predicted = run.get("outputs", {}).get("top_hit_chunk_id")

                if hasattr(example, "reference_outputs") and isinstance(example.reference_outputs, dict):
                    expected = example.reference_outputs.get("expected_chunk_id")
                elif isinstance(example, dict):
                    expected = example.get("reference_outputs", {}).get("expected_chunk_id")

                return {
                    "key": "retrieval_hit",
                    "score": 1.0 if predicted == expected and expected is not None else 0.0,
                }

            evaluate_kwargs: dict[str, Any] = {
                "data": payload,
                "evaluators": [retrieval_hit_evaluator],
            }
            if experiment_name:
                if "experiment_prefix" in signature(evaluate_method).parameters:
                    evaluate_kwargs["experiment_prefix"] = experiment_name
                elif "experiment_name" in signature(evaluate_method).parameters:
                    evaluate_kwargs["experiment_name"] = experiment_name

            experiment_result = evaluate_method(target, **evaluate_kwargs)
            details = {
                "num_cases": len(results),
                "experiment_name": experiment_name,
                "langsmith_result_type": type(experiment_result).__name__,
            }
            return completed_tool_result("langsmith", summary=summary, details=details)

        return completed_tool_result(
            tool_name="langsmith",
            summary=summary,
            details={
                "num_cases": len(results),
                "message": "LangSmith client is installed, but no evaluate method was available. "
                "Returned local compatibility summary only.",
            },
        )
    except Exception as exc:
        return failed_tool_result("langsmith", exc, summary=summary, details={"num_cases": len(results)})


def _build_langchain_embeddings(model: str) -> Any | None:
    """Build a LangChain OpenAI embeddings wrapper when available."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        return None
    return OpenAIEmbeddings(model=model)


def _build_langchain_chat_model(model: str) -> Any | None:
    """Build a LangChain OpenAI chat wrapper when available."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None
    return ChatOpenAI(model=model)


def _instantiate_metric(metric_cls: type, evaluation_model: str | None) -> Any:
    """Instantiate a DeepEval metric with optional model support."""
    kwargs: dict[str, Any] = {"threshold": 0.0}
    if evaluation_model:
        params = signature(metric_cls).parameters
        if "model" in params:
            kwargs["model"] = evaluation_model
        elif "evaluation_model" in params:
            kwargs["evaluation_model"] = evaluation_model
    return metric_cls(**kwargs)


def _build_deepeval_test_case(LLMTestCase: type, result: RetrievalCaseResult) -> Any:
    """Instantiate an LLMTestCase with only supported keyword arguments."""
    available_fields = signature(LLMTestCase).parameters
    kwargs: dict[str, Any] = {}
    if "input" in available_fields:
        kwargs["input"] = result.query
    if "actual_output" in available_fields:
        kwargs["actual_output"] = result.actual_answer
    if "expected_output" in available_fields:
        kwargs["expected_output"] = result.expected_answer
    if "context" in available_fields:
        kwargs["context"] = list(result.reference_contexts)
    if "retrieval_context" in available_fields:
        kwargs["retrieval_context"] = [chunk.text for chunk in result.retrieved_chunks]
    return LLMTestCase(**kwargs)


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
    return payload
