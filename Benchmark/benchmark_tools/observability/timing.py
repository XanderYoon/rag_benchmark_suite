from __future__ import annotations

import statistics
import time
from typing import Callable, TypeVar


T = TypeVar("T")


def estimate_runtime(*, request: dict, history: list[dict]) -> dict[str, float]:
    """Estimate low/expected/high runtime seconds from request shape and recent history."""
    if not isinstance(request, dict):
        raise ValueError("Runtime estimate request must be a dictionary.")
    if not isinstance(history, list):
        raise ValueError("Runtime estimate history must be a list of telemetry records.")

    request_units = _compute_work_units(request=request)
    per_unit_samples = [
        sample
        for sample in (_extract_seconds_per_unit(entry=entry) for entry in history)
        if sample is not None
    ]
    seconds_per_unit = statistics.median(per_unit_samples) if per_unit_samples else 0.12

    expected = max(request_units * seconds_per_unit, 0.05)
    return {
        "low_seconds": max(expected * 0.7, 0.01),
        "expected_seconds": expected,
        "high_seconds": max(expected * 1.5, expected),
    }


def capture_stage_duration(*, stage_name: str, fn: Callable[[], T]) -> tuple[T, float]:
    """Run one stage callable and return its result with elapsed duration seconds."""
    if not str(stage_name).strip():
        raise ValueError("stage_name is required for timing capture.")

    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        raise RuntimeError(
            f"Failed while executing stage '{stage_name}': {type(exc).__name__}: {exc}"
        ) from exc
    duration = time.perf_counter() - started
    return result, max(duration, 0.0)


def _compute_work_units(*, request: dict) -> float:
    """Compute a coarse workload score for estimate scaling."""
    include_auto = bool(request.get("include_auto", True))
    include_verified = bool(request.get("include_verified", False))
    source_count = int(include_auto) + int(include_verified)
    if source_count <= 0:
        source_count = 1

    raw_methods = request.get("retrieval_methods", ["faiss"])
    method_count = len(_normalize_non_empty_list(value=raw_methods)) or 1

    tools_count = len(_normalize_non_empty_list(value=request.get("tools", [])))
    tools_multiplier = max(tools_count, 1)

    auto_probe_count = _to_positive_int_or_default(
        value=request.get("auto_probe_count"),
        default=24,
    )
    estimated_verified = _to_non_negative_int_or_default(
        value=request.get("estimated_verified_cases"),
        default=max(auto_probe_count // 2, 1) if include_verified else 0,
    )
    total_cases = (
        (auto_probe_count if include_auto else 0)
        + (estimated_verified if include_verified else 0)
    )
    total_cases = max(total_cases, 1)

    return float(source_count * method_count * tools_multiplier * total_cases)


def _extract_seconds_per_unit(*, entry: dict) -> float | None:
    """Extract telemetry seconds-per-unit sample from one historical telemetry row."""
    if not isinstance(entry, dict):
        return None

    actual_seconds = _extract_actual_seconds(entry=entry)
    if actual_seconds is None:
        return None

    request_summary = entry.get("request_summary", {})
    if not isinstance(request_summary, dict):
        request_summary = {}

    cases = _to_positive_int_or_default(value=request_summary.get("total_cases"), default=0)
    jobs = _to_positive_int_or_default(value=request_summary.get("job_count"), default=0)
    tools_multiplier = max(
        _to_positive_int_or_default(value=request_summary.get("tools_count"), default=1),
        1,
    )
    if cases <= 0:
        cases = _to_positive_int_or_default(value=entry.get("total_cases"), default=0)
    if jobs <= 0:
        jobs = _to_positive_int_or_default(value=entry.get("job_count"), default=0)

    work_units = cases * jobs * tools_multiplier
    if work_units <= 0:
        return None
    return actual_seconds / float(work_units)


def _extract_actual_seconds(*, entry: dict) -> float | None:
    """Read actual run seconds from supported telemetry shapes."""
    timing = entry.get("timing", {})
    if isinstance(timing, dict):
        raw_value = timing.get("actual_total_seconds")
    else:
        raw_value = None

    if raw_value is None:
        raw_value = entry.get("actual_total_seconds")

    try:
        actual = float(raw_value)
    except (TypeError, ValueError):
        return None
    if actual <= 0:
        return None
    return actual


def _normalize_non_empty_list(*, value: object) -> list[str]:
    """Normalize list-like value into unique non-empty lowercase strings."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in value:
        item = str(raw_item).strip().lower()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return normalized


def _to_positive_int_or_default(*, value: object, default: int) -> int:
    """Convert value to positive integer, otherwise return default."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _to_non_negative_int_or_default(*, value: object, default: int) -> int:
    """Convert value to non-negative integer, otherwise return default."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default
