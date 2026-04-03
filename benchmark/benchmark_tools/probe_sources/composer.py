from __future__ import annotations

from benchmark.benchmark_tools.models import BenchmarkProbe
from benchmark.benchmark_tools.probe_sources.auto_source import AutoProbeSource
from benchmark.benchmark_tools.probe_sources.base import ProbeContext
from benchmark.benchmark_tools.probe_sources.verified_source import VerifiedQuestionProbeSource


def build_probe_buckets(
    *,
    policy: dict,
    context: ProbeContext,
) -> dict[str, list[BenchmarkProbe]]:
    """Build source-separated probe buckets from the normalized selection policy."""
    include_auto = bool(
        policy.get("include_auto_probes", policy.get("include_auto", False))
    )
    include_verified = bool(
        policy.get("include_verified_probes", policy.get("include_verified", False))
    )

    buckets: dict[str, list[BenchmarkProbe]] = {}
    source_load_errors: list[str] = []
    if include_auto:
        try:
            buckets["auto"] = AutoProbeSource().load_probes(context=context)
        except Exception as exc:
            source_load_errors.append(f"auto: {type(exc).__name__}: {exc}")
    if include_verified:
        try:
            buckets["verified"] = VerifiedQuestionProbeSource().load_probes(context=context)
        except Exception as exc:
            source_load_errors.append(f"verified: {type(exc).__name__}: {exc}")

    if not buckets:
        if source_load_errors:
            raise ValueError(
                "Probe bucket composition failed for all enabled sources. "
                f"Errors: {source_load_errors}"
            )
        raise ValueError(
            "Probe bucket composition produced no sources. Enable include_auto and/or include_verified."
        )
    return buckets
