from __future__ import annotations

from benchmark.benchmark_tools.models import BenchmarkProbe, ChunkArtifact
from benchmark.benchmark_tools.probe_sources.base import ProbeContext, ProbeSource


def load_auto_probes(*, artifacts: list[ChunkArtifact], max_cases: int) -> list[BenchmarkProbe]:
    """Create evenly spaced probes from chunk artifacts."""
    if max_cases <= 0:
        raise ValueError(f"max_cases must be positive, got {max_cases}")
    if not artifacts:
        raise ValueError("artifacts must not be empty")

    if len(artifacts) <= max_cases:
        selected = artifacts
    else:
        step = max(len(artifacts) // max_cases, 1)
        selected = artifacts[::step][:max_cases]

    probes: list[BenchmarkProbe] = []
    for index, artifact in enumerate(selected, start=1):
        answer = _summarize_text(artifact.text, limit=240)
        excerpt = _summarize_text(artifact.text, limit=160)
        probes.append(
            BenchmarkProbe(
                case_id=f"probe_{index:03d}",
                query=(
                    "Retrieve the chunk that best matches this excerpt for grounded QA: "
                    f"{excerpt}"
                ),
                expected_chunk_id=artifact.chunk_id,
                expected_answer=answer,
                reference_contexts=[artifact.text],
            )
        )
    return probes


class AutoProbeSource(ProbeSource):
    """Load benchmark probes from auto-generated artifact excerpts."""

    def load_probes(self, *, context: ProbeContext) -> list[BenchmarkProbe]:
        raw_max_cases = context.policy.get("auto_probe_count", 24)
        try:
            max_cases = int(raw_max_cases)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid auto_probe_count '{raw_max_cases}'. Expected a positive integer."
            ) from exc
        return load_auto_probes(artifacts=context.artifacts, max_cases=max_cases)


def _summarize_text(text: str, *, limit: int) -> str:
    """Collapse whitespace and return a bounded snippet."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: max(limit - 3, 0)].rstrip()}..."
