from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Benchmark.benchmark_tools.models import BenchmarkProbe, ChunkArtifact


@dataclass(frozen=True)
class ProbeContext:
    """Carry shared inputs needed by probe sources."""

    artifacts: list[ChunkArtifact]
    chunk_lookup: dict[str, str]
    verified_path: Path
    policy: dict[str, Any]


class ProbeSource(ABC):
    """Define the contract implemented by benchmark probe sources."""

    @abstractmethod
    def load_probes(self, *, context: ProbeContext) -> list[BenchmarkProbe]:
        """Load probes from one source using shared run context."""


def load_probes(*, source: ProbeSource, context: ProbeContext) -> list[BenchmarkProbe]:
    """Load probes through the shared probe-source interface contract."""
    return source.load_probes(context=context)
