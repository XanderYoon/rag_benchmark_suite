from __future__ import annotations

import streamlit as st

from Benchmark.domain.models import Chunk
from Benchmark.domain.models import BenchmarkRecord
from Benchmark.domain.models import EvidenceCandidate


def render_evidence_picker(
    record: BenchmarkRecord,
    chunks_by_id: dict[str, Chunk],
    key_prefix: str,
    candidates: list[EvidenceCandidate] | None = None,
) -> list[str]:
    defaults = set(record.gold_chunk_ids or record.candidate_gold_chunk_ids)
    selected: list[str] = []
    display_candidates = candidates if candidates is not None else record.retrieval_candidates

    st.write("Recommended chunks (check to include in gold chunks):")
    for cand in display_candidates:
        check_key = f"{key_prefix}_chk_{record.question_id}_{cand.chunk_id}"
        checked = st.checkbox(
            f"{cand.rank}. {cand.chunk_id} (score={cand.score:.3f})",
            value=cand.chunk_id in defaults,
            key=check_key,
        )
        chunk = chunks_by_id.get(cand.chunk_id)
        if chunk:
            st.caption(chunk.text)
        else:
            st.caption("Chunk text unavailable.")
        if checked:
            selected.append(cand.chunk_id)

    return selected
