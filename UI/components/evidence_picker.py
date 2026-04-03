from __future__ import annotations

import html

import streamlit as st

from benchmark.domain.models import Chunk
from benchmark.domain.models import BenchmarkRecord
from benchmark.domain.models import EvidenceCandidate


def render_evidence_picker(
    record: BenchmarkRecord,
    chunks_by_id: dict[str, Chunk],
    key_prefix: str,
    candidates: list[EvidenceCandidate] | None = None,
    show_scores: bool = True,
) -> list[str]:
    defaults = set(record.gold_chunk_ids or record.candidate_gold_chunk_ids)
    selected: list[str] = []
    display_candidates = candidates if candidates is not None else record.retrieval_candidates

    st.write("Recommended chunks (check to include in gold chunks):")
    for cand in display_candidates:
        check_key = f"{key_prefix}_chk_{record.question_id}_{cand.chunk_id}"
        label = f"{cand.rank}. {cand.chunk_id} (score={cand.score:.3f})"
        if not show_scores:
            label = f"{cand.rank}. {cand.chunk_id}"
        checked = st.checkbox(
            label,
            value=cand.chunk_id in defaults,
            key=check_key,
        )
        chunk = chunks_by_id.get(cand.chunk_id)
        if chunk:
            snippet = html.escape(chunk.text)
            st.markdown(
                (
                    "<div style='padding-top: 0.2rem; padding-bottom: 0.75rem; color: #2b2b2b; "
                    f"font-size: 0.88rem; line-height: 1.35;'>{snippet}</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                (
                    "<div style='padding-top: 0.2rem; padding-bottom: 0.75rem; color: #4a4a4a; "
                    "font-size: 0.88rem;'>Chunk text unavailable.</div>"
                ),
                unsafe_allow_html=True,
            )
        if checked:
            selected.append(cand.chunk_id)

    return selected
